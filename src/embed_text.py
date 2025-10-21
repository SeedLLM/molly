import os
from argparse import ArgumentParser
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
import torch
from cuml import UMAP
from cuml.preprocessing import StandardScaler
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoModelForMaskedLM

from train import setup_dataset, setup_tokenizers
from utils import get_current_device

import multiprocessing as mp
mp.set_start_method('spawn', force=True)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--text-model-path", required=True)
    parser.add_argument("--dna-rna-model-path", required=True)
    parser.add_argument(
        "--dna-rna-k-tokens",
        type=int,
        default=64,
        help="Number of tokens for DNA sequence projection",
    )
    parser.add_argument("--protein-model-path", required=True)
    parser.add_argument(
        "--protein-k-tokens",
        type=int,
        default=64,
        help="Number of tokens for protein sequence projection",
    )
    parser.add_argument("--train-dataset-path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--max-src-len", type=int, default=1024)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--read-nums", type=int, default=None)
    parser.add_argument("--mode", type=str, default="sft")
    parser.add_argument(
        "--skip-eval", action="store_true", help="Skip evaluation during training"
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=64,
        help="Batch size when computing embeddings",
    )
    return parser.parse_args()


def setup_models(args):
    text_model = AutoModel.from_pretrained(
        args.text_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # device_map=current_device,      # 删掉
    ).cuda()

    dna_rna_model = AutoModelForMaskedLM.from_pretrained(
        args.dna_rna_model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        # device_map=current_device,      # 删掉
    ).cuda()

    protein_model = AutoModelForMaskedLM.from_pretrained(
        args.protein_model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        # device_map=current_device,      # 删掉
    ).cuda()

    # --- 新增：包 DataParallel ---
    text_model     = torch.nn.DataParallel(text_model)
    dna_rna_model  = torch.nn.DataParallel(dna_rna_model)
    protein_model  = torch.nn.DataParallel(protein_model)

    return text_model, dna_rna_model, protein_model


@torch.no_grad()
def encode_text(model, input_ids, attention_mask):
    def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]

    outputs = model(input_ids, attention_mask=attention_mask)
    embeddings = last_token_pool(outputs.last_hidden_state, attention_mask)

    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings


@torch.no_grad()
def encode_omics_batch(model, ids, mask):
    """批量处理omics序列编码"""
    # 确保输入ID保持整数类型（嵌入层需要整数输入）
    # 确保掩码是浮点数类型（注意力计算需要浮点数）
    mask = mask.float()
    
    out = model(ids, attention_mask=mask, output_hidden_states=True)
    hidden = out.hidden_states[-1]  # [B, L, D]
    
    # 使用注意力掩码进行加权平均
    mask_expanded = mask.unsqueeze(-1).to(hidden.dtype)
    weighted_sum = (hidden * mask_expanded).sum(dim=1)
    seq_lengths = mask_expanded.sum(dim=1).clamp(min=1e-9)
    emb = weighted_sum / seq_lengths
    
    # 将输出转换为bfloat16以匹配文本嵌入
    return emb.to(torch.bfloat16)


@torch.no_grad()
def embed_text_batch(args, text_model, dna_rna_model, protein_model, batch):
    device = text_model.module.device
    B = batch["input_ids"].size(0)

    # 1. 文本嵌入
    text_emb = encode_text(
        text_model, 
        batch["input_ids"].to(device, non_blocking=True), 
        batch["attention_mask"].to(device, non_blocking=True)
    )  # [B, D_text]

    # 2. 处理omics数据
    omic_ids = batch["omic_ids"].to(device, non_blocking=True)  # [B, N_seq, L_proj]
    omic_mask = (omic_ids != 1).long().to(device, non_blocking=True)  # 填充token的掩码
    
    # 展平批次和序列维度
    flat_omic_ids = omic_ids.view(-1, omic_ids.size(-1))  # [B*N_seq, L_proj]
    flat_omic_mask = omic_mask.view(-1, omic_mask.size(-1))  # [B*N_seq, L_proj]

    # 过滤pad
    valid_mask = (flat_omic_ids != 1).any(dim=1)
    flat_omic_ids = flat_omic_ids[valid_mask]
    flat_omic_mask = flat_omic_mask[valid_mask]
    
    # 获取每个序列的类型信息
    omic_infos = [
        info for sublist in batch["omic_info_list"]
        for info in sublist
    ]
    seq_types = [info["type"] for info in omic_infos]
    
    # 分离DNA/RNA和蛋白质序列
    dna_rna_indices = [i for i, st in enumerate(seq_types) if st in {"dna", "rna"}]
    protein_indices = [i for i, st in enumerate(seq_types) if st == "protein"]
    
    # 初始化输出张量
    dna_rna_emb = torch.zeros(B, text_emb.size(-1), device=device, dtype=text_emb.dtype)
    protein_emb = torch.zeros(B, text_emb.size(-1), device=device, dtype=text_emb.dtype)
    
    # 批量处理DNA/RNA序列
    if dna_rna_indices:
        dna_rna_ids = flat_omic_ids[dna_rna_indices]
        dna_rna_mask = flat_omic_mask[dna_rna_indices]
        print("DNA/RNA ids  min/max:", dna_rna_ids.min().item(), dna_rna_ids.max().item())

        dna_rna_embs = encode_omics_batch(dna_rna_model, dna_rna_ids, dna_rna_mask)
        
        # 将嵌入向量映射回原始样本
        batch_indices = torch.tensor([i // omic_ids.size(1) for i in dna_rna_indices], 
                                    device=device)
        dna_rna_emb.scatter_add_(0, 
                                batch_indices.unsqueeze(-1).expand_as(dna_rna_embs), 
                                dna_rna_embs)
        
        # 计算每个样本的序列数量用于归一化
        seq_counts = torch.zeros(B, device=device)
        seq_counts.scatter_add_(0, batch_indices, torch.ones_like(batch_indices, dtype=torch.float))
        dna_rna_emb = dna_rna_emb / seq_counts.clamp(min=1).unsqueeze(-1)
    
    # 批量处理蛋白质序列
    if protein_indices:
        protein_ids = flat_omic_ids[protein_indices]
        protein_mask = flat_omic_mask[protein_indices]
        print("Protein ids  min/max:", protein_ids.min().item(), protein_ids.max().item())

        protein_embs = encode_omics_batch(protein_model, protein_ids, protein_mask)
        
        # 将嵌入向量映射回原始样本
        batch_indices = torch.tensor([i // omic_ids.size(1) for i in protein_indices], 
                                    device=device)
        protein_emb.scatter_add_(0, 
                                batch_indices.unsqueeze(-1).expand_as(protein_embs), 
                                protein_embs)
        
        # 计算每个样本的序列数量用于归一化
        seq_counts = torch.zeros(B, device=device)
        seq_counts.scatter_add_(0, batch_indices, torch.ones_like(batch_indices, dtype=torch.float))
        protein_emb = protein_emb / seq_counts.clamp(min=1).unsqueeze(-1)

    # 3. 拼接所有嵌入
    concat_emb = torch.cat([text_emb, dna_rna_emb, protein_emb], dim=1)
    
    return concat_emb


# pylint: disable=too-many-statements
def main():
    args = parse_args()

    # ---------- 1. tokenizer & dataset ----------
    tokenizer, dna_rna_tokenizer, protein_tokenizer = setup_tokenizers(args)
    train_dataset, _ = setup_dataset(
        args, tokenizer, dna_rna_tokenizer, protein_tokenizer
    )

    # ---------- 2. 模型 ----------
    text_model, dna_rna_model, protein_model = setup_models(args)
    text_model.module.resize_token_embeddings(len(tokenizer))
    dna_rna_model.module.resize_token_embeddings(len(dna_rna_tokenizer))
    protein_model.module.resize_token_embeddings(len(protein_tokenizer))

    text_model.eval()
    dna_rna_model.eval()
    protein_model.eval()

    # ---------- 3. 构造 DataLoader ----------
    def collate(batch):
        out = {}
        tensor_keys = {"input_ids", "attention_mask"}
        pad_keys = {"omic_ids"}
        list_keys = {"task", "omic_info_list"}

        for k in batch[0]:
            if k in tensor_keys:
                out[k] = torch.stack([s[k] for s in batch])
            elif k in pad_keys:
                tensors = [s[k] for s in batch]
                max_n = max(t.size(0) for t in tensors)
                padded = []
                for t in tensors:
                    pad_len = max_n - t.size(0)
                    if pad_len > 0:
                        pad_t = torch.full(
                            (pad_len, t.size(1)),
                            fill_value=1,
                            dtype=t.dtype,
                        )
                        padded.append(torch.cat([t, pad_t]))
                    else:
                        padded.append(t)
                out[k] = torch.stack(padded)
            elif k in list_keys:
                out[k] = [s[k] for s in batch]
            else:
                out[k] = [s[k] for s in batch]

        return out

    loader = DataLoader(
        train_dataset,
        batch_size=args.embed_batch_size,
        shuffle=False,
        num_workers=0,
        # num_workers=args.dataloader_num_workers,
        collate_fn=collate,
        pin_memory=False,
        persistent_workers=False,
    )
    
    # ---------- 4. batch 推理 ----------
    all_embs = []
    all_tasks = []
    
    for idx, batch in enumerate(tqdm(loader, desc="Embedding")):
        if args.read_nums and idx * args.embed_batch_size >= args.read_nums:
            break
            
        emb = embed_text_batch(args, text_model, dna_rna_model, protein_model, batch)
        all_embs.append(emb.cpu())
        all_tasks.extend(batch["task"][:emb.size(0)])  # 记录对应的task名称
    
    embeddings = torch.cat(all_embs, dim=0)[: args.read_nums]
    task_names = all_tasks[:len(embeddings)]

    # 存储 embeddings
    embeddings = embeddings.numpy()
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    np.save(output_path / "embeddings.npy", embeddings)

    # ---------- 5. UMAP ----------
    if not args.skip_eval:
        embeddings_gpu = cp.asarray(embeddings)
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings_gpu)
        reducer = UMAP(n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings_scaled)

        df = pd.DataFrame(
            {
                "task": task_names,
                "projection_x": cp.asnumpy(embedding_2d[:, 0]),
                "projection_y": cp.asnumpy(embedding_2d[:, 1]),
            }
        )
        parquet_path = output_path / "umap_projection.parquet"
        df.to_parquet(parquet_path, index=False)


if __name__ == "__main__":
    main()
