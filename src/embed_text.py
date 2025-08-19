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

from train import setup_dataloaders, setup_tokenizers
from utils import get_current_device


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
    current_device = get_current_device()

    text_model = AutoModel.from_pretrained(
        args.text_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=current_device,
    )
    dna_rna_model = AutoModelForMaskedLM.from_pretrained(
        args.dna_rna_model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map=current_device,
    )
    protein_model = AutoModelForMaskedLM.from_pretrained(
        args.protein_model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map=current_device,
    )

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
def encode_omics(model, ids, seq_type, k_tokens):
    """
    ids: [B, L_proj] 已 padding/truncate 到 k_tokens
    返回该条序列的 embedding（取 [CLS] 或 mean-pooling）
    """
    ids = ids.to(model.device)
    attention_mask = (ids != 1).long().to(model.device)  # [B, L_proj]
    out = model(ids, output_hidden_states=True, attention_mask=attention_mask)
    hidden = out.hidden_states[-1]  # [B, L_proj, D]
    # 简单做法：mean-pool
    emb = hidden.mean(dim=1).squeeze(0)  # [D]
    return emb


@torch.no_grad()
def embed_text(args, text_model, dna_rna_model, protein_model, sample):
    device = text_model.device

    input_ids = sample["input_ids"].unsqueeze(0).to(text_model.device)

    if (input_ids >= text_model.config.vocab_size).any():
        print(
            f"Warning: input_ids contains out-of-range tokens: {input_ids[input_ids >= text_model.config.vocab_size]}"
        )

    omic_ids = sample["omic_ids"]

    # 1. text
    attention_mask = sample["attention_mask"].unsqueeze(0).to(text_model.device)
    text_emb = encode_text(text_model, input_ids, attention_mask)

    # 2. omics
    omic_ids = sample["omic_ids"].to(device)  # [N_seq, L_proj]
    omic_info = sample["omic_info_list"]  # list of dict

    dna_rna_embs, protein_embs = [], []
    for i, info in enumerate(omic_info):
        seq_type = info["type"]
        ids = omic_ids[i].unsqueeze(0)  # [1, L_proj]
        if seq_type in {"dna", "rna"}:
            dna_rna_embs.append(
                encode_omics(dna_rna_model, ids, seq_type, args.dna_rna_k_tokens)
            )
        elif seq_type == "protein":
            protein_embs.append(
                encode_omics(protein_model, ids, seq_type, args.protein_k_tokens)
            )
        else:  # pad 等
            pass

    # 取平均
    if dna_rna_embs:
        dna_rna_emb = torch.stack(dna_rna_embs).mean(0)  # [D_dna_rna]
    else:
        dna_rna_emb = torch.zeros(text_model.config.hidden_size, device=device)

    if protein_embs:
        protein_emb = torch.stack(protein_embs).mean(0)  # [D_protein]
    else:
        protein_emb = torch.zeros(text_model.config.hidden_size, device=device)

    # 3. 拼接
    concat = torch.cat(
        [text_emb, dna_rna_emb.unsqueeze(0), protein_emb.unsqueeze(0)], dim=1
    )  # [1, D_text + D_dna_rna + D_protein]
    return concat


# 4. 新增 batch 推理函数（放在 embed_text 后面即可）
@torch.no_grad()
def embed_text_batch(args, text_model, dna_rna_model, protein_model, batch):
    device = text_model.device
    B = batch["input_ids"].size(0)

    # 1. text
    text_emb = encode_text(
        text_model, batch["input_ids"].to(device), batch["attention_mask"].to(device)
    )  # [B, D]

    # 2. omics
    omic_ids = batch["omic_ids"].to(device)  # [B, N_seq, L_proj]
    flat_ids = omic_ids.view(-1, omic_ids.size(-1))  # [B*N_seq, L_proj]

    # 构造「样本 id」索引：0,0,...,1,1,...,2,2,... 长度 = B*N_seq
    N_seq = omic_ids.size(1)
    sample_idx = torch.arange(B, device=device).repeat_interleave(N_seq)

    # 拉平 info
    omic_infos = [info for sub in batch["omic_info_list"] for info in sub]

    def encode_subset(model, idx_list, k):
        if len(idx_list) == 0:
            return torch.zeros(B, text_model.config.hidden_size, device=device)
        sub_ids = flat_ids[idx_list]
        mask = (sub_ids != 1).long()
        out = model(
            sub_ids, attention_mask=mask, output_hidden_states=True
        ).hidden_states[-1]
        sub_emb = out.mean(dim=1)  # [len(idx_list), D]

        # 用 sample_idx 把同一样本的向量平均
        idx_tensor = sample_idx[idx_list]  # [len(idx_list)]
        out_emb = torch.zeros(B, sub_emb.size(-1), device=device)
        count = torch.zeros(B, 1, device=device)

        out_emb.scatter_add_(0, idx_tensor.unsqueeze(-1).expand_as(sub_emb), sub_emb)
        count.scatter_add_(0, idx_tensor.unsqueeze(-1), torch.ones_like(sub_emb[:, :1]))

        return out_emb / count.clamp(min=1)  # [B, D]

    dna_rna_idx = [
        i for i, info in enumerate(omic_infos) if info["type"] in {"dna", "rna"}
    ]
    protein_idx = [i for i, info in enumerate(omic_infos) if info["type"] == "protein"]

    dna_rna_emb = encode_subset(dna_rna_model, dna_rna_idx, args.dna_rna_k_tokens)
    protein_emb = encode_subset(protein_model, protein_idx, args.protein_k_tokens)

    return torch.cat([text_emb, dna_rna_emb, protein_emb], dim=1)


# pylint: disable=too-many-statements
def main():
    args = parse_args()

    # ---------- 1. tokenizer & dataset ----------
    tokenizer, dna_rna_tokenizer, protein_tokenizer = setup_tokenizers(args)
    train_dataset, _ = setup_dataloaders(
        args, tokenizer, dna_rna_tokenizer, protein_tokenizer
    )

    # ---------- 2. 模型 ----------
    text_model, dna_rna_model, protein_model = setup_models(args)
    text_model.resize_token_embeddings(len(tokenizer))
    text_model.eval()
    dna_rna_model.eval()
    protein_model.eval()

    # ---------- 3. 构造 DataLoader ----------
    def collate(batch):
        out = {}
        tensor_keys = {"input_ids", "attention_mask"}
        pad_keys = {"omic_ids"}
        list_keys = {"task_name"}  # 把 task_name 当成 list 处理即可

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
                        padded.append(t[:max_n])
                out[k] = torch.stack(padded)
            elif k in list_keys:
                out[k] = [s[k] for s in batch]  # 保留 list
            else:
                out[k] = [s[k] for s in batch]

        return out

    loader = DataLoader(
        train_dataset,
        batch_size=args.embed_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate,
    )
    # ---------- 4. batch 推理 ----------
    all_embs = []
    for idx, batch in enumerate(tqdm(loader, desc="Embedding")):
        if args.read_nums and idx * args.embed_batch_size >= args.read_nums:
            break
        emb = embed_text_batch(args, text_model, dna_rna_model, protein_model, batch)
        all_embs.append(emb.cpu())  # 省显存
    embeddings = torch.cat(all_embs, dim=0)[: args.read_nums]

    # 存储 embeddings
    embeddings = embeddings.numpy()  # 转为 numpy 数组
    np.save(Path(args.output_dir) / "embeddings.npy", embeddings)

    # ---------- 5. UMAP ----------
    embeddings_gpu = cp.asarray(embeddings)
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_gpu)
    reducer = UMAP(n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings_scaled)

    # 收集 task_name
    task_names = []
    for b in loader:
        task_names.extend(b["task_name"])
        if args.read_nums and len(task_names) >= args.read_nums:
            task_names = task_names[: args.read_nums]
            break
    task_names = task_names[: len(embedding_2d)]

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.DataFrame(
        {
            "task_name": task_names,
            "projection_x": cp.asnumpy(embedding_2d[:, 0]),
            "projection_y": cp.asnumpy(embedding_2d[:, 1]),
        }
    )
    parquet_path = Path(args.output_dir) / "umap_projection.parquet"
    df.to_parquet(parquet_path, index=False)


if __name__ == "__main__":
    main()
