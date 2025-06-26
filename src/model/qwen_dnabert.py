import time
from typing import Optional, List

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, Qwen3ForCausalLM, AutoConfig


class QwenWithBert(nn.Module):
    def __init__(self, config):
        super().__init__()

        # import torch.nn.init as init

        # # 临时替换 init 方法为 no-op, 调试的时候添加，用于加速构建
        # init.kaiming_uniform_ = lambda *args, **kwargs: None
        # init.uniform_ = lambda *args, **kwargs: None
        # init.normal_ = lambda *args, **kwargs: None

        self.text_config = config.text_config
        self.bio_config = config.multimodal_model_config
        # self.model = Qwen3ForCausalLM(self.text_config)
        self.model = AutoModelForCausalLM.from_config(self.text_config)
        # self.bio_model = BertModel(self.bio_config)
        self.bio_model = AutoModel.from_config(self.bio_config)
        self.multimodal_projector = nn.Linear(self.bio_config.hidden_size, self.text_config.hidden_size)
        self.project_token_num = config.project_token_num
    
    def forward(self, **kwargs):
        input_ids = kwargs["input_ids"]                      # [B, L]
        dna_ids_list = kwargs["dna_ids_lists"]                # List[Tensor], 每个样本内多个 DNA 片段
        dna_start_pos_list = kwargs["dna_start_pos_lists"]    # List[List[int]], 每个样本内每段 DNA 的插入位置
        labels = kwargs["labels"]                            # [B, L]

        batch_size, seq_len = input_ids.shape
        hidden_states = self.model.get_input_embeddings()(input_ids)  # [B, L, D]
        hidden_dim = hidden_states.shape[-1]

        # 将 dna_ids_list[b][i] 全部打平
        flat_dna_ids = []
        mapping = []  # (batch_idx, start_pos, length)
        for b in range(batch_size):
            for i, start_pos in enumerate(dna_start_pos_list[b]):
                dna = dna_ids_list[b][i]
                flat_dna_ids.append(dna)
                mapping.append((b, start_pos, len(dna)))

        # 直接堆叠成 tensor，因为所有 dna 序列长度一致
        padded_dna = torch.stack(flat_dna_ids, dim=0).to(hidden_states.device)  # [N, fixed_len]
        dna_embeddings = self.bio_model(padded_dna)[0]  # 直接返回 [N, L_dna, hidden_size]
        proj_out = self.multimodal_projector(dna_embeddings)
        # scatter 回去
        pad_length = self.project_token_num
        for i, (b, start_pos, length) in enumerate(mapping):
            hidden_states[b, start_pos + 1: start_pos + 1 + pad_length, :] = proj_out[i, :pad_length, :]

        # 进入 Qwen
        outputs = self.model(
            inputs_embeds=hidden_states,
            labels=labels
        )
        # outputs.logits
        return outputs.loss, None

    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,               # [B, L]
        dna_ids_lists: List[List[torch.LongTensor]],   # List (batch) of List (segments) of [seg_len]
        dna_start_pos_lists: List[List[int]],         # List (batch) of start positions
        attention_mask: Optional[torch.LongTensor] = None,  # [B, L], 若有
        **generate_kwargs                           # e.g. max_length=..., num_beams=..., do_sample=...
    ) -> torch.LongTensor:
        """
        融合 DNA-BERT 提取的多模态 embedding，使用 Qwen3 的 generate 接口生成序列。

        返回：
            output_ids: [B, T_gen] 生成的 token id 序列
        """
        device = input_ids.device

        # 1. 计算初始 token embedding
        hidden_states = self.model.get_input_embeddings()(input_ids)  # [B, L, D]

        # 2. 将所有 DNA 片段打平并到同设备
        flat_dna = []
        mapping = []  # (batch_idx, start_pos, length)
        for b, (dna_list, pos_list) in enumerate(zip(dna_ids_lists, dna_start_pos_lists)):
            for dna_ids, start in zip(dna_list, pos_list):
                flat_dna.append(dna_ids.to(device))
                mapping.append((b, start, dna_ids.size(0)))

        padded_dna = torch.stack(flat_dna, dim=0)  # [N, seg_len]

        # 3. 用 DNA-BERT 提取 embedding 并投射
        dna_emb = self.bio_model(padded_dna)[0]                     # [N, seg_len, H_bio]
        proj_emb = self.multimodal_projector(dna_emb)               # [N, seg_len, H_text]

        # 4. 把投射后的 embedding 插回到 hidden_states
        for i, (b, start, length) in enumerate(mapping):
            hidden_states[b, start + 1 : start + 1 + self.project_token_num, :] = proj_emb[i, :self.project_token_num, :]

        # 5. 准备 attention_mask（如未传则全1）
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)

        # 6. 调用底层 generate（传入 inputs_embeds 而非 input_ids）
        ml = 2048
        temperature = 0.8
        top_p=0.95
        output_ids = self.model.generate(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            max_length=ml,
            do_sample=True,
            temperature=temperature,
            top_p=top_p
        )
        return output_ids