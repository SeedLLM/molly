import time
from typing import Optional, List, Dict, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM
from transformers.modeling_outputs import CausalLMOutputWithPast



class OmicsOne(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_config = config.text_config
        self.bio_config = config.bio_config
        self.model = AutoModelForCausalLM.from_config(self.text_config)

        self.bio_model = AutoModelForMaskedLM.from_config(self.bio_config, trust_remote_code=True)
        self.multimodal_projector = nn.Linear(self.bio_config.hidden_size, self.text_config.hidden_size)
        self.project_token_num = config.project_token_num
        
        # Special token IDs
        self.dna_start_token_id = None
        self.dna_end_token_id = None  
        self.dna_pad_token_id = None  

    def set_special_tokens(self, tokenizer):
        """Set special token IDs from tokenizer"""
        self.dna_start_token_id = tokenizer.convert_tokens_to_ids("<|dna_start|>")
        self.dna_end_token_id = tokenizer.convert_tokens_to_ids("<|dna_end|>")
        self.dna_pad_token_id = tokenizer.convert_tokens_to_ids("<|dna_pad|>")
        self.rna_start_token_id = tokenizer.convert_tokens_to_ids("<|rna_start|>")
        self.rna_end_token_id = tokenizer.convert_tokens_to_ids("<|rna_end|>")
        self.rna_pad_token_id = tokenizer.convert_tokens_to_ids("<|rna_pad|>")

    def process_dna_sequences(
        self,
        hidden_states: torch.Tensor,
        dna_ids_list: List[List[torch.LongTensor]],
        dna_start_pos_list: List[List[int]],
        device: torch.device
    ) -> torch.Tensor:
        """
        参数：
            hidden_states: 原始 token embedding，形状 [B, L, D]
            dna_ids_list: 每个样本包含若干个 DNA 序列的 token ids，List[List[Tensor]]
            dna_start_pos_list: 每个样本中每个 DNA 插入位置（即 <|dna_start|> 的位置）
            device: 目标设备

        返回：
            注入 DNA 表征后的 hidden_states，形状 [B, L, D]
        """

        batch_size = hidden_states.shape[0]

        flat_dna_ids = []  # 展平成一批 DNA 序列，便于统一送入 BERT
        mapping = []       # 记录每条 DNA 的注入目标：(batch_idx, start_pos, length)

        # 遍历每个样本
        for b in range(batch_size):
            for i, start_pos in enumerate(dna_start_pos_list[b]):
                dna = dna_ids_list[b][i]            # Tensor: [L_dna]
                flat_dna_ids.append(dna.to(device)) # 添加到 flat list 中
                mapping.append((b, start_pos, len(dna)))  # 记录注入位置信息

        if not flat_dna_ids:
            # 如果当前 batch 没有任何 DNA 序列，直接返回原始 embedding
            return hidden_states

        # 将所有 DNA 序列堆叠成一个张量：[N_dna, L_dna]
        padded_dna = torch.stack(flat_dna_ids, dim=0)

        # 使用 BERT/BioBERT 获取 DNA 表征
        dna_outputs = self.bio_model(
            padded_dna,
            attention_mask=(padded_dna != 1).long(),  # 1 是 padding token
            encoder_attention_mask=(padded_dna != 1).long(),
            output_hidden_states=True,
            return_dict=True
        )
        dna_embeddings = dna_outputs['hidden_states'][-1] # 形状：[N_dna, L_dna, H_bio]

        # 映射到主模型的 embedding 空间：[N_dna, L_dna, H_text]
        proj_embeddings = self.multimodal_projector(dna_embeddings)

        # 注入到 hidden_states 中对应位置
        for i, (b, start_pos, length) in enumerate(mapping):
            # 将投影后的 DNA 表征插入到 hidden_states 中 <|dna_start|> 后的位置
            # 注意：最多注入 self.project_token_num 个 token
            # 如果start_pos是 -1，表示没有 DNA 序列
            if start_pos == -1:
                continue
            hidden_states[b, start_pos + 1: start_pos + 1 + self.project_token_num, :] = \
                proj_embeddings[i, :self.project_token_num, :]

        return hidden_states


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        omic_ids: Optional[List[List[torch.LongTensor]]] = None,
        omic_start_pos_list: Optional[List[List[int]]] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor, ...], CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.text_config.use_return_dict

        # Always disable cache during distributed training/evaluation to avoid DynamicCache errors
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            use_cache = False

        # Get token embeddings
        hidden_states = self.model.get_input_embeddings()(input_ids)

        if omic_ids is not None:
            # Sanity check
            for i in range(len(omic_ids)):
                assert len(omic_ids[i]) == len(omic_start_pos_list[i]), \
                    f"Mismatch in DNA count vs start_pos count at index {i}"

            hidden_states = self.process_dna_sequences(
                hidden_states,
                omic_ids,
                omic_start_pos_list,
                input_ids.device
            )

        outputs = self.model(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs


    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        omic_ids: Optional[List[List[torch.LongTensor]]] = None,
        omic_start_pos_list: Optional[List[List[int]]] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: Optional[int] = None,
        num_beams: Optional[int] = None,
        no_repeat_ngram_size: Optional[int] = None,
        **generate_kwargs
    ) -> torch.LongTensor:
        
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            use_cache = False
    
        hidden_states = self.model.get_input_embeddings()(input_ids)

        if omic_ids is not None:
            if omic_start_pos_list is None:
                omic_start_pos_list = []
                for ids in input_ids:
                    positions = (ids == self.dna_start_token_id).nonzero(as_tuple=True)[0].tolist()
                    omic_start_pos_list.append(positions)

            # Sanity check
            for i in range(len(omic_ids)):
                assert len(omic_ids[i]) == len(omic_start_pos_list[i]), \
                    f"Mismatch in DNA count vs start_pos count at index {i}"

            hidden_states = self.process_dna_sequences(
                hidden_states,
                omic_ids,
                omic_start_pos_list,
                input_ids.device
            )

        output_ids = self.model.generate(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=self.model.config.pad_token_id,
            eos_token_id=self.model.config.eos_token_id,
            **generate_kwargs
        )
        return output_ids
