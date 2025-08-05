from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM
from transformers.modeling_outputs import CausalLMOutputWithPast


class OmicsOne(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_config = config.text_config
        self.dna_rna_config = config.dna_rna_config
        self.protein_config = config.protein_config

        self.model = AutoModelForCausalLM.from_config(self.text_config)

        self.dna_rna_model = AutoModelForMaskedLM.from_config(
            self.dna_rna_config, trust_remote_code=True
        )
        self.dna_rna_projector = nn.Linear(
            self.dna_rna_config.hidden_size, self.text_config.hidden_size
        )
        self.dna_rna_project_token_num = config.dna_rna_project_token_num

        self.protein_model = AutoModelForMaskedLM.from_config(
            self.protein_config, trust_remote_code=True
        )
        self.protein_projector = nn.Linear(
            self.protein_config.hidden_size, self.text_config.hidden_size
        )
        self.protein_project_token_num = config.protein_project_token_num

    def set_special_tokens(self, tokenizer):
        """Set special token IDs from tokenizer"""
        self.dna_start_token_id = tokenizer.convert_tokens_to_ids("<|dna_start|>")
        self.dna_end_token_id = tokenizer.convert_tokens_to_ids("<|dna_end|>")
        self.dna_pad_token_id = tokenizer.convert_tokens_to_ids("<|dna_pad|>")
        self.rna_start_token_id = tokenizer.convert_tokens_to_ids("<|rna_start|>")
        self.rna_end_token_id = tokenizer.convert_tokens_to_ids("<|rna_end|>")
        self.rna_pad_token_id = tokenizer.convert_tokens_to_ids("<|rna_pad|>")
        self.protein_start_token_id = tokenizer.convert_tokens_to_ids(
            "<|protein_start|>"
        )
        self.protein_end_token_id = tokenizer.convert_tokens_to_ids("<|protein_end|>")
        self.protein_pad_token_id = tokenizer.convert_tokens_to_ids("<|protein_pad|>")

    def process_omic_sequences(
        self,
        hidden_states: torch.Tensor,
        omic_ids_list: List[List[torch.LongTensor]],
        omic_info_list: List[dict],
        device: torch.device,
    ) -> torch.Tensor:
        def _inject_omic(
            omic_ids: List[torch.LongTensor],
            omic_mappings: List[Tuple[int, int, int]],
            backbone_model: nn.Module,
            projector: nn.Module,
            max_tokens: int,
        ):
            """
            将一类 omic 序列（DNA/RNA 或蛋白质）统一处理并写回 hidden_states
            """
            if not omic_ids:
                return
            padded = torch.stack(omic_ids, dim=0)
            mask = (padded != 1).long()
            assert (
                padded < backbone_model.config.vocab_size
            ).all(), f"out-of-range token: {padded[padded >= backbone_model.config.vocab_size]}"
            try:
                if backbone_model == self.dna_rna_model:
                    out = backbone_model(
                        padded,
                        attention_mask=mask,
                        encoder_attention_mask=mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                elif backbone_model == self.protein_model:
                    out = backbone_model(
                        padded,
                        attention_mask=mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )
            except Exception as e:
                raise RuntimeError(f"Error processing omic sequences: {e}")
            emb = projector(out["hidden_states"][-1])

            for idx, (b, start_pos, _) in enumerate(omic_mappings):
                if start_pos == -1:
                    continue
                k = min(max_tokens, emb.size(1))
                hidden_states[b, start_pos + 1 : start_pos + 1 + k] = emb[idx, :k]

        batch_size = hidden_states.shape[0]

        dna_rna_ids, dna_rna_map = [], []
        protein_ids, protein_map = [], []

        for b in range(batch_size):
            for omic_id, info in zip(omic_ids_list[b], omic_info_list[b]):
                omic_type = info["type"]
                start_pos = info["start"]
                oid = omic_id.to(device)
                if omic_type in ("dna", "rna"):
                    dna_rna_ids.append(oid)
                    dna_rna_map.append((b, start_pos, len(oid)))
                elif omic_type == "protein":
                    protein_ids.append(oid)
                    protein_map.append((b, start_pos, len(oid)))
                elif omic_type == "pad":
                    continue
                else:
                    raise ValueError(f"Unsupported omic type: {omic_type}")

        _inject_omic(
            dna_rna_ids,
            dna_rna_map,
            self.dna_rna_model,
            self.dna_rna_projector,
            self.dna_rna_project_token_num,
        )

        _inject_omic(
            protein_ids,
            protein_map,
            self.protein_model,
            self.protein_projector,
            self.protein_project_token_num,
        )

        return hidden_states

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        omic_ids: Optional[List[List[torch.LongTensor]]] = None,
        omic_info_list: Optional[List[List[int]]] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor, ...], CausalLMOutputWithPast]:
        return_dict = (
            return_dict if return_dict is not None else self.text_config.use_return_dict
        )

        # Always disable cache during distributed training/evaluation to avoid DynamicCache errors
        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            use_cache = False

        # Get token embeddings
        hidden_states = self.model.get_input_embeddings()(input_ids)

        if omic_ids is not None:
            # Sanity check
            for i in range(len(omic_ids)):
                assert len(omic_ids[i]) == len(
                    omic_info_list[i]
                ), f"Mismatch in DNA count vs start_pos count at index {i}"

            hidden_states = self.process_omic_sequences(
                hidden_states, omic_ids, omic_info_list, input_ids.device
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
        omic_info_list: Optional[List[List[dict]]] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: Optional[int] = None,
        num_beams: Optional[int] = None,
        no_repeat_ngram_size: Optional[int] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:

        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            generate_kwargs.setdefault("use_cache", False)

        hidden_states = self.model.get_input_embeddings()(input_ids)

        if omic_ids is not None:
            for i in range(len(omic_ids)):
                assert len(omic_ids[i]) == len(
                    omic_info_list[i]
                ), f"Mismatch in omic count vs info count at index {i}"

            hidden_states = self.process_omic_sequences(
                hidden_states, omic_ids, omic_info_list, input_ids.device
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
            **generate_kwargs,
        )
        return output_ids
