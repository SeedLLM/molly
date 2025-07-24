"""
不训练的BioMLLM推理
"""
import os
import json
import argparse
from argparse import Namespace
from functools import partial
import re
import traceback
from typing import List, Tuple

import torch
import pandas as pd
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
from peft import PeftModel, PeftConfig  # 添加PEFT库导入

# Import our custom modules
from src.model import QwenWithNt, get_qwen_nt_config
from src.utils.tools import print_rank_0
from src.dataset.dna_rna_dataset import OmicsTestDataset, qwen_dna_collate_fn, DatasetConfig, qwen_dna_collate_fn_inference


def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal LLM Inference")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="Local rank for distributed inference, ignored if single GPU")
    parser.add_argument("--text_model_path", type=str, required=True, 
                        help="Path to text model (Qwen)")
    parser.add_argument("--bio_model_path", type=str, required=True,
                        help="Path to bio model")
    parser.add_argument("--multimodal_k_tokens", type=int, default=128,
                        help="Number of multimodal tokens used for DNA sequence projection (should match training)")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to dataset file (jsonl/csv/xlsx/json) for batch inference")
    parser.add_argument("--output_path", type=str, default="./results",
                        help="Directory to save batch inference outputs")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum tokens to generate per example")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for dataset inference")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu","cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process (None=all)")
    parser.add_argument("--json_file", type=str)
    return parser.parse_args()


class MultiModalInfer:
    """Batch inference helper for Qwen + NT multimodal model (training-style DNA injection)."""

    def __init__(self, args):
        self.args = args

        # ‑- Seed ----------------------------------------------------------------
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        # ‑- Device --------------------------------------------------------------
        self.device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

        # ‑- Tokenisers ----------------------------------------------------------
        print("Loading tokenizers…")
        self.text_tokenizer = AutoTokenizer.from_pretrained(args.text_model_path, trust_remote_code=True)
        self.bio_tokenizer = AutoTokenizer.from_pretrained(args.bio_model_path, trust_remote_code=True)

        # Ensure DNA special tokens exist in text tokenizer
        special_tokens = ["<|dna_start|>", "<|dna_pad|>", "<|dna_end|>",
                          "<|rna_start|>", "<|rna_pad|>", "<|rna_end|>"]
        self.text_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

        # ‑- Model --------------------------------------------------------------
        print("Loading multimodal model…")
        self._load_model()

    def _load_model(self):
        args = self.args
        
        # Get model configuration
        model_config = get_qwen_nt_config(args.text_model_path, args.bio_model_path)
        model_config.project_token_num = args.multimodal_k_tokens
        print(f"Setting project_token_num to {model_config.project_token_num}")

        self.model = QwenWithNt(model_config)
        self.model.set_special_tokens(self.text_tokenizer)
        
        print(f"Loading base Qwen model from {args.text_model_path}")
        qwen_model = AutoModelForCausalLM.from_pretrained(
            args.text_model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        )
        self.model.model.load_state_dict(qwen_model.state_dict())
        del qwen_model
                    
        # 加载NT模型参数
        print(f"Loading NT model from {args.bio_model_path}")
        bio_model = AutoModelForMaskedLM.from_pretrained(args.bio_model_path, trust_remote_code=True)
        
        self.model.bio_model.load_state_dict(bio_model.state_dict())
        del bio_model
        
        # Move model to device and set to evaluation mode
        self.model = self.model.to(torch.bfloat16).to(self.device)
        self.model.eval()

    def run_dataset(self):
        """Run inference sample-by-sample without padding."""
        args = self.args
        json_file = open(args.json_file, 'a', encoding='utf-8')
        test_config = DatasetConfig(
            max_len = args.max_length,
            max_src_len = args.max_length,
            mode = 'sft',
            padding = True,
            input_field = 'input',
            output_field = 'output',
            multimodal_k_tokens = args.multimodal_k_tokens,
            type = 'inference'
        )
        test_dataset = OmicsTestDataset(args.dataset_path, 
                                     self.text_tokenizer, 
                                     test_config,
                                     self.bio_tokenizer,
                                     num_workers=4,
                                     type='inference')
        
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            collate_fn=qwen_dna_collate_fn_inference
        )

        for batch_idx, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Infer"):
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    omic_ids=batch['omic_ids'],
                    omic_start_pos_list=batch['omic_start_pos_list'],
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=True,
                )
    
            decoded = self.text_tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for i in range(len(decoded)):
                sample_data = {
                    'decoded_output': decoded[i],
                    'input': batch['input'][i],  # Get the input for the i-th sample
                    'gt_output': batch['raw_output'][i],  # Get the ground truth output for the i-th sample
                    'gt_label': batch['raw_label'][i],  # Get the label for the i-th sample
                    'task': batch['raw_task'][i],  # Get the task for the i-th sample
                    'kind': batch['raw_kind'][i],  # Get the kind for the i-th sample
                }

                # Write the sample data to the JSON file
                json.dump(sample_data, json_file, ensure_ascii=False)
                json_file.write("\n")  # Add a newline for better separation between records
                json_file.flush()

        json_file.close()


if __name__ == "__main__":
    args = parse_args()
    inferer = MultiModalInfer(args)
    inferer.run_dataset()
