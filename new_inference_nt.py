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
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig

# Import our custom modules
from src.model import QwenWithNt, get_qwen_nt_config
from src.utils.tools import print_rank_0


def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal LLM Inference (DNA + Text)")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="Local rank for distributed inference, ignored if single GPU")
    parser.add_argument("--text_model_path", type=str, required=True, 
                        help="Path to text model (Qwen)")
    parser.add_argument("--bio_model_path", type=str, required=True,
                        help="Path to DNA-BERT model")
    parser.add_argument("--trained_model_path", type=str, required=True, 
                        help="Path to trained multimodal model checkpoint")
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
        self.dna_tokenizer = AutoTokenizer.from_pretrained(args.bio_model_path, trust_remote_code=True)

        # Ensure DNA special tokens exist in text tokenizer
        special_tokens = ["<|dna_start|>", "<|dna_pad|>", "<|dna_end|>"]
        self.text_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

        # ‑- Model --------------------------------------------------------------
        print("Loading multimodal model…")
        self._load_model()

    def _load_model(self):
        args = self.args
        
        # Get model configuration
        model_config = get_qwen_nt_config(args.text_model_path, args.bio_model_path)
        
        # Explicitly set project_token_num to match training
        model_config.project_token_num = args.multimodal_k_tokens
        print(f"Setting project_token_num to {model_config.project_token_num}")

        # >>> ADD THIS LINE to fix the architecture mismatch <<<
        model_config.bio_config.intermediate_size = 8192

        # Create regular training-time model (DNA injected at <|dna_start|>)
        self.model = QwenWithNt(model_config)
        # Important: inform the model of special token IDs so it can find <|dna_start|> etc.
        self.model.set_special_tokens(self.text_tokenizer)
        
        # Load checkpoint
        print(f"Loading checkpoint from {args.trained_model_path}")
        try:
            # Try to load with torch.load first
            ckpt = torch.load(args.trained_model_path, map_location="cpu")
            
            # Check if it's a DeepSpeed checkpoint or direct state dict
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            else:
                state_dict = ckpt
                
            # Load state dict - use strict=False since we added new methods
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            print(f"Model loaded successfully")
            if missing:
                print(f"Missing keys: {len(missing)} keys")
            if unexpected:
                print(f"Unexpected keys: {len(unexpected)} keys")
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            print("Loading base models without fine-tuned weights...")
            
            # Load Qwen model parameters
            print(f"Loading Qwen model from {args.text_model_path}")
            qwen_model = AutoModelForCausalLM.from_pretrained(
                args.text_model_path, 
                trust_remote_code=True, 
                torch_dtype=torch.bfloat16
            )
            self.model.model.load_state_dict(qwen_model.state_dict())
            del qwen_model  # Free memory
            
            # Load NT model parameters
            print(f"Loading NT model from {args.bio_model_path}")
            
            # 加载DNA模型配置
            from src.model.esm_config import EsmConfig
            from src.model.modeling_esm import EsmModel
            
            # 直接从预训练模型加载配置，确保与预训练模型完全一致
            bio_config = EsmConfig.from_pretrained(args.bio_model_path)
            
            # 更新模型配置中的bio_config
            model_config.bio_config = bio_config
            
            # 重新创建bio_model部分，使用正确的配置
            self.model.bio_model = EsmModel(bio_config)
            
            # 加载预训练权重
            print(f"Loading NT model from {args.bio_model_path}")
            dna_model = EsmModel.from_pretrained(
                args.bio_model_path,
                config=bio_config
            )
            
            # 加载权重时，strict=False以跳过不匹配的部分
            self.model.bio_model.load_state_dict(dna_model.state_dict(), strict=False)
            del dna_model  # Free memory
            
            print("Base models loaded successfully")
        
        # Move model to device and set to evaluation mode
        self.model = self.model.to(torch.bfloat16).to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Helper: tokenize a DNA string to fixed length k
    # ------------------------------------------------------------------

    def _encode_dna_sequence(self, dna_seq: str) -> torch.LongTensor:
        """Tokenize a DNA sequence with DNABERT tokenizer and pad / truncate to k tokens."""
        k = self.args.multimodal_k_tokens
        ids = self.dna_tokenizer(dna_seq, add_special_tokens=False)["input_ids"]
        if len(ids) < k:
            ids += [self.dna_tokenizer.pad_token_id] * (k - len(ids))
        else:
            ids = ids[:k]
        return torch.LongTensor(ids)

    # ------------------------------------------------------------------
    # Sample processing (mimic training-time _process_text)
    # ------------------------------------------------------------------

    def process_sample(self, raw_text: str):
        """Convert raw text containing <dna>…<dna> tags into model-ready tensors.

        The logic strictly follows `IterableMultimodalDNADataSet._process_text`:
        1.  Extract all DNA sequences in order.
        2.  Build `input_ids` as:  [BOS] + (placeholders for each DNA) + encoded clean text + [EOS]
        3.  Return `dna_ids_list` (list[Tensor]) that aligns with the placeholders.
        """

        pattern = r"<dna>\s*([ACGTacgt]+)\s*<dna>"

        # 1) Extract sequences & clean text
        extracted_dnas: List[str] = re.findall(pattern, raw_text)
        clean_text = re.sub(pattern, "", raw_text).strip()

        # 2) Construct input_ids
        input_ids: List[int] = []

        if self.text_tokenizer.bos_token_id is not None:
            input_ids.append(self.text_tokenizer.bos_token_id)

        # Placeholders for each DNA
        start_id = self.text_tokenizer.convert_tokens_to_ids("<|dna_start|>")
        pad_id = self.text_tokenizer.convert_tokens_to_ids("<|dna_pad|>")
        end_id = self.text_tokenizer.convert_tokens_to_ids("<|dna_end|>")

        dna_ids_list: List[torch.LongTensor] = []
        k = self.args.multimodal_k_tokens

        for dna_seq in extracted_dnas:
            input_ids.append(start_id)
            input_ids.extend([pad_id] * k)
            input_ids.append(end_id)
            dna_ids_list.append(self._encode_dna_sequence(dna_seq))

        # Append (clean) text after all DNA placeholders
        if clean_text:
            input_ids.extend(self.text_tokenizer.encode(clean_text, add_special_tokens=False))

        # Truncate & add EOS
        # if len(input_ids) > self.args.max_length - 1:
        #     input_ids = input_ids[: self.args.max_length - 1]
        # input_ids.append(self.text_tokenizer.eos_token_id)

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.LongTensor(input_ids),
            "attention_mask": torch.LongTensor(attention_mask),
            "dna_ids_list": dna_ids_list if dna_ids_list else None,
        }

    # ------------------------------------------------------------------
    # Single-sample inference loop (no padding, no batching)
    # ------------------------------------------------------------------

    def run_dataset(self):
        """Run inference sample-by-sample without padding."""
        args = self.args
        
        # Determine file format and read dataset
        ext = args.dataset_path.split('.')[-1].lower()
        reader = {
            'csv': pd.read_csv,
            'xlsx': pd.read_excel,
            'jsonl': partial(pd.read_json, lines=True),
            'json': partial(pd.read_json, lines=False)
        }.get(ext)
        
        if not reader:
            raise ValueError(f"Unsupported file format: {ext}. Supported formats: csv, xlsx, jsonl, json")
        
        print(f"Reading dataset from {args.dataset_path}")
        df = reader(args.dataset_path)
        
        # Limit number of samples if specified
        if args.max_samples:
            df = df.head(args.max_samples)
        
        # Make sure 'input' column exists
        if 'input' not in df.columns:
            raise ValueError("Dataset must contain an 'input' column")
        
        print(f"Processing {len(df)} samples one-by-one…")
        results = []

        for idx, text in tqdm(enumerate(df['input'].tolist()), total=len(df), desc="Infer"):
            sample = self.process_sample(text)

            # Prepare tensors without padding (batch dimension = 1)
            input_ids = sample['input_ids'].unsqueeze(0).to(self.device)
            attention_mask = sample['attention_mask'].unsqueeze(0).to(self.device)
            dna_ids_lists = [sample.get('dna_ids_list', None)]

            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        dna_ids_lists=dna_ids_lists,
                        attention_mask=attention_mask,
                        max_length=args.max_length,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        do_sample=True,
                    )

                decoded = self.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
                results.append(decoded)

                if idx <= 10:
                    print("\nSample output:")
                    print(decoded[:500] + "..." if len(decoded) > 500 else decoded)
                    print("-" * 50)
            except Exception as e:
                print(f"Error during generation: {str(e)}")

        # Add results to dataframe and save
        result_df = df.head(len(results)).copy()
        result_df['output'] = results
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_path, exist_ok=True)
        
        # Save to output file
        output_file = os.path.join(args.output_path, f"predictions_{len(results)}_samples.jsonl")
        result_df.to_json(output_file, orient='records', lines=True, force_ascii=False)
        print(f"Saved predictions to {output_file}")


if __name__ == "__main__":
    args = parse_args()
    inferer = MultiModalInfer(args)
    inferer.run_dataset()
