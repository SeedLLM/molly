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
from src.dataset.dna_rna_dataset import DNARNADataset, qwen_dna_collate_fn, DatasetConfig, qwen_dna_collate_fn_inference


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
    parser.add_argument("--use_lora", action="store_true",
                        help="Whether to load LoRA weights (if available)")
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
        
        # 检查是否使用LoRA
        if args.use_lora:
            print("LoRA mode enabled, checking for LoRA weights...")
            lora_path = os.path.join(args.trained_model_path, "lora_weights")
            
            if os.path.exists(lora_path):
                print(f"Found LoRA adapter at {lora_path}")
                try:
                    # 加载基础模型
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
                    
                    # 加载LoRA适配器
                    print(f"Applying LoRA adapter from {lora_path}")
                    # config = PeftConfig.from_pretrained(lora_path)
                    self.model.model = PeftModel.from_pretrained(self.model.model, lora_path)
                    
                    # 加载多模态投影器权重（如果存在）
                    projector_path = os.path.join(args.trained_model_path, "pytorch_model.bin")
                    if os.path.exists(projector_path):
                        print(f"Loading projector weights from {projector_path}")
                        try:
                            state_dict = torch.load(projector_path, map_location="cpu")
                            projector_dict = {}
                            for key, value in state_dict.items():
                                if "multimodal_projector" in key:
                                    new_key = key.replace("multimodal_projector.", "")
                                    projector_dict[new_key] = value
                            
                            if projector_dict:
                                missing, unexpected = self.model.multimodal_projector.load_state_dict(projector_dict, strict=False)
                                print(f"Projector weights loaded. Missing keys: {missing}, Unexpected keys: {unexpected}")
                                print(f"Projector weights loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
                        except Exception as e:
                            print(f"Error loading projector weights: {e}")
                            print(traceback.format_exc())

                except Exception as e:
                    print(f"Error loading LoRA model: {e}")
                    print(traceback.format_exc())
                    print("Falling back to standard model loading...")
                    self._load_standard_model()
            else:
                print(f"LoRA adapter not found at {lora_path}, falling back to standard model...")
                self._load_standard_model()
        else:
            # 标准模型加载
            self._load_standard_model()
            
        # Move model to device and set to evaluation mode
        self.model = self.model.to(torch.bfloat16).to(self.device)
        self.model.eval()

    def _load_standard_model(self):
        args = self.args
        print(f"Loading standard checkpoint from {args.trained_model_path}")
        
        try:
            # 尝试直接加载权重
            ckpt_path = os.path.join(args.trained_model_path, "pytorch_model.bin")
            if not os.path.exists(ckpt_path):
                ckpt_path = args.trained_model_path
                
            ckpt = torch.load(ckpt_path, map_location="cpu")
            
            # 检查是否为DeepSpeed检查点
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            else:
                state_dict = ckpt
                
            # 加载状态字典
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
            # model_config.bio_config = bio_config
            
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
        test_dataset = DNARNADataset(args.dataset_path, 
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

                # Print the sample data for debugging
                print(sample_data)  # Check the content before writing to the file

                # Write the sample data to the JSON file
                json.dump(sample_data, json_file, ensure_ascii=False)
                json_file.write("\n")  # Add a newline for better separation between records
                json_file.flush()

        json_file.close()


if __name__ == "__main__":
    args = parse_args()
    inferer = MultiModalInfer(args)
    inferer.run_dataset()
