import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

from dataset.omics_dataset import (
    DatasetConfig,
    OmicsDataset,
    qwen_omics_collate_fn_inference,
)
from model.config import get_omics_one_config

# pylint: disable=no-name-in-module
from model.omics_one import OmicsOne
from utils import str2bool


def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal LLM Inference")
    parser.add_argument(
        "--local-rank",
        type=int,
        default=0,
        help="Local rank for distributed inference, ignored if single GPU",
    )
    parser.add_argument(
        "--text-model-path", type=str, required=True, help="Path to the Qwen model"
    )
    parser.add_argument(
        "--dna-rna-model-path",
        type=str,
        required=True,
        help="Path to the DNA-BERT model",
    )
    parser.add_argument(
        "--dna-rna-k-tokens",
        type=int,
        default=64,
        help="Number of tokens for DNA sequence projection",
    )
    parser.add_argument(
        "--protein-model-path",
        type=str,
        default=None,
        help="Path to the protein encoder checkpoint",
    )
    parser.add_argument(
        "--protein-k-tokens",
        type=int,
        default=64,
        help="Number of tokens for protein sequence projection",
    )
    parser.add_argument(
        "--trained-model-path",
        type=str,
        required=True,
        help="Path to trained multimodal model checkpoint",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Whether to load LoRA weights (if available)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to dataset file (jsonl/csv/xlsx/json) for batch inference",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum tokens to generate per example",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for dataset inference"
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (None=all)",
    )
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--json-file", type=str)
    # Optimization
    parser.add_argument("--attn_impl",
                        type=str,
                        default='flash_attention_2',
                        choices=['sdpa', 'flash_attention_2', 'flash_attention_3'],
                        help="FlashAttn Implementation, support sdpa, flash_attention_2 or flash_attention_3")
    
    parser.add_argument("--use_liger",
                        type=str2bool,
                        default=False,
                        help="Whether to use liger for optimizer state offload, see https://github.com/linkedin/Liger-Kernel")
    return parser.parse_args()


class MultiModalInfer:
    """Batch inference helper for Qwen + NT multimodal model (training-style DNA injection)."""

    def __init__(self, arg):
        self.args = arg

        # ‑- Seed ----------------------------------------------------------------
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)

        # ‑- Device --------------------------------------------------------------
        if not torch.cuda.is_available():
            raise Exception(f'cuda not available, please check env.')
        self.device = torch.device(
            "cuda:0"
            if self.args.device == "cuda" and torch.cuda.is_available()
            else "cpu"
        )

        # ‑- Tokenisers ----------------------------------------------------------
        print("Loading tokenizers…")
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            self.args.text_model_path, trust_remote_code=True
        )
        self.dna_rna_tokenizer = AutoTokenizer.from_pretrained(
            self.args.dna_rna_model_path, trust_remote_code=True
        )
        self.protein_tokenizer = AutoTokenizer.from_pretrained(
            self.args.protein_model_path, trust_remote_code=True
        )

        # Ensure DNA special tokens exist in text tokenizer
        special_tokens = [
            "<|dna_start|>",
            "<|dna_pad|>",
            "<|dna_end|>",
            "<|rna_start|>",
            "<|rna_pad|>",
            "<|rna_end|>",
            "<|protein_start|>",
            "<|protein_pad|>",
            "<|protein_end|>",
        ]
        self.text_tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}
        )

        # ‑- Model --------------------------------------------------------------
        print("Loading multimodal model…")
        self._load_model()

    def _load_model(self):
        model_config = get_omics_one_config(
            self.args.text_model_path,
            self.args.dna_rna_model_path,
            self.args.protein_model_path,
        )
        model_config.dna_rna_project_token_num = self.args.dna_rna_k_tokens
        model_config.protein_project_token_num = self.args.protein_k_tokens
        print(
            f"Setting dna_rna project_token_num to {model_config.dna_rna_project_token_num}"
        )
        print(
            f"Setting protein project_token_num to {model_config.protein_project_token_num}"
        )

        with torch.device("cpu"):
            self.model = OmicsOne(config=model_config)
        self.model.set_special_tokens(self.text_tokenizer)

        print(f"Loading base Qwen model from {self.args.text_model_path}")
        qwen_model = AutoModelForCausalLM.from_pretrained(
            args.text_model_path,
            torch_dtype="auto",
            trust_remote_code=True,
            attn_implementation=args.attn_impl,
        )
        if args.use_liger:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen3
            apply_liger_kernel_to_qwen3()
        self.model.model = qwen_model

        print(f"Loading NT model from {self.args.dna_rna_model_path}")
        dna_rna_model = AutoModelForMaskedLM.from_pretrained(
            args.dna_rna_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model.dna_rna_model = dna_rna_model

        protein_model = AutoModelForMaskedLM.from_pretrained(
            args.protein_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model.protein_model = protein_model

        # 检查是否使用LoRA
        if self.args.use_lora:
            from peft import PeftModel
            print("LoRA mode enabled, checking for LoRA weights...")
            lora_path = self.args.trained_model_path

            print("LoRA mode enabled")
            self.model.model = PeftModel.from_pretrained(
                self.model.model, lora_path, torch_dtype=torch.bfloat16
            )

            dna_rna_projector_path = os.path.join(
                self.args.trained_model_path, "dna_rna_projector.bin"
            )
            if os.path.exists(dna_rna_projector_path):
                self.model.dna_rna_projector.load_state_dict(
                    torch.load(dna_rna_projector_path, map_location="cpu")
                )
                print("dna_rna projector loaded.")

            protein_projector_path = os.path.join(
                self.args.trained_model_path, "protein_projector.bin"
            )
            if os.path.exists(protein_projector_path):
                self.model.protein_projector.load_state_dict(
                    torch.load(protein_projector_path, map_location="cpu")
                )
                print("dna_rna projector loaded.")

        else:
            print("LoRA mode not enabled, loading base model...")
            train_model_path = os.path.join(
                self.args.trained_model_path, "pytorch_model.bin"
            )
            print(train_model_path)
            if os.path.exists(train_model_path):
                self.model.load_state_dict(
                    torch.load(train_model_path, map_location="cpu")
                )
                print("Multimodal loaded.")

        # Move model to device and set to evaluation mode
        self.model = self.model.to(torch.bfloat16).to(self.device)
        self.model.eval()

    def run_dataset(self):
        """Run inference sample-by-sample without padding."""
        os.makedirs(os.path.dirname(self.args.json_file), exist_ok=True)
        with open(self.args.json_file, "a", encoding="utf-8") as json_file:

            test_config = DatasetConfig(
                max_len=self.args.max_length,
                max_src_len=self.args.max_length,
                mode="sft",
                padding=True,
                input_field="input",
                output_field="output",
                dna_rna_k_tokens=self.args.dna_rna_k_tokens,
                protein_k_tokens=self.args.protein_k_tokens,
                type="Test",
            )
            test_dataset = OmicsDataset(
                self.args.dataset_path,
                self.text_tokenizer,
                dataset_config=test_config,
                dna_rna_tokenizer=self.dna_rna_tokenizer,
                protein_tokenizer=self.protein_tokenizer,
                num_workers=4,
                type="Test",
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.args.batch_size,
                collate_fn=qwen_omics_collate_fn_inference,
            )

            for _, batch in tqdm(
                enumerate(test_dataloader), total=len(test_dataloader), desc="Infer"
            ):
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=batch["input_ids"].to(self.device),
                        attention_mask=batch["attention_mask"].to(self.device),
                        omic_ids=batch["omic_ids"],
                        omic_info_list=batch["omic_info_list"],
                        do_sample=True,
                        max_length=self.args.max_length,
                        temperature=self.args.temperature,
                        top_p=self.args.top_p,
                        top_k=self.args.top_k,
                        repetition_penalty=self.args.repetition_penalty,
                    )
                # input_text = self.text_tokenizer.batch_decode(
                #     batch["input_ids"].to(self.device), skip_special_tokens=True
                # )
                # print("Input text:", input_text)

                decoded = self.text_tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )

                for i, value in enumerate(decoded):
                    sample_data = {
                        "decoded_output": value,
                        "input": batch["input"][i],
                        "gt_output": batch["raw_output"][i], 
                        "gt_label": batch["raw_label"][i],
                        "task": batch["raw_task"][i],
                        "kind": batch["raw_kind"][i],
                    }

                    # Write the sample data to the JSON file
                    json.dump(sample_data, json_file, ensure_ascii=False)
                    json_file.write("\n")
                    json_file.flush()

        json_file.close()


if __name__ == "__main__":
    args = parse_args()
    inferer = MultiModalInfer(args)
    inferer.run_dataset()
