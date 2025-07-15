import os
import pandas as pd
import torch
import numpy as np
import re
from typing import Dict, Any, List, Tuple, Optional
from torch.utils.data import Dataset


class DNARNADataset(Dataset):
    """Dataset for DNA/RNA data from Parquet, formatted for Bio-LLM."""

    def __init__(
        self,
        parquet_file: str,
        tokenizer,
        dataset_config,
        multimodal_tokenizer=None,
        read_nums=None,
        shuffle=False,
        seed=42,
        **kwargs
    ):
        """
        Initialize the dataset by loading a Parquet file and formatting each example.

        Args:
            parquet_file: Path to the Parquet file.
            tokenizer: Text tokenizer for processing text parts.
            dataset_config: Configuration for the dataset.
            multimodal_tokenizer: Tokenizer for DNA/RNA sequences.
            read_nums: Maximum number of samples to read.
            shuffle: Whether to shuffle the dataset.
            seed: Random seed for shuffling.
            **kwargs: Additional arguments.
        """
        self.parquet_file = parquet_file
        self.tokenizer = tokenizer
        self.dna_tokenizer = multimodal_tokenizer
        self.dataset_config = dataset_config
        self.project_token_num = kwargs.get("multimodal_k_tokens", 32)
        self.shuffle = shuffle
        self.seed = seed

        # Configuration parameters
        self.max_len = dataset_config.max_len
        self.max_src_len = dataset_config.max_src_len
        self.mode = dataset_config.mode
        self.cal_metric_pos = dataset_config.cal_metric_pos
        self.padding = dataset_config.padding

        # Special tokens for DNA and RNA sequences
        self.dna_start_token = "<|dna_start|>"
        self.dna_end_token = "<|dna_end|>"
        self.dna_pad_token = "<|dna_pad|>"
        self.rna_start_token = "<|rna_start|>"
        self.rna_end_token = "<|rna_end|>"
        self.rna_pad_token = "<|rna_pad|>"

        # Load data
        print(f"Loading parquet data from {parquet_file}")
        df = pd.read_parquet(parquet_file)
        
        # Limit samples if specified
        if read_nums:
            df = df.head(read_nums)
            
        # Shuffle if needed
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            df = df.sample(frac=1, random_state=rng).reset_index(drop=True)
        
        self.data = []
        for _, row in df.iterrows():
            formatted = self._format_parquet_example(row)
            self.data.append(formatted)
        
        print(f"Loaded {len(self.data)} samples from parquet file")

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a specific item from the dataset."""
        # Ensure index is valid
        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"Index {idx} out of bounds for dataset with {len(self.data)} items")
            
        sample = self.data[idx]
        return self.process_sample(sample)
    
    def _format_parquet_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a Parquet example into DNA-LLM format suitable for processing.
        
        The Parquet structure already has:
        - "input": Clean text (tags removed)
        - "sequence": List of extracted sequences
        - "kind": Hyphen-separated types like "dna" or "dna-rna"
        """
        # Get sequences and kinds from parquet
        sequences = example.get("sequence", [])
        kinds_string = example.get("kind", "").lower()
        kinds = kinds_string.split("-") if kinds_string else []
        
        # Clean text and output from parquet
        input_text = example.get("input", "").strip()
        output_text = example.get("output", "").strip()
        reasoning = example.get("think", "").strip()
        
        # Store sequence types along with sequences
        sequence_data = []
        for i, seq in enumerate(sequences):
            seq_type = kinds[i] if i < len(kinds) else "dna"  # Default to DNA if type not specified
            sequence_data.append({"sequence": seq, "type": seq_type})
        
        return {
            "input": input_text,
            "output": output_text,
            "reasoning": reasoning,
            "sequence_data": sequence_data,  # Store sequences with their types
            "task": example.get("task", ""),
            "kind": kinds_string,
            "label": example.get("label", "")
        }
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Process a sample into model-ready format with tokenized sequences.
        """
        input_ids = []
        dna_ids_list = []
        dna_start_pos_list = []
        attention_mask = []

        input_text = sample["input"]
        output_text = sample["output"]
        sequence_data = sample["sequence_data"]

        # Add Qwen system prompt format
        input_ids.extend(self.tokenizer.encode("<|im_start|>system\nYou are a helpful knowledgeable and precise biomedical assistant.<|im_end|>\n<|im_start|>user\n", add_special_tokens=False))
        
        # Insert sequences immediately after user start tag
        for seq_info in sequence_data:
            seq = seq_info["sequence"]
            seq_type = seq_info["type"]
            # Determine which special tokens to use based on sequence type
            if seq_type == "rna":
                start_token = self.rna_start_token
                pad_token = self.rna_pad_token
                end_token = self.rna_end_token
            else:  # Default to DNA
                start_token = self.dna_start_token
                pad_token = self.dna_pad_token
                end_token = self.dna_end_token
            
            # Encode sequence
            encoded_seq = self._encode_sequence(seq, seq_type)
            dna_ids_list.append(encoded_seq)
            
            # Add placeholders
            dna_start_pos_list.append(len(input_ids))
            input_ids.extend([
                self.tokenizer.convert_tokens_to_ids(start_token),
                *[self.tokenizer.convert_tokens_to_ids(pad_token)] * self.project_token_num,
                self.tokenizer.convert_tokens_to_ids(end_token)
            ])
        
        # Add input text after all sequences
        if input_text:
            input_ids.extend(self.tokenizer.encode(input_text, add_special_tokens=False))
        
        # End user input and start assistant
        input_ids.extend(self.tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False))

        # Process output based on mode
        if self.mode == 'sft':
            output_ids = self.tokenizer.encode(output_text, add_special_tokens=False)
        else:
            output_ids = []

        # Add EOS token based on mode
        if self.mode == 'pretrain':
            input_ids.append(self.tokenizer.eos_token_id)
        else:
            output_ids.append(self.tokenizer.eos_token_id)
        
        input_len = len(input_ids)
        input_ids.extend(output_ids)
        
        # Create labels
        if self.mode == 'sft':
            labels = [-100] * input_len + output_ids  # Use -100 to ignore input tokens in loss calculation
        else:
            labels = input_ids.copy()
        
        # Truncate if necessary
        if len(input_ids) > self.max_len:
            eos_token = input_ids[-1]
            input_ids = input_ids[:self.max_len - 1] + [eos_token]
            labels = labels[:self.max_len - 1] + [self.tokenizer.eos_token_id]
        
        # Calculate metric position
        cal_metric_pos = None
        if self.cal_metric_pos is not None:
            cal_metric_pos = input_len + 1 + self.cal_metric_pos
        elif len(output_ids) > 0:
            cal_metric_pos = input_len + 1
        
        attention_mask = [1] * len(input_ids)

        # Add padding if needed
        if self.padding:
            pad_len = self.max_len - len(input_ids)
            if pad_len > 0:
                input_ids.extend([self.tokenizer.pad_token_id] * pad_len)
                labels.extend([-100] * pad_len)
                attention_mask.extend([0] * pad_len)

        # Convert to tensors
        return {
            "input_ids": torch.LongTensor(input_ids),
            "dna_ids_list": [torch.LongTensor(dna_ids) for dna_ids in dna_ids_list] if dna_ids_list else None,
            "labels": torch.LongTensor(labels),
            "attention_mask": torch.LongTensor(attention_mask),
            "cal_metric_pos": cal_metric_pos,
        }
    
    def _encode_sequence(self, seq: str, seq_type: str) -> torch.LongTensor:
        """
        Tokenize a DNA/RNA sequence and pad/truncate to fixed length.
        """
        if not self.dna_tokenizer:
            raise ValueError("DNA/RNA tokenizer is required but not provided")
            
        ids = self.dna_tokenizer(seq, add_special_tokens=False)["input_ids"]
        
        # Pad or truncate to fixed length
        if len(ids) < self.project_token_num:
            ids += [self.dna_tokenizer.pad_token_id] * (self.project_token_num - len(ids))
        else:
            ids = ids[:self.project_token_num]
            
        return torch.LongTensor(ids)


def qwen_dna_collate_fn(batch):
    """
    Collate function for DataLoader with multimodal DNA batches.
    Handles variable length DNA sequences and attention masks.
    
    Args:
        batch: List of samples from the dataset
    
    Returns:
        Batched tensors suitable for model input
    """
    input_ids = [sample["input_ids"] for sample in batch]
    labels = [sample["labels"] for sample in batch]
    attention_mask = [sample["attention_mask"] for sample in batch]
    
    # Handle metric positions
    cal_metric_pos = [sample.get("cal_metric_pos", None) for sample in batch]
    
    # Get all DNA sequences from all samples
    all_dna_ids = []
    dna_counts = []
    
    for sample in batch:
        dna_ids_list = sample.get("dna_ids_list", [])
        dna_counts.append(len(dna_ids_list) if dna_ids_list else 0)
        if dna_ids_list:
            all_dna_ids.extend(dna_ids_list)
    
    # Pad sequences
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    
    # Create return dict
    result = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }
    
    # Add DNA sequences if they exist
    if sum(dna_counts) > 0:
        result["dna_ids_list"] = all_dna_ids
        result["dna_counts"] = dna_counts
    
    # Add cal_metric_pos if any non-None values
    if any(pos is not None for pos in cal_metric_pos):
        result["cal_metric_pos"] = cal_metric_pos
    
    return result


def format_parquet_for_bio_llm(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a Parquet example (from <dna>/<rna>标注的数据) into DNA-LLM chat format.
    """
    # 安全提取序列和种类
    sequences = example.get("sequence", [])
    kinds = example.get("kind", "").lower().split("-")

    # 构建 DNA 位置占位（与 KEGG 格式保持一致）
    dna_placeholder_count = len(sequences)
    dna_placeholders = [{"type": "bio", "text": None} for _ in range(dna_placeholder_count)]

    return {
        "prompt": [
            {
                "role": "user",
                "content": [
                    *dna_placeholders,
                    {"type": "text", "text": example.get("input", "").strip()},
                ],
            },
            {
                "role": "assistant",
                "reasoning_content": example.get("reasoning", "").strip(),
                "content": [
                    {"type": "text", "text": f"Answer: {example.get('output', '').strip()}"},
                ],
            },
        ],
        "bio_sequences": sequences,
        "answer": example.get("output", "").strip(),
        "label": example.get("label", "").strip(),
        "task": example.get("task", ""),
        "kind": example.get("kind", ""),
    }


if __name__ == "__main__":

    # parquet_path = "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/stage3_train_data/stage3_train_nt.parquet"

    # # Step 1: 加载 parquet 文件为 Hugging Face Dataset
    # dataset = load_dataset("parquet", data_files=parquet_path)["train"]

    # # Step 2: 映射到标准 DNA-LLM 格式
    # dataset = dataset.map(format_parquet_for_bio_llm)

    # # Step 3: 打印一条样本进行检查
    # print(dataset[2])
    import sys
    import torch
    from transformers import AutoTokenizer
    import argparse
    from dataclasses import dataclass
    from pathlib import Path
    
    # Simple config class for testing
    @dataclass
    class TestConfig:
        max_len: int = 1024
        max_src_len: int = 1024
        mode: str = 'sft'
        cal_metric_pos: int = None
        padding: bool = True
        input_field: str = 'input'
        output_field: str = 'output'
    
    def run_test_dataset():
        """Full test case for the DNA/RNA dataset"""
        # Parse arguments
        parser = argparse.ArgumentParser(description="Test DNA/RNA Dataset")
        parser.add_argument("--parquet_path", type=str, 
                            default="/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/stage3_train_data/stage3_train_nt.parquet")
        parser.add_argument("--text_model_path", type=str, default="/tos-bjml-ai4agr/lijinzhe/BioMLLM/Qwen3-4B")
        parser.add_argument("--bio_model_path", type=str, default="/tos-bjml-ai4agr/lijinzhe/BioModel/nucleotide-transformer/")
        parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to test")
        args = parser.parse_args()
        
        # Check if files exist
        parquet_path = Path(args.parquet_path)
        if not parquet_path.exists():
            print(f"Error: Parquet file not found at {parquet_path}")
            sys.exit(1)
        
        print(f"Loading tokenizers from {args.text_model_path} and {args.bio_model_path}")
        
        try:
            # Load tokenizers
            text_tokenizer = AutoTokenizer.from_pretrained(args.text_model_path, trust_remote_code=True)
            dna_tokenizer = AutoTokenizer.from_pretrained(args.bio_model_path, trust_remote_code=True)
            
            # Add DNA and RNA special tokens to text tokenizer
            special_tokens = [
                "<|dna_start|>", "<|dna_pad|>", "<|dna_end|>",
                "<|rna_start|>", "<|rna_pad|>", "<|rna_end|>"
            ]
            text_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            
            # Create test config
            config = TestConfig()
            
            # Create dataset
            print(f"Creating dataset from {parquet_path}")
            dataset = DNARNADataset(
                args.parquet_path,
                text_tokenizer,
                config,
                multimodal_tokenizer=dna_tokenizer,
                read_nums=None
            )
            
            # Print dataset info
            print(f"\n{'='*50}")
            print(f"Dataset loaded with {len(dataset)} samples")
            print(f"{'='*50}\n")
            
            # Test individual sample processing
            sample_idx = 5000
            raw_sample = dataset.data[sample_idx]
            processed_sample = dataset[sample_idx]
            
            print(f"Sample {sample_idx} raw data:")
            print(f"- Input: {raw_sample['input'][:100]}...")
            print(f"- Output: {raw_sample['output'][:100]}...")
            print(f"- Sequences: {len(raw_sample['sequence_data'])} sequences")
            for i, seq_data in enumerate(raw_sample['sequence_data']):
                print(f"  - Seq {i+1}: {seq_data['type']} - {seq_data['sequence'][:30]}...")
            
            print(f"\nProcessed sample:")
            print(f"- Input IDs shape: {processed_sample['input_ids'].shape}")
            print(f"- Labels shape: {processed_sample['labels'].shape}")
            print(f"- Attention mask shape: {processed_sample['attention_mask'].shape}")
            
            if processed_sample['dna_ids_list']:
                print(f"- DNA sequences: {len(processed_sample['dna_ids_list'])}")
                for i, seq in enumerate(processed_sample['dna_ids_list']):
                    print(f"  - Seq {i+1} shape: {seq.shape}")
            
            # Test dataloader
            print(f"\n{'='*50}")
            print(f"Testing DataLoader with batch size 2")
            print(f"{'='*50}\n")
            
            dataloader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=2, 
                collate_fn=qwen_dna_collate_fn
            )
            
            batch = next(iter(dataloader))
            print(f"Batch keys: {batch.keys()}")
            print(f"- input_ids shape: {batch['input_ids'].shape}")
            print(f"- labels shape: {batch['labels'].shape}")
            print(f"- attention_mask shape: {batch['attention_mask'].shape}")
            
            if 'dna_ids_list' in batch:
                print(f"- DNA sequences in batch: {len(batch['dna_ids_list'])}")
                print(f"- DNA counts per sample: {batch['dna_counts']}")
                
            # Decode a sample to verify
            print(f"\n{'='*50}")
            print(f"Decoding first sample in batch")
            print(f"{'='*50}\n")
            
            # Get the first sample from the batch
            sample_input_ids = batch['input_ids'][4500]
            
            # Decode tokens and print
            decoded_text = text_tokenizer.decode(sample_input_ids)
            print(f"Decoded text: {decoded_text}")
            
            print(f"\n{'='*50}")
            print("Test completed successfully!")
            print(f"{'='*50}")
            
        except Exception as e:
            import traceback
            print(f"Error during testing: {str(e)}")
            traceback.print_exc()
            sys.exit(1)
    
    # Run the test
    if len(sys.argv) > 1 and sys.argv[1] == "--huggingface":
        # Import for HuggingFace dataset testing
        from datasets import load_dataset
        
        parquet_path = "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/stage3_train_data/stage3_train_nt.parquet"
        # Step 1: 加载 parquet 文件为 Hugging Face Dataset
        dataset = load_dataset("parquet", data_files=parquet_path)["train"]
        # Step 2: 映射到标准 DNA-LLM 格式
        dataset = dataset.map(format_parquet_for_bio_llm)
        # Step 3: 打印一条样本进行检查
        print(dataset[2])
    else:
        # Run the full test
        run_test_dataset()
