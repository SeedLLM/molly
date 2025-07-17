import os
import pandas as pd
import torch
import numpy as np
from typing import Dict, Any
from torch.utils.data import Dataset
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial


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
        num_workers=4,
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
            num_workers: Number of workers for data loading.
            **kwargs: Additional arguments.
        """

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.parquet_file = parquet_file
        self.tokenizer = tokenizer
        self.dna_tokenizer = multimodal_tokenizer
        self.dataset_config = dataset_config
        self.project_token_num = kwargs.get("multimodal_k_tokens", 32)
        self.shuffle = shuffle
        self.seed = seed
        self.num_workers = num_workers

        # Configuration parameters
        self.max_len = dataset_config.max_len
        self.max_src_len = dataset_config.max_src_len
        self.mode = dataset_config.mode
        self.cal_metric_pos = dataset_config.cal_metric_pos
        self.padding = dataset_config.padding

        # Special tokens
        self._precompute_token_ids()

        # 预定义固定内容的分词结果
        self.system_prompt_ids = self.tokenizer.encode(
            "<|im_start|>system\nYou are a helpful knowledgeable and precise biomedical assistant.<|im_end|>\n<|im_start|>user\n",
            add_special_tokens=False
        )
        self.assistant_start_ids = self.tokenizer.encode(
            "<|im_end|>\n<|im_start|>assistant\n", 
            add_special_tokens=False
        )

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

        print(f"Preprocessing {len(df)} samples with {self.num_workers} workers...")

        with Pool(self.num_workers) as pool:
            results = pool.imap(
                partial(self._preprocess_sample, tokenizer=self.tokenizer),
                df.to_dict('records'),
                chunksize=min(100, max(1, len(df) // (self.num_workers * 10)))
            )
            self.data = []
            with tqdm(total=len(df), desc="Preprocessing", unit="sample") as pbar:
                for result in results:
                    self.data.append(result)
                    pbar.update(1)
        
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
        processed = self.process_sample(sample)
        assert len(processed['dna_ids_list']) == len(processed['dna_start_pos_list']), \
            f"Mismatch in DNA IDs and start positions for sample {idx}: {len(processed['dna_ids_list'])} vs {len(processed['dna_start_pos_list'])}"
        return processed

    def _precompute_token_ids(self):
        """预计算所有特殊token的ID"""
        self.dna_start_id = self.tokenizer.convert_tokens_to_ids("<|dna_start|>")
        self.dna_end_id = self.tokenizer.convert_tokens_to_ids("<|dna_end|>")
        self.dna_pad_id = self.tokenizer.convert_tokens_to_ids("<|dna_pad|>")
        self.rna_start_id = self.tokenizer.convert_tokens_to_ids("<|rna_start|>")
        self.rna_end_id = self.tokenizer.convert_tokens_to_ids("<|rna_end|>")
        self.rna_pad_id = self.tokenizer.convert_tokens_to_ids("<|rna_pad|>")
        self.eos_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id

    @staticmethod
    def _preprocess_sample(sample: dict, tokenizer) -> dict:
        """
        Format a Parquet example into DNA-LLM format suitable for processing.
        
        The Parquet structure already has:
        - "input": Clean text (tags removed)
        - "sequence": List of extracted sequences
        - "kind": Hyphen-separated types like "dna" or "dna-rna"
        """
        sequences = sample.get("sequence", [])
        kinds_string = sample.get("kind", "").lower()
        kinds = kinds_string.split("-") if kinds_string else []
        
        # 预处理文本内容
        input_text = sample.get("input", "").strip()
        output_text = sample.get("output", "").strip()
        reasoning = sample.get("think", "").strip()
        
        # 提前分词文本内容
        input_token_ids = tokenizer.encode(input_text, add_special_tokens=False) if input_text else []
        output_token_ids = tokenizer.encode(output_text, add_special_tokens=False) if output_text else []
        reasoning_token_ids = tokenizer.encode(reasoning, add_special_tokens=False) if reasoning else []
        
        # 处理序列数据
        sequence_data = []
        for i, seq in enumerate(sequences):
            seq_type = kinds[i] if i < len(kinds) else "dna"
            # 仅存储原始序列，编码将在__getitem__中批量处理
            sequence_data.append({
                "sequence": seq, 
                "type": seq_type,
                "length": len(seq)
            })
        
        return {
            "input_token_ids": input_token_ids,
            "output_token_ids": output_token_ids,
            "reasoning_token_ids": reasoning_token_ids,
            "sequence_data": sequence_data,
            "task": sample.get("task", ""),
            "kind": kinds_string,
            "label": sample.get("label", "")
        }

    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Process a sample into model-ready format with tokenized sequences.
        """
        input_ids = list(self.system_prompt_ids)

        dna_ids_list = []
        dna_start_pos_list = []

        
        # Insert sequences immediately after user start tag
        for seq_info in sample["sequence_data"]:
            seq = seq_info["sequence"]
            seq_type = seq_info["type"]

            encoded_seq = self._encode_sequence(seq, seq_type)
            dna_ids_list.append(encoded_seq)

            dna_start_pos_list.append(len(input_ids))

            # Determine which special tokens to use based on sequence type
            start_token_id = self.dna_start_id if seq_type == "dna" else self.rna_start_id
            pad_token_id = self.dna_pad_id if seq_type == "dna" else self.rna_pad_id
            end_token_id = self.dna_end_id if seq_type == "dna" else self.rna_end_id

            # Add start token, padding, and end token
            input_ids.extend([
                start_token_id,
                *[pad_token_id] * self.project_token_num,
                end_token_id
            ])

        # 添加预处理的用户输入
        input_ids.extend(sample["input_token_ids"])

        # 添加助手起始标记
        input_ids.extend(self.assistant_start_ids)

        # Process output based on mode
        if self.mode == 'sft':
            output_ids = list(sample["output_token_ids"])
        else:
            output_ids = []

        # Add EOS token based on mode
        if self.mode == 'pretrain':
            input_ids.append(self.eos_id)
        else:
            output_ids.append(self.eos_id)
        
        input_len = len(input_ids)
        input_ids.extend(output_ids)
        
        # Create labels
        # Use -100 to ignore input tokens in loss calculation
        labels = [-100] * input_len + output_ids if self.mode == 'sft' else input_ids.copy()
        
        # Truncate if necessary
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len-1] + [self.eos_id]
            labels = labels[:self.max_len-1] + [self.eos_id]
        
        # Calculate metric position
        cal_metric_pos = None
        if self.cal_metric_pos is not None:
            cal_metric_pos = input_len + 1 + self.cal_metric_pos
        elif len(output_ids) > 0:
            cal_metric_pos = input_len + 1
        
        attention_mask = [1] * len(input_ids)

        # Add padding if needed
        if self.padding and (pad_len := self.max_len - len(input_ids)) > 0:
            input_ids.extend([self.pad_id] * pad_len)
            labels.extend([-100] * pad_len)
            attention_mask.extend([0] * pad_len)


        # Convert to tensors
        return {
            "input_ids": torch.LongTensor(input_ids),
            "dna_ids_list":  torch.stack(dna_ids_list) if dna_ids_list else None,
            "dna_start_pos_list": torch.LongTensor(dna_start_pos_list) if dna_start_pos_list else None,
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

        
        encoding = self.dna_tokenizer(
            seq, 
            padding='max_length',
            max_length=self.project_token_num,
            truncation=True,
            return_tensors='pt'
        )
        return encoding['input_ids'].squeeze(0)


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
    cal_metric_pos = [sample.get("cal_metric_pos") for sample in batch]
    dna_start_pos_lists = [sample.get("dna_start_pos_list", []) for sample in batch]
    dna_ids = [sample.get("dna_ids_list", None) for sample in batch]
    
    dna_counts = [len(dna_ids_list) for dna_ids_list in dna_start_pos_lists]
        
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )
    dna_ids = torch.nn.utils.rnn.pad_sequence(
        dna_ids, batch_first=True, padding_value=0
    ) if dna_ids else None

    dna_start_pos_lists = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(pos) for pos in dna_start_pos_lists],
        batch_first=True, padding_value=-1
    ) if dna_start_pos_lists else None
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "dna_ids_list": dna_ids,
        "dna_start_pos_list": dna_start_pos_lists,
        "dna_counts": dna_counts,
        "cal_metric_pos": cal_metric_pos,
    }


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
            
            # print(f"Sample {sample_idx} raw data:")
            # print(f"- Input: {raw_sample['input'][:100]}...")
            # print(f"- Output: {raw_sample['output'][:100]}...")
            # print(f"- Sequences: {len(raw_sample['sequence_data'])} sequences")
            # for i, seq_data in enumerate(raw_sample['sequence_data']):
            #     print(f"  - Seq {i+1}: {seq_data['type']} - {seq_data['sequence'][:30]}...")
            
            print(f"\nProcessed sample:")
            print(f"- Input IDs shape: {processed_sample['input_ids'].shape}")
            print(f"- Labels shape: {processed_sample['labels'].shape}")
            print(f"- Attention mask shape: {processed_sample['attention_mask'].shape}")
            
            if processed_sample['dna_ids_list'] is not None:
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
            sample_input_ids = batch['input_ids'][1]
            
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
