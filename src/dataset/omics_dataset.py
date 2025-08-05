import os
import re
from tkinter import N
import pandas as pd
import torch
import numpy as np
from typing import Dict, Any, List
from torch.utils.data import Dataset
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """OmicsDataset所需的配置类"""
    max_len: int = 1024
    max_src_len: int = 1024
    mode: str = 'sft'
    cal_metric_pos: int = None
    padding: bool = True
    input_field: str = 'input'
    output_field: str = 'output'
    dna_rna_k_tokens: int = 128
    protein_k_tokens: int = 128
    type: str = None
    

class OmicsDataset(Dataset):
    """Dataset for DNA/RNA data from Parquet, formatted for Bio-LLM."""
    def __init__(
        self,
        parquet_file: str,
        tokenizer,
        dataset_config,
        dna_rna_tokenizer=None,
        protein_tokenizer=None,
        read_nums=None,
        shuffle=False,
        seed=42,
        num_workers=0,
        type=None,
        **kwargs
    ):
        """
        Initialize the dataset by loading a Parquet file and formatting each example.

        Args:
            parquet_file: Path to the Parquet file.
            tokenizer: Text tokenizer for processing text parts.
            dataset_config: Configuration for the dataset.
            dna_rna_tokenizer: Tokenizer for DNA/RNA sequences.
            read_nums: Maximum number of samples to read.
            shuffle: Whether to shuffle the dataset.
            seed: Random seed for shuffling.
            num_workers: Number of workers for data loading.
            type: Dataset type. "Train / Eval" or "Test"
            **kwargs: Additional arguments.
        """

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.parquet_file = parquet_file
        self.tokenizer = tokenizer
        self.dna_rna_tokenizer = dna_rna_tokenizer
        self.protein_tokenizer = protein_tokenizer
        self.dataset_config = dataset_config
        self.shuffle = shuffle
        self.seed = seed
        self.num_workers = num_workers

        # Configuration parameters
        self.max_len = dataset_config.max_len
        self.max_src_len = dataset_config.max_src_len
        self.dna_rna_project_token_num = dataset_config.dna_rna_k_tokens
        self.protein_project_token_num = dataset_config.protein_k_tokens
        self.mode = dataset_config.mode
        self.cal_metric_pos = dataset_config.cal_metric_pos
        self.padding = dataset_config.padding
        self.dataset_type = type

        # Special tokens
        self._pretokenize_special_tokens()

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
        n_samples = len(df)
        self.data = [None] * n_samples
        # self.data = []

        with Pool(self.num_workers) as pool:
            results = pool.map(
                partial(self._preprocess_sample, tokenizer=self.tokenizer),
                df.to_dict('records'),
                # chunksize=2500
                chunksize=min(1000, max(1, len(df) // (self.num_workers * 10)))
            )
            with tqdm(total=len(df), desc="Preprocessing", unit="sample") as pbar:
                for idx, result in enumerate(results):
                    self.data[idx] = result
                    # self.data.append(result)
                    pbar.update(1)
        assert all(item is not None for item in self.data), "存在未填充的位置！"
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
        assert len(processed['omic_ids']) == len(processed['omic_info_list']), \
            f"Mismatch in Omic IDs and Omic info for sample {idx}: {len(processed['omic_ids'])} vs {len(processed['omic_info_list'])}"
        return processed

    def _pretokenize_special_tokens(self):
        self.dna_start_token = "<|dna_start|>"
        self.dna_end_token = "<|dna_end|>"
        self.dna_pad_token = "<|dna_pad|>"
        self.rna_start_token = "<|rna_start|>"
        self.rna_end_token = "<|rna_end|>"
        self.rna_pad_token = "<|rna_pad|>"
        self.protein_start_token = "<|protein_start|>"
        self.protein_end_token = "<|protein_end|>"
        self.protein_pad_token = "<|protein_pad|>"
        self.dna_start_id = self.tokenizer.convert_tokens_to_ids(self.dna_start_token)
        self.dna_end_id = self.tokenizer.convert_tokens_to_ids(self.dna_end_token)
        self.dna_pad_id = self.tokenizer.convert_tokens_to_ids(self.dna_pad_token)
        self.rna_start_id = self.tokenizer.convert_tokens_to_ids(self.rna_start_token)
        self.rna_end_id = self.tokenizer.convert_tokens_to_ids(self.rna_end_token)
        self.rna_pad_id = self.tokenizer.convert_tokens_to_ids(self.rna_pad_token)
        self.protein_start_id = self.tokenizer.convert_tokens_to_ids(self.protein_start_token)
        self.protein_end_id = self.tokenizer.convert_tokens_to_ids(self.protein_end_token)
        self.protein_pad_id = self.tokenizer.convert_tokens_to_ids(self.protein_pad_token)

        self.eos_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id

    def _preprocess_sample(self, sample: dict, tokenizer) -> dict:
        """
        Format a Parquet example into DNA-LLM format suitable for processing.
        
        The Parquet structure already has:
        - "input": Clean text (tags removed)
        - "kind": Hyphen-separated types like "dna" or "dna-rna"
        """

        kinds_string = sample.get("kind", "").lower()
        kinds = kinds_string.split("-") if kinds_string else []
        kinds = list(set(kinds))
        
        # 预处理文本内容
        input_text = sample.get("input", "").strip()
        output_text = sample.get("output", "").strip()
        reasoning = sample.get("think", "").strip()

        # 提取 DNA/RNA 序列，并记录位置
        pattern_map = {
            "dna": r"<dna>\s*([ACGTNacgtn]+)\s*<dna>",
            "rna": r"<rna>\s*([ACGTNacgtn]+)\s*<rna>",
            "protein": r"<protein>\s*([ACDEFGHIKLMNPQRSTVWYBXZOU]+)\s*<protein>"
        }

        # Determine which special tokens to use based on sequence type
        tag_map = {
            "dna": {
                "start": self.dna_start_id,
                "pad": self.dna_pad_id,
                "end": self.dna_end_id,
            },
            "rna": {
                "start": self.rna_start_id,
                "pad": self.rna_pad_id,
                "end": self.rna_end_id,
            },
            "protein": {
                "start": self.protein_start_id,
                "pad": self.protein_pad_id,
                "end": self.protein_end_id
            }
        }

        seq_info: List[Dict[str, any]] = []
        raw_seqs: List[str] = []

        for kind in kinds:
            pat = pattern_map.get(kind)
            if not pat:
                continue
            for m in re.finditer(pat, input_text, flags=re.IGNORECASE):
                raw_seq = m.group(1).upper()
                seq_info.append({"type": kind, "start": m.start(), "end": m.end()})
                raw_seqs.append(raw_seq)
        

        input_ids = list(self.system_prompt_ids)
        omic_info_list = []

        start = 0
        # encode 非序列部分，并记录序列起始位置
        for info in sorted(seq_info, key=lambda x: x["start"], reverse=False):
            seq_type = info["type"]
            s, e = info["start"], info["end"]

            input_ids.extend(
                tokenizer.encode(input_text[start:s], add_special_tokens=False)
            )
            omic_info_list.append({
                "type": seq_type,
                "start": len(input_ids)
            })

            input_ids.append(tag_map[seq_type]["start"])
            if seq_type in ['dna', 'rna']:
                input_ids.extend([tag_map[seq_type]["pad"]] * self.dna_rna_project_token_num)
            else:
                input_ids.extend([tag_map[seq_type]["pad"]] * self.protein_project_token_num)
            input_ids.append(tag_map[seq_type]["end"])

            start = e

        # 添加剩余文本
        if start < len(input_text):
            input_ids.extend(
                tokenizer.encode(input_text[start:], add_special_tokens=False)
            )
        
        # Encode the sequence
        output_ids = tokenizer.encode(output_text, add_special_tokens=False) if output_text else []
        reasoning_ids = tokenizer.encode(reasoning, add_special_tokens=False) if reasoning else []

        
        # 处理序列数据
        omic_ids_list = []

        for i, seq in enumerate(raw_seqs):
            seq_type = seq_info[i]["type"]
            encoded_seq = self._encode_sequence(seq, seq_type)
            omic_ids_list.append(encoded_seq)

        if self.dataset_type == "Test":
            return {
                "input_ids": input_ids,
                "output_ids": output_ids,
                "reasoning_token_ids": reasoning_ids,
                "omic_ids_list": omic_ids_list,
                "omic_info_list": omic_info_list,
                "task": sample.get("task", ""),
                "kind": kinds_string,
                "label": sample.get("label", ""),
                "raw_input": input_text,
                "raw_output": output_text,
            }
        return {
            "input_ids": input_ids,
            "output_ids": output_ids,
            "reasoning_token_ids": reasoning_ids,
            "omic_ids_list": omic_ids_list,
            "omic_info_list": omic_info_list,
            "task": sample.get("task", ""),
            "kind": kinds_string,
            "label": sample.get("label", "")
        }

    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Process a sample into model-ready format with tokenized sequences.
        """
        input_ids = sample["input_ids"]

        # 添加助手起始标记
        input_ids.extend(self.assistant_start_ids)

        # Process output based on mode
        if self.mode == 'sft':
            output_ids = sample["output_ids"]
        else:
            output_ids = []

        # Add EOS token based on mode
        if self.dataset_type != "Test":
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
                # print(f"Truncating input_ids from {len(input_ids)} to {self.max_len-1}")
                input_ids = input_ids[:self.max_len-1] + [self.eos_id]
                labels = labels[:self.max_len-1] + [self.eos_id]

            # Calculate metric position
            cal_metric_pos = None
            if self.cal_metric_pos is not None:
                cal_metric_pos = input_len + 1 + self.cal_metric_pos
            elif len(output_ids) > 0:
                cal_metric_pos = input_len + 1
        else:
            input_len = len(input_ids)
        
        
        attention_mask = [1] * len(input_ids)
        if self.dataset_type != "Test":
            omic_start_pos_list = sample["omic_info_list"]

            if self.padding and (pad_len := self.max_len - len(input_ids)) > 0:
                input_ids[:0] = [self.pad_id] * pad_len
                labels[:0] = [-100] * pad_len
                attention_mask[:0] = [0] * pad_len
                for i in range(len(omic_start_pos_list)):
                    omic_start_pos_list[i]["start"] += pad_len
            # Convert to tensors
            return {
                "input_ids": torch.LongTensor(input_ids),
                "omic_ids": torch.stack(sample["omic_ids_list"]),
                "omic_info_list": omic_start_pos_list,
                "labels": torch.LongTensor(labels),
                "attention_mask": torch.LongTensor(attention_mask),
                "task": sample["task"],
                "kind": sample["kind"],
                "raw_label": sample["label"],
                "raw_input": sample["raw_input"],
                "raw_output": sample["raw_output"],
                }
        else:
            # Add padding if needed
            if self.padding and (pad_len := self.max_len - len(input_ids)) > 0:
                input_ids.extend([self.pad_id] * pad_len)
                labels.extend([-100] * pad_len)
                attention_mask.extend([0] * pad_len)
            
            return {
                "input_ids": torch.LongTensor(input_ids),
                "omic_ids": torch.stack(sample["omic_ids_list"]),
                "omic_info_list": sample["omic_info_list"],
                "labels": torch.LongTensor(labels),
                "attention_mask": torch.LongTensor(attention_mask),
                "cal_metric_pos": cal_metric_pos,
            }
    
    def _encode_sequence(self, seq: str, seq_type: str) -> torch.LongTensor:
        """
        Tokenize a DNA/RNA sequence and pad/truncate to fixed length.
        """
        if not self.dna_rna_tokenizer:
            raise ValueError("DNA/RNA tokenizer is required but not provided")
        if not self.protein_tokenizer:
            raise ValueError("Protein tokenizer is required but not provided")

        if seq_type.lower() in ['dna', 'rna']:        
            encoding = self.dna_rna_tokenizer(
                seq, 
                padding='max_length',
                max_length= self.dna_rna_project_token_num,
                truncation=True,
                return_tensors='pt'
            )
        else:
            encoding = self.protein_tokenizer(
                seq, 
                padding='max_length',
                max_length= self.protein_project_token_num,
                truncation=True,
                return_tensors='pt'
            )
        return encoding['input_ids'].squeeze(0)
    

def qwen_omics_collate_fn(batch):
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
    omic_info_lists = [sample.get("omic_info_list", []) for sample in batch]
    omic_counts = [len(omic_info_list) for omic_info_list in omic_info_lists]
    omic_ids = [sample.get("omic_ids", None) for sample in batch]
      
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )
    omic_ids = torch.nn.utils.rnn.pad_sequence(
        omic_ids, batch_first=True, padding_value=1
    ) if omic_ids else None

    # Pad omic_info_lists to the same length as omic_ids
    for i in range(len(omic_info_lists)):
        if len(omic_info_lists[i]) < omic_ids.shape[1]:
            omic_info_lists[i].extend([{"type": "pad", "start": -1}] * (omic_ids.shape[1] - len(omic_info_lists[i])))
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "omic_ids": omic_ids,
        "omic_info_list": omic_info_lists,
        "omic_counts": omic_counts,
        "cal_metric_pos": cal_metric_pos,
    }


def qwen_omics_collate_fn_inference(batch):
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
    omic_info_lists = [sample.get("omic_info_list", []) for sample in batch]
    omic_counts = [len(omic_info_list) for omic_info_list in omic_info_lists]
    omic_ids = [sample.get("omic_ids", None) for sample in batch]

    
    raw_input = [sample.get("raw_input") for sample in batch]
    raw_output = [sample.get("raw_output") for sample in batch]
    raw_label = [sample.get("raw_label") for sample in batch]
    raw_task = [sample.get("task") for sample in batch]
    raw_kind = [sample.get("kind") for sample in batch]

        
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )
    omic_ids = torch.nn.utils.rnn.pad_sequence(
        omic_ids, batch_first=True, padding_value=1
    ) if omic_ids else None

    # Pad omic_info_lists to the same length as omic_ids
    for i in range(len(omic_info_lists)):
        if len(omic_info_lists[i]) < omic_ids.shape[1]:
            omic_info_lists[i].extend([{"type": "pad", "start": -1}] * (omic_ids.shape[1] - len(omic_info_lists[i])))
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "omic_ids": omic_ids,
        "omic_start_pos_list": omic_info_lists,
        "omic_counts": omic_counts,
        "cal_metric_pos": cal_metric_pos,
        "input": raw_input,
        "raw_output": raw_output,
        "raw_label": raw_label,
        "raw_task": raw_task,
        "raw_kind": raw_kind,
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
    pass