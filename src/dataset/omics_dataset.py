import os
import re
import pdb
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Callable
from collections import defaultdict

import numpy as np
import bisect
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor
from utils.tools import is_main_process
from utils import time_count

@dataclass
class DatasetConfig:
    # length of text_omics_ids + label_ids + ml_format 
    max_len: int = 8192
    mode: str = "sft"
    padding: bool = True
    input_field: str = "input"
    output_field: str = "output"
    dna_rna_k_tokens: int = 128
    protein_k_tokens: int = 128
    type: str = ''

class OmicsDataset(Dataset):
    """Dataset for DNA/RNA data from Parquet, formatted for Bio-LLM."""

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
        self.dna_start_id = self.tokenizer.convert_tokens_to_ids(
            self.dna_start_token)
        self.dna_end_id = self.tokenizer.convert_tokens_to_ids(
            self.dna_end_token)
        self.dna_pad_id = self.tokenizer.convert_tokens_to_ids(
            self.dna_pad_token)
        self.rna_start_id = self.tokenizer.convert_tokens_to_ids(
            self.rna_start_token)
        self.rna_end_id = self.tokenizer.convert_tokens_to_ids(
            self.rna_end_token)
        self.rna_pad_id = self.tokenizer.convert_tokens_to_ids(
            self.rna_pad_token)
        self.protein_start_id = self.tokenizer.convert_tokens_to_ids(
            self.protein_start_token)
        self.protein_end_id = self.tokenizer.convert_tokens_to_ids(
            self.protein_end_token)
        self.protein_pad_id = self.tokenizer.convert_tokens_to_ids(
            self.protein_pad_token)

        self.eos_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id

        self._regex_map = {
            "dna":
            re.compile(r"<dna>\s*([ACGTNacgtn]+)\s*</dna>"),
            "rna":
            re.compile(r"<rna>\s*([ACGTNacgtn]+)\s*</rna>"),
            "protein":
            re.compile(
                r"<protein>\s*([ACDEFGHIKLMNPQRSTVWYBXZOU]+)\s*</protein>"),
        }

    def convert_source_to_id(self, source:str):
        if 'antibody_antigen' in source:
            return 0
        elif 'cpd-prom_core' in source:
            return 1
        elif 'CRISPROnTarget' in source:
            return 2
        elif 'emp-H' in source:
            return 3
        elif 'enhancer_activity' in source:
            return 4
        elif 'Fluorescence-Fluorescence' in source:
            return 5
        elif 'FunctionEC-FunctionEC' in source:
            return 6
        elif 'Isoform-Isoform' in source:
            return 7
        elif 'MeanRibosomeLoading-MeanRibosomeLoading' in source:
            return 8
        elif 'Modification-Modification' in source:
            return 9
        elif 'NoncodingRNAFamily-NoncodingRNAFamily' in source:
            return 10
        elif 'pd-prom_300' in source:
            return 11
        elif 'ProgrammableRNASwitches-ProgrammableRNASwitches' in source:
            return 12
        elif 'promoter_enhancer_interaction' in source:
            return 13
        elif 'rna_protein_interaction' in source:
            return 14
        elif 'Solubility-Solubility' in source:
            return 15
        elif 'Stability-Stability' in source:
            return 16
        elif 'Thermostability-Thermostability' in source:
            return 17
        elif 'tf-h' in source:
            return 18
        elif 'tf-m' in source:
            return 19
        else:
            return 100

    def greedy_knapsack(self, numbers: List[int], capacity: int) -> List[List[int]]:
        r"""Implement efficient greedy algorithm with binary search for the knapsack problem."""
        numbers.sort()  # sort numbers in ascending order for binary search
        knapsacks = []

        def search_for_fit(numbers: list[int], capacity: int) -> int:
            r"""Find the index of largest number that fits into the knapsack with the given capacity."""
            index = bisect.bisect(numbers, capacity)
            return -1 if index == 0 else (index - 1)

        while numbers:
            current_knapsack = []
            remaining_capacity = capacity

            while True:
                index = search_for_fit(numbers, remaining_capacity)
                if index == -1:
                    break  # no more numbers fit in this knapsack

                remaining_capacity -= numbers[index]  # update the remaining capacity
                current_knapsack.append(numbers.pop(index))  # add the number to knapsack

            knapsacks.append(current_knapsack)

        return knapsacks

    def __init__(
        self,
        parquet_file: str,
        tokenizer,
        dataset_config:"DatasetConfig",
        dna_rna_tokenizer=None,
        protein_tokenizer=None,
        read_nums=None,
        type=None,
        packing=True,
        **kwargs,
    ):
        """
        Initialize the dataset by loading a Parquet file and formatting each example.

        Args:
            parquet_file: Path to the Parquet file.
            tokenizer: Text tokenizer for processing text parts.
            dataset_config: Configuration for the dataset.
            dna_rna_tokenizer: Tokenizer for DNA/RNA sequences.
            read_nums: Maximum number of samples to read.
            type: Dataset type. "Train / Eval" or "Test"
            **kwargs: Additional arguments.
        """

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.parquet_file = parquet_file
        self.tokenizer = tokenizer
        self.dna_rna_tokenizer = dna_rna_tokenizer
        self.protein_tokenizer = protein_tokenizer
        self.dataset_config = dataset_config

        # Configuration parameters
        self.max_len = dataset_config.max_len
        self.dna_rna_project_token_num = dataset_config.dna_rna_k_tokens
        self.protein_project_token_num = dataset_config.protein_k_tokens
        self.mode = dataset_config.mode
        self.padding = dataset_config.padding
        self.dataset_type = type
        self.packing = packing

        if "test" in self.dataset_type.lower() and self.packing:
            raise Exception('Packing not support test dataset, please disable.')

        # Special tokens
        self._pretokenize_special_tokens()

        # 预定义固定内容的分词结果
        self.system_user_ids = self.tokenizer.encode(
            "<|im_start|>system\nYou are a helpful knowledgeable and precise biomedical assistant.<|im_end|>\n<|im_start|>user\n",
            add_special_tokens=False,
        )
        self.assistant_start_ids = self.tokenizer.encode(
            "<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)

        # Load data
        print(f"Loading parquet data from {parquet_file}")
        df = pd.read_parquet(parquet_file, engine='pyarrow', use_threads=True, memory_map=True)

        # Limit samples if specified
        if read_nums:
            df = df.head(read_nums)

        if self.packing:
            self.batch_input_indices = self.pack_input_ids(df=df, max_token_length=self.max_len)
            self.dataset_length = len(self.batch_input_indices)
        else:
            self.batch_input_indices = None
            self.dataset_length = len(df)
        
        self.df = df

    def pack_input_ids(self, df: pd.core.frame.DataFrame, max_token_length: int) -> List[List[int]]:
        """Return a list of packed index"""

        # 如果有缓存，直接加载
        cache_file = '.cache/indices.pt'
        # if not is_main_process() and os.path.exists(cache_file):
        if os.path.exists(cache_file):
            print(f'Load cache data from {cache_file}')
            return torch.load(cache_file)

        batch_input_indices = []
        chunk = 1000

        for start in tqdm(range(0, len(df), chunk), disable=not is_main_process()):
            end = start + min(chunk, len(df) - start)

            # 用于根据 length 查询合适的 id
            length2ids: Dict[str, List[int]] = defaultdict(list)
            lengths = []
            for i in range(start, end):
                sample = self.format_raw(sample=df.loc[i], text_tokenizer=self.tokenizer, encode_sequence_fn=lambda _s, _i: [])
                input_length = len(sample['input_ids'])
                lengths.append(input_length)
                length2ids[input_length].append(i)

            # sort by descend
            knapsacks = self.greedy_knapsack(lengths, max_token_length-1)

            chunk_indexs = []
            for knapsack in knapsacks:
                indexes = [length2ids[length].pop() for length in knapsack]
                batch_input_indices.append(indexes)

        # caching
        if is_main_process():
            print(f'Dump cache data to {cache_file}')
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            torch.save(batch_input_indices, cache_file)
        return batch_input_indices

    def __len__(self) -> int:
        """Return the number of items in the (packed) dataset."""
        return self.dataset_length

    def __getitem__(self, idx: int) -> List[Dict[str, List[int]]]:
        """Return a specific item from the (packed) dataset."""
        # Ensure index is valid
        if idx < 0 or idx >= self.dataset_length:
            raise IndexError(
                f"Index {idx} out of bounds for dataset with {self.dataset_length} items"
            )

        if self.packing:
            samples = []
            for index in self.batch_input_indices[idx]:
                sample = self.format_raw(sample=self.df.loc[index], text_tokenizer=self.tokenizer, encode_sequence_fn=self._encode_sequence)
                samples.append(sample)
            return samples

        sample = self.format_raw(sample=self.df.loc[idx], text_tokenizer=self.tokenizer, encode_sequence_fn=self._encode_sequence)
        return [sample]


    def format_raw(self, sample: pd.core.series.Series, text_tokenizer, encode_sequence_fn: Callable[[str, List[int]], List[int]]) -> Dict[str, Any]:
        """
        Format a Parquet example into DNA-LLM format suitable for processing.

        The Parquet structure already has:
        - "input": Clean text (tags removed)
        - "kind": Hyphen-separated types like "dna" or "dna-rna"
        """
        # 预处理文本内容
        input_text = sample.get("input", "").strip()
        output_text = sample.get("output", "").strip()
        reasoning = sample.get("think", "").strip()

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
                "end": self.protein_end_id,
            },
        }

        seq_info: List[Dict[str, Any]] = []

        for kind in ["dna", "rna", "protein"]:
            pat = self._regex_map.get(kind)
            if pat is None:
                continue
            for m in pat.finditer(input_text):
                raw_seq = m.group(1).upper()
                seq_info.append({
                    "type": kind,
                    "start": m.start(),
                    "end": m.end(),
                    "seq": raw_seq, 
                })

        input_ids = list(self.system_user_ids)
        omic_info_list = []
        
        start = 0
        # encode 非序列部分，并记录序列起始位置
        for info in sorted(seq_info, key=lambda x: x["start"], reverse=False):
            seq_type = info["type"]
            s, e = info["start"], info["end"]
            # 处理序列前的文本
            input_ids.extend(
                text_tokenizer.encode(input_text[start:s],
                                 add_special_tokens=False))

            # encode 序列本身，但不填入 text input_ids
            seq = info["seq"]
            encoded_seq = encode_sequence_fn(seq, seq_type)
            omic_info_list.append({"type": seq_type, "start": len(input_ids), "id": encoded_seq})

            # 序列所在位置留个 placeholder
            input_ids.append(tag_map[seq_type]["start"])
            if seq_type in ["dna", "rna"]:
                input_ids.extend([tag_map[seq_type]["pad"]] *
                                 self.dna_rna_project_token_num)
            else:
                input_ids.extend([tag_map[seq_type]["pad"]] *
                                 self.protein_project_token_num)
            input_ids.append(tag_map[seq_type]["end"])

            start = e

        # 添加末尾剩余文本
        if start < len(input_text):
            input_ids.extend(
                text_tokenizer.encode(input_text[start:], add_special_tokens=False))
        
        # 添加 assistant 部分标签
        input_ids.extend(self.assistant_start_ids)

        # Encode the sequence
        output_ids = (text_tokenizer.encode(output_text, add_special_tokens=False)
                      if output_text else [])
        reasoning_ids = (text_tokenizer.encode(reasoning, add_special_tokens=False)
                         if reasoning else [])

        # 处理 output 数据
        labels = None

        # Add EOS token based on mode
        if self.dataset_type != "Test":
            output_ids.append(self.eos_id)

            input_len = len(input_ids)
            input_ids.extend(output_ids)

            # Create labels
            # Use -100 to ignore input tokens in loss calculation
            labels = ([-100] * input_len +
                      output_ids if self.mode == "sft" else input_ids.copy())

            # Truncate if necessary
            if len(input_ids) > self.max_len:
                # print(f"Truncating input_ids from {len(input_ids)} to {self.max_len-1}")
                input_ids = input_ids[:self.max_len - 1] + [self.eos_id]
                labels = labels[:self.max_len - 1] + [self.eos_id]
        else:
            input_len = len(input_ids)

        attention_mask = [1] * len(input_ids)
        task_name = sample.get("task", "")
        task_num = sample.get("task_num", -1)

        if self.dataset_type == "Test":
            omic_start_pos_list = omic_info_list

            if self.padding and (pad_len := self.max_len - len(input_ids)) > 0:
                # padding left
                input_ids[:0] = [self.pad_id] * pad_len
                attention_mask[:0] = [0] * pad_len
                for i, _ in enumerate(omic_start_pos_list):
                    omic_start_pos_list[i]["start"] += pad_len
            # Convert to tensors
            return {
                "input_ids": torch.LongTensor(input_ids),
                "omic_info_list": omic_start_pos_list,
                "attention_mask": torch.LongTensor(attention_mask),
                "task": task_name,
                "raw_label": sample.get("label", ""),
                "raw_input": input_text,
                "raw_output": output_text,
                "task_label": self.convert_source_to_id(task_name),
                "task_num": task_num,
            }

        # return raw list
        return {
            "input_ids": input_ids,
            "omic_info_list": omic_info_list,
            "labels": labels,
            "attention_mask": attention_mask,
            "task_label": self.convert_source_to_id(task_name),
            "task_num": task_num,
        }

    def _encode_sequence(self, seq: str, seq_type: str) -> torch.LongTensor:
        """
        Tokenize a DNA/RNA sequence and pad/truncate to fixed length.
        """
        if not self.dna_rna_tokenizer:
            raise ValueError("DNA/RNA tokenizer is required but not provided")
        if not self.protein_tokenizer:
            raise ValueError("Protein tokenizer is required but not provided")

        if seq_type.lower() in ["dna", "rna"]:
            encoding = self.dna_rna_tokenizer(
                seq,
                padding="max_length",
                max_length=self.dna_rna_project_token_num,
                truncation=True,
                return_tensors="pt",
            )
        elif seq_type.lower() == "protein":
            encoding = self.protein_tokenizer(
                seq,
                padding="max_length",
                max_length=self.protein_project_token_num,
                truncation=True,
                return_tensors="pt",
            )
        else:
            raise ValueError(f"Unsupported sequence type: {seq_type}")
        return encoding["input_ids"].squeeze(0)


def qwen_omics_collate_fn(batch: List[Dict[str, Any]], max_token_length: int, pad_id: int, eos_id: int):
    """
    Collate function for DataLoader with multimodal DNA batches.
    Handles variable length DNA sequences and attention masks.

    Args:
        batch: List of (packed) samples from the dataset

    Returns:
        Batched tensors suitable for model input
    """

    def concat(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        pack_input_ids = []
        pack_labels = []
        pack_attention_mask = []
        pack_position_ids = []
        pack_omic_info_list = []
        # 任务类型 ID，用于分子 loss 统计
        pack_task_label = []
        # 训练数据 ID
        pack_task_num = []

        offset = 0
        for sample in samples:
            len_i = len(sample["input_ids"])
            pack_input_ids   += sample["input_ids"]
            pack_labels      += sample["labels"]
            pack_attention_mask += sample["attention_mask"]
            pack_position_ids += list(range(len_i))

            if not pack_task_label:
                pack_task_label.append(sample["task_label"])
                pack_task_num.append(sample["task_num"])

            # 调整 omic_info 的 start
            for info in sample.get("omic_info_list", []):
                pack_omic_info_list.append({"type": info["type"], "start": info["start"] + offset, "id": info["id"]})
            offset += len_i

        # 整体截断
        if len(pack_input_ids) > max_token_length:
            pack_input_ids   = pack_input_ids[:max_token_length-1] + [eos_id]
            pack_labels      = pack_labels[:max_token_length-1]   + [eos_id]
            pack_attention_mask = pack_attention_mask[:max_token_length]
            pack_position_ids   = pack_position_ids[:max_token_length]


        # pad 到 max_len（trainer 要求固定长度）
        pad_len = max_token_length - len(pack_input_ids)
        if pad_len > 0:
            pack_input_ids.extend([pad_id] * pad_len)
            pack_labels.extend([-100] * pad_len)
            pack_attention_mask.extend([0] * pad_len)
            pack_position_ids.extend([0] * pad_len)

        return {
            "input_ids": torch.LongTensor(pack_input_ids),
            "labels": torch.LongTensor(pack_labels),
            "position_ids": torch.LongTensor(pack_position_ids),
            "attention_mask": torch.LongTensor(pack_attention_mask),
            "omic_info_list": pack_omic_info_list,
            "task_label": torch.LongTensor(pack_task_label), 
            "task_num": torch.LongTensor(pack_task_num),
        }

    # 内部折叠，输入 List[List[Dict]]， 折叠内层 List，得到 List[Dict]
    concated = [concat(samples) for samples in batch]
    # 内外翻转，输入 List[Dict]， 翻转成 Dict[str,List]
    pivot: Dict[str, List[str]] = {}
    for d in concated:
        for k, v in d.items():
            pivot.setdefault(k, []).append(v)

    # assign
    input_ids = pivot["input_ids"]
    labels = pivot["labels"]
    attention_mask = pivot["attention_mask"]
    omic_info_list = pivot["omic_info_list"]
    position_ids = pivot["position_ids"]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids,
                                                batch_first=True,
                                                padding_value=0)

    position_ids = torch.nn.utils.rnn.pad_sequence(position_ids,
                                                batch_first=True,
                                                padding_value=0)

    labels = torch.nn.utils.rnn.pad_sequence(labels,
                                             batch_first=True,
                                             padding_value=-100)

    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask,
                                                     batch_first=True,
                                                     padding_value=0)

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "omic_info_list": omic_info_list,
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
    attention_mask = [sample["attention_mask"] for sample in batch]
    omic_info_lists = [sample.get("omic_info_list", []) for sample in batch]

    raw_input = [sample.get("raw_input") for sample in batch]
    raw_output = [sample.get("raw_output") for sample in batch]
    raw_label = [sample.get("raw_label") for sample in batch]
    raw_task = [sample.get("task") for sample in batch]
    raw_kind = [sample.get("kind") for sample in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids,
                                                batch_first=True,
                                                padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask,
                                                     batch_first=True,
                                                     padding_value=0)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "omic_info_list": omic_info_lists,
        "input": raw_input,
        "raw_output": raw_output,
        "raw_label": raw_label,
        "raw_task": raw_task,
        "raw_kind": raw_kind,
    }
