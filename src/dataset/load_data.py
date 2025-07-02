import math
from typing import Optional, Union, Tuple
from dataclasses import dataclass
import random

import numpy as np
import torch
import torch.distributed
from torch.utils.data import DataLoader, IterableDataset

from .dna_dataset import IterableMultimodalDNADataSet, TransformersCompatibleDNADataset
from ..utils.tools import print_rank_0

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

@dataclass
class DatasetConfig:
    # More feasible if any new parameter should be added.
    max_len: int
    max_src_len: int
    meta_prompt: str =''
    input_field: str = 'input'
    output_field: str = 'output'
    mode: str = 'pretrain',
    prefix: str='Q:'
    postfix: str='A:'
    padding: bool = True
    apply_chat_template: bool = False
    cal_metric_pos: Optional[int] = None
    encode_single_gene: bool = False


# 多模态任务，需要自己实现一个 🌟
class DataCollator():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        if not examples:
            return {}

        # Initialize lists for batch
        batch = {
            "input_ids": [],
            "labels": [],
            "attention_mask": [],
            "dna_ids_lists": [],
            "dna_start_pos_lists": [],
            "cal_metric_pos": []
        }

        # Process each example
        for example in examples:
            # Handle required fields
            for key in ["input_ids", "labels", "attention_mask"]:
                value = example[key]
                if isinstance(value, list):
                    value = torch.LongTensor(value)
                batch[key].append(value)

            # Handle optional DNA-related fields
            if example.get("dna_ids_list") is not None and example.get("dna_start_pos_list") is not None:
                batch["dna_ids_lists"].append(example["dna_ids_list"])
                batch["dna_start_pos_lists"].append(example["dna_start_pos_list"])
            
            # Handle metric position
            if example.get("cal_metric_pos") is not None:
                batch["cal_metric_pos"].append(example["cal_metric_pos"])

        # Stack tensors
        batch["input_ids"] = torch.stack(batch["input_ids"])
        batch["labels"] = torch.stack(batch["labels"])
        batch["attention_mask"] = torch.stack(batch["attention_mask"])

        # Handle DNA sequences if present
        if batch["dna_ids_lists"] and batch["dna_start_pos_lists"]:
            # Pad DNA sequences to max length in batch
            max_dna_sequences = max(len(dna_list) for dna_list in batch["dna_ids_lists"])
            padded_dna_ids_lists = []
            padded_dna_start_pos_lists = []
            
            for dna_list, pos_list in zip(batch["dna_ids_lists"], batch["dna_start_pos_lists"]):
                if len(dna_list) < max_dna_sequences:
                    # Pad with empty DNA sequences
                    dna_list.extend([torch.zeros_like(dna_list[0])] * (max_dna_sequences - len(dna_list)))
                    # Make sure pos_list is a tensor and has the correct size
                    if isinstance(pos_list, list):
                        pos_list = torch.tensor(pos_list, dtype=torch.long)
                    # Pad positions with zeros
                    current_size = pos_list.size(0)
                    if current_size < max_dna_sequences:
                        padding = torch.zeros(max_dna_sequences - current_size, dtype=torch.long)
                        pos_list = torch.cat([pos_list, padding])
                
                padded_dna_ids_lists.append(torch.stack(dna_list))
                padded_dna_start_pos_lists.append(pos_list)
            
            # Ensure all tensors in padded_dna_start_pos_lists have the same size
            for i, pos_list in enumerate(padded_dna_start_pos_lists):
                if pos_list.size(0) != max_dna_sequences:
                    # Resize tensor to match max_dna_sequences
                    if pos_list.size(0) > max_dna_sequences:
                        padded_dna_start_pos_lists[i] = pos_list[:max_dna_sequences]
                    else:
                        padding = torch.zeros(max_dna_sequences - pos_list.size(0), dtype=torch.long)
                        padded_dna_start_pos_lists[i] = torch.cat([pos_list, padding])
            
            batch["dna_ids_lists"] = torch.stack(padded_dna_ids_lists)
            batch["dna_start_pos_lists"] = torch.stack(padded_dna_start_pos_lists)
        else:
            # Remove DNA fields if no DNA sequences present
            del batch["dna_ids_lists"]
            del batch["dna_start_pos_lists"]

        # Convert metric positions to tensor if present
        if batch["cal_metric_pos"]:
            batch["cal_metric_pos"] = torch.tensor(batch["cal_metric_pos"], dtype=torch.long)
        else:
            del batch["cal_metric_pos"]

        return batch


def get_train_eval_args(args, is_train):
    return ('TRAIN' if is_train else 'EVAL',
            args.train_dataset_path if is_train else args.eval_dataset_path, 
            args.max_len if is_train else args.eval_max_len, 
            args.max_src_len if is_train else args.eval_max_src_len,
            args.read_nums if is_train else args.eval_read_nums,
            args.batch_size_per_gpu if is_train else args.eval_batch_size_per_gpu)


def load_dataloder(args, tokenizer, dp_rank, num_dp_ranks, dataset_kwargs, is_train, return_transformers_dataset=True):
    flag, dataset_path, max_len, max_src_len, read_nums, batch_size_per_gpu = get_train_eval_args(args, is_train)
    if dataset_path is None:
        print_rank_0("The data set path is None!")
        return None
    data_collator = DataCollator(tokenizer)

    dataset_config = DatasetConfig(max_len=max_len,
                                   max_src_len=max_src_len,
                                   mode=args.mode, 
                                   meta_prompt=args.meta_prompt,
                                   prefix=args.prefix,
                                   postfix=args.postfix,
                                   padding=(args.batching_stretegy == 'padding'),
                                   apply_chat_template=False)

    dataset_kwargs = dict(dataset_config=dataset_config,
                          tokenizer=tokenizer,
                          global_rank=args.global_rank,
                          dp_rank=dp_rank,
                          num_dp_ranks=num_dp_ranks,
                          shuffle=True,
                          **dataset_kwargs)

    # Create the iterable dataset
    iterable_dataset = IterableMultimodalDNADataSet(data_path=dataset_path, read_nums=read_nums, **dataset_kwargs)
    
    # Ensure iterable_dataset has an estimated_len attribute for progress tracking
    if not hasattr(iterable_dataset, 'estimated_len') or iterable_dataset.estimated_len is None:
        if read_nums is not None:
            # If read_nums is provided, use it divided by world_size
            world_size = num_dp_ranks if num_dp_ranks > 0 else 1
            iterable_dataset.estimated_len = read_nums // world_size
            print_rank_0(f"Setting estimated dataset length to {iterable_dataset.estimated_len} for {flag}")
    
    # If we need a Transformers-compatible dataset, convert it
    if return_transformers_dataset:
        # For transformers trainer, we need to return a map-style dataset
        transformers_dataset = TransformersCompatibleDNADataset(
            iterable_dataset, 
            max_size=read_nums
        )
        return transformers_dataset
    
    # 如果直接使用IterableDataset，直接返回iterable_dataset
    return iterable_dataset