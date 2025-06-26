import math
from typing import Optional, Union
from dataclasses import dataclass
import random

import numpy as np
import torch
import torch.distributed
from torch.utils.data import DataLoader, IterableDataset

from .dna_dataset import IterableMultimodalDNADataSet
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
        input_ids_list, labels_list, cal_metric_pos_list, dna_ids_lists, dna_start_pos_lists = [], [], [], [], []
        attention_masks_list = []
        for instance in examples:
            input_ids = torch.LongTensor(instance["input_ids"]) if isinstance(instance["input_ids"], list) else instance["input_ids"]
            labels = torch.LongTensor(instance["labels"]) if isinstance(instance["labels"], list) else instance["labels"]
            attention_masks = torch.LongTensor(instance["attention_masks"]) if isinstance(instance["labels"], list) else instance["labels"]
            cal_metric_pos = instance.get("cal_metric_pos", None)
            dna_ids_list = instance.get("dna_ids_list", None)
            dna_start_pos_list = instance.get("dna_start_pos_list", None)

            input_ids_list.append(input_ids) 
            labels_list.append(labels)
            attention_masks_list.append(attention_masks)
            cal_metric_pos_list.append(cal_metric_pos)
            dna_ids_lists.append(dna_ids_list)
            dna_start_pos_lists.append(dna_start_pos_list)

        if None in cal_metric_pos_list:
            cal_metric_pos_list = None
        if None in dna_ids_lists or None in dna_start_pos_lists:
            dna_ids_lists = None
            dna_start_pos_lists = None

        return {"input_ids": torch.stack(input_ids_list),
                "dna_ids_lists": dna_ids_lists,
                "labels": torch.stack(labels_list),
                "attention_mask": torch.stack(attention_masks_list),
                "cal_metric_pos_tensor": torch.tensor(cal_metric_pos_list) if cal_metric_pos_list is not None else None,
                "dna_start_pos_lists": dna_start_pos_lists}


def get_train_eval_args(args, is_train):
    return ('TRAIN' if is_train else 'EVAL',
            args.train_dataset_path if is_train else args.eval_dataset_path, 
            args.max_len if is_train else args.eval_max_len, 
            args.max_src_len if is_train else args.eval_max_src_len,
            args.read_nums if is_train else args.eval_read_nums,
            args.batch_size_per_gpu if is_train else args.eval_batch_size_per_gpu)


def load_dataloder(args, tokenizer, dp_rank, num_dp_ranks, dataset_kwargs, is_train):
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

    dataset = IterableMultimodalDNADataSet(data_path=dataset_path, read_nums=read_nums, **dataset_kwargs)
    
    is_iterable_dataset = isinstance(dataset, IterableDataset)
    dataset_sampler = None
    dataloader = DataLoader(dataset,
                            collate_fn=data_collator,
                            shuffle=False,
                            drop_last=True,
                            sampler=dataset_sampler,
                            batch_size=batch_size_per_gpu,
                            generator=torch.Generator(),
                            worker_init_fn=seed_worker,)
    
    msgs = []
    if not isinstance(dataset, IterableDataset):
        msgs.extend([
            f"{flag} DATALOADER LENGTH: {len(dataloader)}",
            f"{flag} DATASET LENGTH: {len(dataset)}",
        ])
    else:
        msgs.extend([
            f"{flag} IterableDataset does not support __len__, skipping length print",
        ])

    if is_train:
        assert args.epochs is not None or args.train_iters is not None, 'Must provide epochs or train_iters'
        
        if args.epochs is not None:
            dataset_len = getattr(dataset, 'estimated_len', None)
            assert dataset_len is not None and dataset_len > 0, \
                "IterableDataset must define `estimated_len` or set `read_nums`"

            micro_update_steps_one_epoch = math.ceil(dataset_len / batch_size_per_gpu)
            args.num_micro_update_steps = args.epochs * micro_update_steps_one_epoch
        else:
            args.num_micro_update_steps = args.train_iters

        args.num_global_update_steps = math.ceil(args.num_micro_update_steps / args.gradient_accumulation_steps)
        args.num_warmup_steps = int(args.num_global_update_steps * args.warmup) + 1

        msgs.extend([
            f"NUMBER OF MICRO UPDATE STEPS: {args.num_micro_update_steps}",
            f"NUMBER OF GLOBAL UPDATE STEPS: {args.num_global_update_steps}",
            f"NUMBER OF WARMUP STEPS: {args.num_warmup_steps}",
            f"Base learning rate is {args.lr}"
        ])

    for msg in msgs:
        print_rank_0(f"--->{msg}")

    return dataloader