import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset
import json
import re
import numpy as np

class IterableMultimodalDNADataSet(IterableDataset):
    def __init__(
        self,
        data_path,
        tokenizer,
        dataset_config,
        multimodal_tokenizer=None,
        read_nums=None,
        shuffle=False,
        seed=42,
        start_step=0,
        **kwargs
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.dna_tokenizer = multimodal_tokenizer
        self.dataset_config = dataset_config
        self.project_token_num = kwargs.get("multimodal_k_tokens", 32)
        self.read_nums = read_nums
        self.shuffle = shuffle
        self.seed = seed
        self.start_step = start_step

        self.max_len = dataset_config.max_len
        self.max_src_len = dataset_config.max_src_len
        self.mode = dataset_config.mode
        self.meta_prompt = dataset_config.meta_prompt
        self.prefix = dataset_config.prefix
        self.postfix = dataset_config.postfix
        self.cal_metric_pos = dataset_config.cal_metric_pos
        self.padding = dataset_config.padding

        # 注意现在的meta_prompt, prefix, postfix都是None

        # Special tokens for DNA sequences
        self.dna_start_token = "<|dna_start|>"
        self.dna_end_token = "<|dna_end|>"
        self.dna_pad_token = "<|dna_pad|>"

        # Load data
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.lines = f.readlines()

        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        self.line_count = len(self.lines)
        if self.read_nums:
            self.lines = self.lines[:self.read_nums]

    @property
    def estimated_len(self):
        return len(self.lines) // self.world_size

    def _load_sample(self, idx, line):
        try:
            sample = json.loads(line.strip())
            # Validate required fields
            assert self.dataset_config.input_field in sample, f"Missing input field: {self.dataset_config.input_field}"
            if self.mode == 'sft':
                assert self.dataset_config.output_field in sample, f"Missing output field: {self.dataset_config.output_field}"
            return sample
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse JSON at line {idx}")
            return None
        except AssertionError as e:
            print(f"Warning: {str(e)} at line {idx}")
            return None

    def _get_start_step(self, total_lines):
        return self.start_step % total_lines if total_lines > 0 else 0

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        with open(self.data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Shard data for distributed training
        lines = lines[self.rank::self.world_size]
        
        # Further shard data for multiple workers
        lines = lines[worker_id::num_workers]

        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.rank + worker_id)
            rng.shuffle(lines)

        step = self._get_start_step(len(lines))
        while step < len(lines):
            line = lines[step]
            step += 1
            sample = self._load_sample(step, line)
            if sample:
                try:
                    yield self.process_sample(sample)
                except Exception as e:
                    print(f"Warning: Failed to process sample at step {step}: {str(e)}")
                    continue

    def process_sample(self, sample):
        input_ids = []
        dna_ids_list = []
        dna_start_pos_list = []
        attention_mask = []
        pos = 0

        # DNA sequence pattern with optional whitespace
        pattern = r"<dna>\s*([ACGTacgt]+)\s*<dna>"

        input_text = sample[self.dataset_config.input_field]
        output_text = sample.get(self.dataset_config.output_field, "")

        # Process input text with DNA sequences
        self._process_text(input_text, input_ids, dna_ids_list, dna_start_pos_list, pos, pattern, True)

        # Process output based on mode
        if self.mode == 'sft':
            input_ids.extend(self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False))
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
        # Calculate metric position, 一般用于output的开始位置
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

    # 存在一个修改计划，需要将序列提前，并在原来的位置加一个DNA 1, DNA 2例如这种形式
    def _process_text(self, input_text, input_ids, dna_ids_list, dna_start_pos_list, pos, pattern, first_text_piece_tag):
        # Split text by DNA tags
        matches = list(re.finditer(pattern, input_text))
        extracted_dnas = [match.group(1) for match in matches]
        clean_text = re.sub(pattern, "", input_text)

        # 添加Qwen系统提示格式，替代BOS token
        input_ids.extend(self.tokenizer.encode("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n", add_special_tokens=False))

        for dna_seq in extracted_dnas:
            encoded_dna = self.dna_tokenizer(dna_seq, return_tensors='pt', padding='max_length', truncation=True, max_length=self.project_token_num)["input_ids"].squeeze(0)
            dna_ids_list.append(encoded_dna)
            # Truncate or pad DNA tokens
            dna_start_pos_list.append(len(input_ids))
            input_ids.extend([
                self.tokenizer.convert_tokens_to_ids(self.dna_start_token),
                *[self.tokenizer.convert_tokens_to_ids(self.dna_pad_token)] * self.project_token_num,
                self.tokenizer.convert_tokens_to_ids(self.dna_end_token)
            ])

        clean_text = clean_text.strip()
        if clean_text:
            input_ids.extend(self.tokenizer.encode(clean_text, add_special_tokens=False))
        input_ids.extend(self.tokenizer.encode("<|im_end|>\n", add_special_tokens=False))

class TransformersCompatibleDNADataset(torch.utils.data.Dataset):
    """
    A wrapper around IterableMultimodalDNADataSet to make it compatible with transformers Trainer.
    This loads all data into memory for random access.
    """
    def __init__(self, iterable_dataset, max_size=None):
        """
        Initialize the dataset.
        
        Args:
            iterable_dataset: The underlying iterable dataset
            max_size: Maximum number of samples to load
        """
        self.data = []
        self.max_size = max_size
        
        # Load all data into memory
        print(f"Loading data into memory for TransformersCompatibleDNADataset...")
        for i, sample in enumerate(iterable_dataset):
            self.data.append(sample)
            if max_size is not None and i >= max_size - 1:
                break
        
        print(f"Loaded {len(self.data)} samples into memory")
    
    def __len__(self):
        """
        Get the length of the dataset.
        
        Returns:
            int: The number of samples in the dataset
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: The index of the sample
            
        Returns:
            dict: The sample at the given index
        """
        return self.data[idx]
