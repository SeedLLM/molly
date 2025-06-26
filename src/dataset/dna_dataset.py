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
            return json.loads(line.strip())
        except:
            return None

    def _get_start_step(self, total_lines):
        return self.start_step % total_lines
    

    def __iter__(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # 按 GPU 分片数据
        lines = lines[self.rank::self.world_size]

        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.rank)
            rng.shuffle(lines)

        step = self._get_start_step(len(lines))
        while step < len(lines):
            line = lines[step]
            step += 1
            sample = self._load_sample(step, line)
            if sample:
                yield self.process_sample(sample)

    def process_sample(self, sample):
        input_ids = []
        dna_ids_list = []
        dna_start_pos_list = []
        pos = 0
        pattern = r"<dna>([ACGTacgt]+)<dna>"

        input_text, output_text = sample[self.dataset_config.input_field], sample.get(self.dataset_config.output_field, "")
        self._process_text(input_text, input_ids, dna_ids_list, dna_start_pos_list, pos, pattern, True)

        if self.mode == 'sft':
            output_ids = self.tokenizer.encode(output_text)
        else:
            output_ids = []
            if output_text:
                input_ids += self.tokenizer.encode(output_text)

        if self.mode == 'pretrain':
            input_ids.append(self.tokenizer.eos_token_id)
        else:
            output_ids.append(self.tokenizer.eos_token_id)

        if len(input_ids) > self.max_src_len:
            input_ids = input_ids[:self.max_src_len]
        if len(output_ids) > (self.max_len - len(input_ids)):
            output_ids = output_ids[:(self.max_len - len(input_ids))]

        input_len = len(input_ids)
        input_ids += output_ids

        cal_metric_pos = (input_len + 1 + self.cal_metric_pos) if self.cal_metric_pos is not None else (input_len + 1 if len(output_ids) == 3 else None)

        if self.mode == 'sft':
            labels = [self.tokenizer.pad_token_id] * input_len + output_ids
        else:
            labels = input_ids.copy()

        attention_masks = [1] * len(input_ids)

        if self.padding:
            pad_len = self.max_len - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            labels += [self.tokenizer.pad_token_id] * pad_len
            attention_masks += [0] * pad_len

        return {
            "input_ids": torch.LongTensor(input_ids),
            "dna_ids_list": [torch.LongTensor(dna_ids) for dna_ids in dna_ids_list],
            "dna_start_pos_list": torch.LongTensor(dna_start_pos_list),
            "labels": torch.LongTensor(labels),
            "attention_masks": torch.LongTensor(attention_masks),
            "cal_metric_pos": cal_metric_pos,
        }

    def _process_text(self, input_text, input_ids, dna_ids_list, dna_pos_list, pos, pattern, first_text_piece_tag):
        dna_emb_tokens = []
        matches = list(re.finditer(pattern, input_text))
        extracted_dnas = [match.group(1) for match in matches]
        clean_text = re.sub(pattern, "", input_text)

        for dna_seq in extracted_dnas:
            encoded_dna = self.dna_tokenizer(dna_seq)["input_ids"]
            if len(encoded_dna) >= self.project_token_num:
                encoded_dna = encoded_dna[:self.project_token_num]
            else:
                pad_len = self.project_token_num - len(encoded_dna)
                encoded_dna += [self.dna_tokenizer.pad_token_id] * pad_len
            dna_ids_list.append(encoded_dna)
            dna_pos_list.append(len(dna_emb_tokens))

            placeholder = [
                self.tokenizer.convert_tokens_to_ids("<|dna_start|>"),
                *[self.tokenizer.convert_tokens_to_ids("<|dna_pad|>")] * self.project_token_num,
                self.tokenizer.convert_tokens_to_ids("<|dna_end|>")
            ]
            dna_emb_tokens += placeholder

        input_ids.extend(dna_emb_tokens)
        clean_text = clean_text.strip()
        if clean_text:
            word_ids = [self.tokenizer.bos_token_id] if first_text_piece_tag and self.tokenizer.bos_token_id else []
            # print(self.meta_prompt, self.prefix, self.tokenizer.encode(clean_text))
            # word_ids += self.meta_prompt + self.prefix + self.tokenizer.encode(clean_text)
            # 修改 🌟
            word_ids += self.tokenizer.encode(clean_text)
            input_ids += word_ids
        # input_ids += self.postfix
