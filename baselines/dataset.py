
import re
from typing import Dict, List
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

class ClassificationDataset(Dataset):
    def __init__(self,
                 file_path: str,
                 dna_rna_tokenizer: AutoTokenizer = None,
                 protein_tokenizer: AutoTokenizer = None,
                 model_type: str = "NT",
                 dna_rna_k_tokens: int = 1024,
                 protein_k_tokens: int = 1024,
                 label2id: Dict[str, int] = None,
                 multi_label: bool = False,
                 shuffle: bool = False):
        self.file_path = file_path
        self.dna_rna_tokenizer = dna_rna_tokenizer
        self.protein_tokenizer = protein_tokenizer
        self.model_type = model_type
        self.dna_rna_k_tokens = dna_rna_k_tokens
        self.protein_k_tokens = protein_k_tokens

        self.label2id = label2id
        self.multi_label = multi_label

        self.data = pd.read_parquet(file_path)
        if shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)

        self._regex_map = {
            "dna":     re.compile(r"<dna>\s*([ACGTNacgtn]+)\s*<dna>"),
            "rna":     re.compile(r"<rna>\s*([ACGTNacgtn]+)\s*<rna>"),
            "protein": re.compile(r"<protein>\s*([ACDEFGHIKLMNPQRSTVWYBXZOU]+)\s*<protein>")
        }

    
    def __len__(self):
        return len(self.data)


    def _extract_seqs(self, text: str) -> Dict[str, List[str]]:
        """
        Returns:
            {
                "dna":   ["ACGT...", ...],
                "rna":   ["ACGU...", ...],
                "protein": ["MKTLL...", ...]
            }    
        """
        seqs = {"dna": [], "rna": [], "protein": []}
        for kind, pat in self._regex_map.items():
            for m in pat.finditer(text):
                raw_seq = m.group(1).upper()
                seqs[kind].append(raw_seq)
        return seqs

    
    def _tokenize_single(self, seq: str, tokenizer: AutoTokenizer, max_len: int) -> Dict:
        return tokenizer(
            seq,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )


    def __getitem__(self, idx: int) -> Dict:
        row = self.data.iloc[idx].to_dict()
        input_text = row.get("input",  "").strip()
        label = row.get("label",  "").strip()

        # 解析序列
        seqs = self._extract_seqs(input_text)

        if self.multi_label:
            if "EC" in label:
                label = label.replace("EC", "")
            labels = label.split(",")
            label_tensor = torch.zeros(len(self.label2id), dtype=torch.float)
            for lab in labels:
                lab = lab.strip()
                if lab in self.label2id:
                    idx = self.label2id[lab]
                    label_tensor[idx] = 1.0
            if label_tensor.sum() == 0:
                raise ValueError(f"None of the labels in {labels} are in label2id mapping.")
        else:
            if isinstance(label, (int, float)):
                label_tensor = torch.tensor(label, dtype=torch.long if isinstance(label, int) else torch.float)
            elif label in ['positive', 'negative']:
                label_tensor = torch.tensor(1 if label == 'positive' else 0, dtype=torch.long)
            else:
                label_tensor = torch.tensor(float(label), dtype=torch.float)

        batch = {
            "labels": label_tensor
        }

        # 根据model_type确认 tokenize 策略
        if self.model_type == "NT":
            assert self.dna_rna_tokenizer is not None, "dna_rna_tokenizer is required for NT model"
            assert len(seqs["dna"]) + len(seqs["rna"]) == 1, "NT model requires exactly one DNA or RNA sequence"
            seq = seqs["dna"][0] if len(seqs["dna"]) == 1 else seqs["rna"][0]
            tokenized = self._tokenize_single(seq, self.dna_rna_tokenizer, self.dna_rna_k_tokens)

            batch.update({
                "x1": tokenized["input_ids"].squeeze(0),
                "mask1": tokenized["attention_mask"].squeeze(0)
            })
        elif self.model_type == "ESM":
            assert self.protein_tokenizer is not None, "protein_tokenizer is required for ESM model"
            assert len(seqs["protein"]) == 1, "ESM model requires exactly one Protein sequence"
            seq = seqs["protein"][0]
            tokenized = self._tokenize_single(seq, self.protein_tokenizer, self.protein_k_tokens)

            batch.update({
                "x1": tokenized["input_ids"].squeeze(0),
                "mask1": tokenized["attention_mask"].squeeze(0)
            })
        elif self.model_type == "NT+ESM":
            assert self.dna_rna_tokenizer is not None, "dna_rna_tokenizer is required for NT+ESM model"
            assert self.protein_tokenizer is not None, "protein_tokenizer is required for NT+ESM model"
            assert (len(seqs["dna"]) + len(seqs["rna"]) == 1) and (len(seqs["protein"]) == 1), \
                "NT+ESM model requires exactly one DNA/RNA sequence and one Protein sequence"
            dna_rna_seq = seqs["dna"][0] if len(seqs["dna"]) == 1 else seqs["rna"][0]
            protein_seq = seqs["protein"][0]

            dna_rna_tokenized = self._tokenize_single(dna_rna_seq, self.dna_rna_tokenizer, self.dna_rna_k_tokens)
            protein_tokenized = self._tokenize_single(protein_seq, self.protein_tokenizer, self.protein_k_tokens)

            batch.update({
                "x1": dna_rna_tokenized["input_ids"].squeeze(0),
                "mask1": dna_rna_tokenized["attention_mask"].squeeze(0),
                "x2": protein_tokenized["input_ids"].squeeze(0),
                "mask2": protein_tokenized["attention_mask"].squeeze(0)
            })
        elif self.model_type == "NT+NT":
            assert self.dna_rna_tokenizer is not None, "dna_rna_tokenizer is required for NT+NT model"
            assert len(seqs["dna"]) + len(seqs["rna"]) == 2, "NT+NT model requires exactly two DNA/RNA sequences"
            seqs_list = seqs["dna"] + seqs["rna"]
            tokenized_1 = self._tokenize_single(seqs_list[0], self.dna_rna_tokenizer, self.dna_rna_k_tokens)
            tokenized_2 = self._tokenize_single(seqs_list[1], self.dna_rna_tokenizer, self.dna_rna_k_tokens)

            batch.update({
                "x1": tokenized_1["input_ids"].squeeze(0),
                "mask1": tokenized_1["attention_mask"].squeeze(0),
                "x2": tokenized_2["input_ids"].squeeze(0),
                "mask2": tokenized_2["attention_mask"].squeeze(0)
            })
        elif self.model_type == "ESM+ESM":
            assert self.protein_tokenizer is not None, "protein_tokenizer is required for ESM+ESM model"
            assert len(seqs["protein"]) == 2, "ESM+ESM model requires exactly two Protein sequences"
            tokenized_1 = self._tokenize_single(seqs["protein"][0], self.protein_tokenizer, self.protein_k_tokens)
            tokenized_2 = self._tokenize_single(seqs["protein"][1], self.protein_tokenizer, self.protein_k_tokens)

            batch.update({
                "x1": tokenized_1["input_ids"].squeeze(0),
                "mask1": tokenized_1["attention_mask"].squeeze(0),
                "x2": tokenized_2["input_ids"].squeeze(0),
                "mask2": tokenized_2["attention_mask"].squeeze(0)
            })
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        return batch


class ClassificationCollator:
    """
    Batch everything that ClassificationDataset returns.
    支持：
        - NT / ESM：       input_ids, attention_mask
        - NT+ESM：         dna_rna_input_ids, dna_rna_attention_mask,
                           protein_input_ids, protein_attention_mask
        - NT+NT / ESM+ESM：*_1_* 与 *_2_* 两组 ids & mask
    """

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: list 长度 = batch_size，每个元素是 __getitem__ 返回的 dict
        Returns:
            dict，所有张量已 padding 并拼成 batch
        """

        labels = torch.stack([f.pop("labels") for f in features])

        batch = {"labels": labels}

        all_keys = set()
        for feat in features:
            all_keys.update(feat.keys())

        for key in all_keys:
            tensors = [feat[key] for feat in features]
            batch[key] = pad_sequence(tensors, batch_first=True, padding_value=1)

        return batch
