#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge jsonl files with <dna>/<rna>/<protein> tags into one Parquet,
while validating that <dna>/<rna> sequences only contain IUPAC-compliant bases.
"""

import json
import re
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def merge_biomarkers_to_parquet(
    in_dir: str | Path,
    jsonl_files: Iterable[str | Path],
    out_parquet: str | Path,
    *,
    markers: Iterable[str] = ("dna", "rna"),
) -> pd.DataFrame:
    # ---------- 预处理 ----------
    in_dir = Path(in_dir)
    out_parquet = Path(out_parquet)
    markers = tuple(m.lower() for m in markers)

    tag_re = re.compile(rf"<({'|'.join(markers)})>(.*?)<\1>", re.IGNORECASE | re.DOTALL)

    # IUPAC 合法大写碱基正则
    nt_base_regex = re.compile(r"^[ACGT]+$")

    allow_base = {
        "dna": nt_base_regex,
        "rna": nt_base_regex,
    }

    removed_empty_label = 0
    removed_invalid_seq = 0
    records: List[dict] = []

    # ---------- 处理单条 ----------
    def parse_line(obj: dict) -> None:
        nonlocal removed_empty_label, removed_invalid_seq
        raw_input = obj.get("input", "")
        label_val = obj.get("label")

        if label_val in (None, "", {}) or label_val is False:
            removed_empty_label += 1
            return

        label_str = json.dumps(label_val, ensure_ascii=False) if isinstance(label_val, dict) else str(label_val)

        seqs: List[str] = []
        kinds: List[str] = []

        def _strip(m: re.Match) -> str:
            kind = m.group(1).lower()
            seq_raw = m.group(2)
            seq_clean = re.sub(r"[^A-Za-z]", "", seq_raw).upper()  # 大写化 + 清除非字母字符
            kinds.append(kind)
            seqs.append(seq_clean)
            return ""

        cleaned_input = tag_re.sub(_strip, raw_input).strip()

        for k, s in zip(kinds, seqs):
            if k in allow_base and not allow_base[k].fullmatch(s):
                print(f"[filtered] kind={k}, invalid sequence: {s}")
                removed_invalid_seq += 1
                return

        kind_str = "-".join(kinds)

        records.append(
            {
                "task": obj.get("task", ""),
                "input": cleaned_input,
                "think": "",
                "output": obj.get("output", ""),
                "label": label_str,
                "sequence": seqs,
                "kind": kind_str,
            }
        )

    # ---------- 扫描所有文件 ----------
    for fname in jsonl_files:
        fpath = (in_dir / fname) if not Path(fname).is_absolute() else Path(fname)
        with open(fpath, encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    removed_invalid_seq += 1
                    continue
                parse_line(obj)

    # ---------- 保存 ----------
    df = pd.DataFrame(
        records,
        columns=["task", "input", "think", "output", "label", "sequence", "kind"],
    )
    df.to_parquet(out_parquet, engine="pyarrow", index=False)

    print(
        f"✅ Parquet saved: {out_parquet}\n"
        f"   kept                      : {len(df)}\n"
        f"   removed_empty_label       : {removed_empty_label}\n"
        f"   removed_invalid_sequence  : {removed_invalid_seq}"
    )
    return df


# ------------------- 示例调用 -------------------
if __name__ == "__main__":
    base_dir = "/fs-computility/ai4agr/lijinzhe/code/BioMLLM_V2/data_tools/sample"
    files = [
        "balanced_dna_rna_val_dataset.jsonl",
    ]
    out_path = f"{base_dir}/val_nt.parquet"

    merge_biomarkers_to_parquet(
        base_dir,
        files,
        out_path,
        markers=("dna", "rna"),  # 后续可拓展 protein
    )
