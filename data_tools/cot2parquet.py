"""
Convert TF-M JSON arrays under a directory into one Parquet, with:
- reserved token replacement:
    <|reserved_special_token_1|> -> <dna>
    <|reserved_special_token_2|> -> <rna>
    <|reserved_special_token_3|> -> <protein>
  and ensure closing tags </dna>, </rna>, </protein>.
- IUPAC validation for sequences inside <dna>/<rna>/<protein> ... </...>
- Parquet schema identical to the user's reference: columns =
  ["task", "input", "think", "output", "label", "kind"]
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple

import pandas as pd


# --------- constants ---------
IDX2MARKER = {"1": "dna", "2": "rna", "3": "protein"}

# 匹配占位符：<|reserved_special_token_1|> 或 </|reserved_special_token_1|>
RESERVED_TAG_RE = re.compile(r"(</?)\|reserved_special_token_(\d+)\|>", re.IGNORECASE)

# DNA/RNA/Protein IUPAC regex
NT_REGEX = re.compile(r"^[ACGTN]+$")
AA_REGEX = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBXZOU]+$")

ALLOW_BASE = {"dna": NT_REGEX, "rna": NT_REGEX, "protein": AA_REGEX}


def replace_reserved_tokens(text: str) -> str:
    """将占位符替换为统一形式的 <dna>/<rna>/<protein>（忽略开闭合差异）。"""
    if not text:
        return text

    def _repl(m: re.Match) -> str:
        idx = m.group(2)
        marker = IDX2MARKER.get(idx)
        if not marker:
            return m.group(0)
        return f"<{marker}>"

    return RESERVED_TAG_RE.sub(_repl, text)


def unify_to_open_tags(text: str, markers: Tuple[str, ...]) -> str:
    """把 </dna>/<rna>/<protein> 统一替换为 <dna>/<rna>/<protein>（大小写不敏感）。"""
    if not text:
        return text
    for m in markers:
        text = re.sub(rf"</{re.escape(m)}\s*>", f"<{m}>", text, flags=re.IGNORECASE)
    return text

def normalize_nt(kind: str, seq_raw: str) -> str:
    """去非字母 + 大写；RNA 将 U->T。"""
    seq = re.sub(r"[^A-Za-z]", "", (seq_raw or "")).upper()
    # if kind == "rna":
    #     seq = seq.replace("U", "T")
    return seq


def extract_and_validate_kinds(raw_input: str, markers: Tuple[str, ...]) -> Tuple[str, bool]:
    """
    抽取 <marker>...<marker> 之间的序列（开闭合标签已统一成相同形式）
    返回 (kind_str, is_valid).
    """
    if not raw_input:
        return "", True

    # 使用 <marker> ... <marker> 来抽取（因为没有闭合标签了，改为宽松匹配）
    tag_re = re.compile(rf"<({'|'.join(map(re.escape, markers))})>([A-Za-z]+)<\1>", re.IGNORECASE)
    kinds, seqs = [], []
    for m in tag_re.finditer(raw_input):
        kind = m.group(1).lower()
        payload = m.group(2)
        seq = normalize_nt(kind, payload)
        kinds.append(kind)
        seqs.append(seq)

    for k, s in zip(kinds, seqs):
        rgx = ALLOW_BASE.get(k)
        if rgx is not None and not rgx.fullmatch(s):
            return "-".join(kinds), False

    return "-".join(kinds), True


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} 顶层不是 JSON 数组。")
    return data


def pick(d: Dict[str, Any], path: List[str], default: str = "") -> str:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur if isinstance(cur, str) else default


def convert_dir_to_parquet(
    in_dir: Path,
    out_parquet: Path,
    *,
    markers: Tuple[str, ...] = ("dna", "rna", "protein"),
    parquet_engine: str = "pyarrow",
) -> pd.DataFrame:
    removed_empty_label = 0
    removed_invalid_seq = 0
    bad_json_files = 0
    kept = 0

    records: List[Dict[str, Any]] = []

    json_files = sorted(in_dir.glob("*.json"))

    for fp in json_files:
        print(fp)
        try:
            arr = load_json_array(fp)
        except Exception as e:
            print(f"[skip] 解析失败: {fp} ({e})")
            bad_json_files += 1
            continue

        for obj in arr:
            raw_input = obj.get("input", "")
            text = replace_reserved_tokens(raw_input)

            text = unify_to_open_tags(text, markers)

            # label
            if "label" in obj and obj["label"] not in (None, "", {}, False):
                label_str = obj["label"] if isinstance(obj["label"], str) else json.dumps(obj["label"], ensure_ascii=False)
            else:
                lbl = pick(obj, ["metadata", "meta_data", "label"], "")
                if lbl in ("", None):
                    removed_empty_label += 1
                    continue
                label_str = lbl

            # output
            output_str = obj.get("output") or obj.get("answer", "")

            # task
            task_str = obj.get("task", "") or pick(obj, ["metadata", "meta_data", "task"], "")

            # kind & validation
            kind_str, valid = extract_and_validate_kinds(text, markers)
            if not valid:
                removed_invalid_seq += 1
                continue
            # print(task_str, text, output_str, label_str, kind_str)
            if len(output_str) <= 5:
                continue
            records.append(
                {
                    "task": task_str,
                    "input": text,  # 已统一成只有开标签
                    "think": "",
                    "output": output_str,
                    "label": label_str,
                    "kind": kind_str,
                }
            )
            kept += 1

    df = pd.DataFrame(records, columns=["task", "input", "think", "output", "label", "kind"])
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, engine=parquet_engine, index=False)

    print(
        "✅ Parquet saved\n"
        f"   path                      : {out_parquet}\n"
        f"   kept                      : {len(df)}\n"
        f"   removed_empty_label       : {removed_empty_label}\n"
        f"   removed_invalid_sequence  : {removed_invalid_seq}\n"
        f"   bad_json_files            : {bad_json_files}\n"
    )
    return df

# 在这里修改任务
Kind = "Solubility"
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert TF-M JSON arrays to a single Parquet with biomarker validation.")
    p.add_argument(
        "--in-dir",
        type=Path,
        default=Path(f"/share/org/YZWL/yzwl_lijz/dataset_res/omics_data/CotExp/rawCOT/{Kind}"),
        help="输入目录（包含多个 .json 文件）",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(f"/share/org/YZWL/yzwl_lijz/dataset_res/omics_data/CotExp/COT/{Kind}/cot.parquet"),
        help="输出 Parquet 路径",
    )
    p.add_argument(
        "--markers",
        type=str,
        default="dna,rna,protein",
        help="以逗号分隔的标签集合（默认 dna,rna,protein）",
    )
    p.add_argument(
        "--engine",
        type=str,
        default="pyarrow",
        choices=["pyarrow", "fastparquet"],
        help="Parquet 引擎（默认 pyarrow）",
    )
    return p.parse_args()


def main():
    args = parse_args()
    markers = tuple(m.strip().lower() for m in args.markers.split(",") if m.strip())
    convert_dir_to_parquet(args.in_dir, args.out, markers=markers, parquet_engine=args.engine)


if __name__ == "__main__":
    main()