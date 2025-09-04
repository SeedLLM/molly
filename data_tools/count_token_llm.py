#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 Qwen3-4B tokenizer 统计 parquet 文件中 output 列的 token 数量，并打印结果
"""

import argparse
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm


def count_tokens(parquet_file: Path, model_path: Path, column: str = "output") -> None:
    # 1. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 2. 读取 parquet
    df = pd.read_parquet(parquet_file, engine="pyarrow")
    if column not in df.columns:
        raise ValueError(f"列 {column} 不存在，当前列有: {list(df.columns)}")

    # 3. 统计
    total_tokens = 0
    token_counts = []

    for text in tqdm(df[column].astype(str), desc="Counting tokens"):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        n = len(tokens)
        # if n <= 20:
        #     print(text)
        token_counts.append(n)
        total_tokens += n

    # 4. 打印统计结果
    print("✅ 统计完成")
    print(f"文件: {parquet_file}")
    print(f"模型: {model_path}")
    print(f"总行数: {len(df)}")
    print(f"总 token 数: {total_tokens}")
    print(f"平均 token 数: {total_tokens / len(df):.2f}")
    print(f"最大 token 数: {max(token_counts)}")
    print(f"最小 token 数: {min(token_counts)}")


def parse_args():
    p = argparse.ArgumentParser(description="使用 Qwen3-4B tokenizer 统计 parquet 文件中 output 列的 token 数")
    p.add_argument(
        "--parquet",
        type=Path,
        default=Path("/share/org/YZWL/yzwl_lijz/dataset_res/omics_data/CotExp/COT/Enhancer-Promoter/cot.parquet"),
        help="输入 parquet 文件路径",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=Path("/share/org/YZWL/yzwl_lijz/base_llm/Qwen3-4B"),
        help="Qwen3-4B 模型目录",
    )
    p.add_argument("--column", type=str, default="output", help="要统计的列 (默认 output)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    count_tokens(args.parquet, args.model, args.column)