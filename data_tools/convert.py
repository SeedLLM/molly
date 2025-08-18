"""
用于将一些公开数据例如bioreason转换为BioMLLM的格式
"""
# pylint: skip-file
import glob
import json
import os
from typing import List, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# bioreson kegg
def convert_kegg_parquet_format(src_paths: List[str], dst_path: str) -> None:
    """
    将多个源 Parquet 文件合并并转换为指定格式的新 Parquet 文件。

    参数
    ----
    src_paths : List[str]
        原始 parquet 文件路径列表（支持绝对/相对路径）。
    dst_path : str
        转换后唯一 parquet 文件路径（若包含目录需保证目录已存在）。
    """
    dfs = [pd.read_parquet(p) for p in src_paths]
    df = pd.concat(dfs, ignore_index=True)

    new_input = []
    new_output = []

    for _, row in df.iterrows():
        # 构造 input
        inp = (
            "<dna>"
            + str(row["reference_sequence"])
            + "<dna>"
            + "<dna>"
            + str(row["variant_sequence"])
            + "<dna>"
            + str(row["question"])
        )
        new_input.append(inp)

        # 构造 output
        out = (
            "<think>\n"
            + str(row["reasoning"])
            + "\n</think>\n"
            + "\nAnswer: "
            + str(row["answer"])
        )
        new_output.append(out)

    new_df = pd.DataFrame(
        {
            "task": "kegg",
            "input": new_input,
            "think": "",
            "output": new_output,
            "label": df["answer"].astype(str),
            "kind": "dna-dna",
        }
    )

    # 4. 写出
    table = pa.Table.from_pandas(new_df, preserve_index=False)
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    pq.write_table(table, dst_path, compression="snappy")
    print(f"转换完成，已生成：{dst_path}")


# bioreason vec
def convert_bioreason_vec_format(src_paths: List[str], dst_path: str) -> None:
    """
    将多个源 Parquet 文件合并并转换为指定格式的新 Parquet 文件。
    """
    dfs = [pd.read_parquet(p) for p in src_paths]
    df = pd.concat(dfs, ignore_index=True)

    new_input = []
    new_output = []

    for idx, row in df.iterrows():
        ref = str(row["reference_sequence"]).strip()
        var = str(row["variant_sequence"]).strip()

        # 跳过空序列
        if not ref or not var:
            print(
                f"[跳过] 行号 {idx}: reference_sequence={repr(ref)}, variant_sequence={repr(var)}"
            )
            continue
        # 构造 input
        inp = (
            "<dna>"
            + str(row["reference_sequence"])
            + "<dna>"
            + "<dna>"
            + str(row["variant_sequence"])
            + "<dna>"
            + str(row["question"])
        )
        new_input.append(inp)

        # 构造 output
        answer_str = str(row["answer"])
        # 检测是否包含换行符
        if "\n" in answer_str:
            print(f"警告: answer 包含换行符: {answer_str[:100]}...")

        out = (
            "<think>\n"
            + str(row["answer"])
            + "\n</think>\n"
            + "\nAnswer: "
            + str(row["answer"])
        )
        new_output.append(out)

    new_df = pd.DataFrame(
        {
            "task": "kegg",
            "input": new_input,
            "think": "",
            "output": new_output,
            "label": df["answer"].astype(str),
            "kind": "dna-dna",
        }
    )

    # 4. 写出
    table = pa.Table.from_pandas(new_df, preserve_index=False)
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    pq.write_table(table, dst_path, compression="snappy")
    print(f"转换完成，已生成：{dst_path}")


# 将模型输出的结果变为测评的格式（bio-instruct）
def convert_multi_json(
    src_paths: Union[str, List[str]],
    dst_path: str,
    *,
    src_encoding: str = "utf-8",
    dst_encoding: str = "utf-8",
    keep_separate: bool = False,  # True: 每个源文件单独输出
) -> None:
    """
    批量转换多个 JSON/JSONL 文件到评测格式。

    src_paths : 单个文件、文件列表或含通配符的文件夹路径
    dst_path  : 输出文件或输出目录（keep_separate=True 时）
    keep_separate: False 合并输出；True 保持与源文件同名
    """
    if isinstance(src_paths, str):
        # 支持通配符 / 文件夹
        if os.path.isdir(src_paths):
            src_files = glob.glob(os.path.join(src_paths, "*.json*"))
        else:
            src_files = [src_paths]
    else:
        src_files = src_paths

    if keep_separate:
        os.makedirs(dst_path, exist_ok=True)

    merged_data = []

    for src in src_files:
        with open(src, "r", encoding=src_encoding) as f:
            records = [json.loads(line) for line in f if line.strip()]

        converted = [
            {
                "task": rec.get("task"),
                "input": rec.get("input"),
                "model_output": rec.get("decoded_output"),
                "label": rec.get("gt_label"),
                "kind": rec.get("kind"),
            }
            for rec in records
        ]

        if keep_separate:
            # 保持原文件名，仅后缀改为 .jsonl
            base_name = os.path.splitext(os.path.basename(src))[0] + ".jsonl"
            out_file = os.path.join(dst_path, base_name)
            with open(out_file, "w", encoding=dst_encoding) as fout:
                for item in converted:
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"✅ 单独输出 {len(converted)} 条 → {out_file}")
        else:
            merged_data.extend(converted)

    if not keep_separate:
        os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
        with open(dst_path, "w", encoding=dst_encoding) as fout:
            for item in merged_data:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"✅ 合并输出 {len(merged_data)} 条 → {dst_path}")


# -------------------- 使用示例 --------------------
if __name__ == "__main__":
    # 1. 填写你的实际路径
    # src_list = [
    #     "/tos-bjml-ai4agr/lijinzhe/dataset/variant_effect_coding/train-00000-of-00001.parquet"
    #     # "/tos-bjml-ai4agr/lijinzhe/dataset/kegg/data/train-00000-of-00001.parquet"
    # ]
    # dst_file = "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/VariantEC/train.parquet"

    # 2. 调用函数
    # convert_kegg_parquet_format(src_list, dst_file)
    # convert_bioreason_vec_format(src_list, dst_file)
    src_paths = (
        "/fs-computility/ai4agr/lijinzhe/code/BioMLLM_V2/res/inference/Qwen3_4B_Omics_sft_dna_protein_task_exp2_5epoch"
    )
    # dst_path = "/fs-computility/ai4agr/lijinzhe/code/BioMLLM_V2/res/bio_instruct_judge/Omics_sft_0805_exp1_16k.json"
    dst_path = "/fs-computility/ai4agr/lijinzhe/code/BioMLLM_V2/res/bio_instruct_judge/Qwen3_4B_Omics_sft_dna_protein_task_exp2_5epoch.json"
    convert_multi_json(src_paths, dst_path)
