# pylint: skip-file
# 将parquet文件按照比例进行划分
import os
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split


def split_parquet_evenly(
    src_path: str, dst_dir: str, n: int, *, keep_index: bool = False
) -> List[str]:
    """
    将单个 parquet 文件按行平均切成 n 份，并保存到 dst_dir。
    返回生成的文件路径列表（按顺序）。
    """
    if n <= 0:
        raise ValueError("n must be positive")

    os.makedirs(dst_dir, exist_ok=True)

    df = pd.read_parquet(src_path)
    total_rows = len(df)
    rows_per_part = total_rows // n
    remainder = total_rows % n

    file_paths = []
    start = 0
    for i in range(n):
        # 余数均匀分摊到前面几份
        end = start + rows_per_part + (1 if i < remainder else 0)
        part_df = df.iloc[start:end].reset_index(drop=not keep_index)
        out_file = os.path.join(dst_dir, f"part_{i}.parquet")
        part_df.to_parquet(out_file, engine="pyarrow", index=keep_index)
        file_paths.append(out_file)
        start = end

    print(f"✅ 已切分为 {n} 份，保存在 {dst_dir}")
    return file_paths


def write_task_distribution(
    src_path: str, dst_dir: str, *, file_name: str = "task_distribution.txt"
) -> str:
    """
    统计 parquet 文件中 task 字段的分布并写出 txt。
    返回输出文件完整路径。
    """
    os.makedirs(dst_dir, exist_ok=True)
    out_file = os.path.join(dst_dir, file_name)

    df = pd.read_parquet(src_path, columns=["task"])
    counts = df["task"].value_counts().sort_values(ascending=False)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("task\tcount\n")
        for task, cnt in counts.items():
            f.write(f"{task}\t{cnt}\n")

    print(f"✅ task 分布已保存至 {out_file}")
    return out_file


# 按照任务等比例切割数据
def split_parquet_by_task(
    input_file: str,
    output_file_part1: str,
    output_file_part2: str,
    test_size: float = 0.3,
    random_state: int = 42,
):
    """
    将parquet文件按照'task'字段分成两份，并保存为两个新的parquet文件。

    参数：
    - input_file: 原始parquet文件路径。
    - output_file_part1: 分割后第一部分保存的parquet文件路径。
    - output_file_part2: 分割后第二部分保存的parquet文件路径。
    - test_size: 分割比例，默认为0.5，即50%分配给每部分。
    - random_state: 随机种子，确保每次分割一致。
    """
    # 读取 parquet 文件
    df = pd.read_parquet(input_file)

    # 按照 'task' 字段分组
    grouped = df.groupby("task")

    # 分为两份
    df_part1 = []
    df_part2 = []

    for _, group in grouped:
        # 使用 train_test_split 来按比例拆分每个任务组
        part1, part2 = train_test_split(
            group, test_size=test_size, random_state=random_state
        )  # 0.5表示50%的比例
        df_part1.append(part1)
        df_part2.append(part2)

    # 将拆分后的两部分合并
    df_part1 = pd.concat(df_part1, axis=0)
    df_part2 = pd.concat(df_part2, axis=0)

    # 保存为两个新的 parquet 文件
    df_part1.to_parquet(output_file_part1, index=False)
    df_part2.to_parquet(output_file_part2, index=False)

    print(
        f"Data split successfully into two parts: '{output_file_part1}' and '{output_file_part2}'"
    )


def sample_parquet_data(input_file_path, output_file_path):
    """
    读取Parquet文件，根据任务字段进行数据采样：
    - 'NoncodingRNAFamily-NoncodingRNAFamily'任务保留50%数据
    - 其他任务保留8%数据
    最后将采样后的数据保存到指定的输出Parquet文件中。

    :param input_file_path: 输入Parquet文件路径
    :param output_file_path: 输出Parquet文件路径
    """
    # 读取Parquet文件
    df = pd.read_parquet(input_file_path)

    # 定义一个函数，根据任务名称选择采样比例
    def sample_data(group):
        task_name = group["task"].iloc[0]
        if task_name == "NoncodingRNAFamily-NoncodingRNAFamily":
            return group.sample(frac=0.5, random_state=42)
        else:
            return group.sample(frac=0.08, random_state=42)

    # 按任务进行分组并采样
    sampled_df = df.groupby("task").apply(sample_data).reset_index(drop=True)

    # 输出到新的Parquet文件
    sampled_df.to_parquet(output_file_path)


# ---- 使用示例 ----
if __name__ == "__main__":
    src = "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/TargetTask0729/dev_target_task.parquet"
    dst = "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/TargetTask0729/API_Infer"
    n_parts = 4  # 改成你想要的份数

    split_parquet_evenly(src, dst, n_parts)
    # write_task_distribution(src, dst)
#     split_parquet_by_task(
#     input_file='/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/TargetTask0729/train_wo_s3_target_task.parquet',
#     output_file_part1='/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/TargetTask0729/train_part1.parquet',
#     output_file_part2='/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/TargetTask0729/train_part2.parquet'
# )
# input_file_path = "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/TargetTask0729/train_wo_s3_target_task.parquet"
# output_file_path = "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/TargetTask0729/sample_train_wo_s3_target_task.parquet"

# sample_parquet_data(input_file_path, output_file_path)
