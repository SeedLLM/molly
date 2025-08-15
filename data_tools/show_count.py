# pylint: skip-file
from collections import defaultdict

import pandas as pd


def count_rows_in_parquet(files: list):
    """
    This function takes a list of Parquet file paths, reads each file, and returns the number of rows.

    Args:
    files (list): List of paths to Parquet files.

    Returns:
    dict: A dictionary with file paths as keys and number of rows as values.
    """
    file_row_counts = {}

    for file_path in files:
        try:
            # Load the Parquet file into a DataFrame
            df = pd.read_parquet(file_path)
            # Store the number of rows
            file_row_counts[file_path] = len(df)
        except Exception as e:
            # If an error occurs, store the error message
            file_row_counts[file_path] = f"Error: {e}"

    return file_row_counts


def count_task_in_parquet(input_file, task_names):
    """
    统计输入的 Parquet 文件中，每个 task 匹配 task_names 序列中的数据量。

    参数:
    - input_file: Parquet 文件路径。
    - task_names: 包含多个任务名称的列表。

    返回:
    - dict: 每个 task 模式对应的数据量
    """
    task_count = defaultdict(int)  # 使用 defaultdict 来统计每个任务的数量

    try:
        # 读取 Parquet 文件
        df = pd.read_parquet(input_file)

        # 确保文件中有 task 列
        if "task" not in df.columns:
            print(f"Error: The 'task' column is not found in {input_file}.")
            return {}

        # 遍历 task 列并统计任务数量
        for task in df["task"]:
            for task_name in task_names:
                if task_name in task:
                    task_count[task_name] += 1
                    break  # 一旦匹配上一个任务名，就停止继续检查其他任务名
    except Exception as e:
        print(f"Error processing file {input_file}: {e}")

    return dict(task_count)  # 将 defaultdict 转换为普通 dict 返回


if __name__ == "__main__":
    # input_file = '/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/TargetTask0808/all2stage3/stage3_target_task_0808.parquet'  # Parquet 文件路径
    # task_names = ["cpd-prom_core", "FunctionEC-FunctionEC",
    #               "Solubility-Solubility", "antibody_antigen",
    #               "rna_protein_interaction", "tf-m", "pd-prom_300",
    #               "NoncodingRNAFamily-NoncodingRNAFamily",
    #               "Modification-Modification", "promoter_enhancer"]  # 需要模糊匹配的任务名

    # # 统计每个任务在 Parquet 文件中的数据量
    # task_counts = count_task_in_parquet(input_file, task_names)
    # print(f"Task counts: {task_counts}")

    # count_rows_in_parquet Example usage:
    files = [
        "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/TargetTask0808/val_10_task.parquet",
        "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/TargetTask0808/train_10_task.parquet",
        "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/TargetTask0808/train_10_task_wos3.parquet",
        "/fs-computility/ai4agr/lijinzhe/code/BioEval/filtered_tasks.parquet",
        "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/Ablation/DNA/val_dna_task.parquet",
        "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/Ablation/DNA/train_dna_task.parquet",
        "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/Ablation/RNA/train_rna_task.parquet",
        "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/Ablation/RNA/val_rna_task.parquet",
        "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/Ablation/Protein/val_protein_task.parquet",
        "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/Ablation/Protein/train_protein_task.parquet",
        "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/Ablation/DNA_RNA/train_dna_rna_task.parquet",
        "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/Ablation/DNA_RNA/val_dna_rna_task.parquet",
        "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/Ablation/DNA_Protein/val_dna_protein_task.parquet",
        "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/Ablation/DNA_Protein/train_dna_protein_task.parquet",
        "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/Ablation/RNA_Protein/val_rna_protein_task.parquet",
        "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/Ablation/RNA_Protein/train_rna_protein_task.parquet",
        # '/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/TargetTask0729/training/train_wo_s3_all.parquet'
    ]

    row_counts = count_rows_in_parquet(files)
    for file, count in row_counts.items():
        print(f"File: {file}, Rows: {count}")
