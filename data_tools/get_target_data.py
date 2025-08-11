# pylint: skip-file
import json
from collections import defaultdict


def filter_jsonl_by_task(input_files, task_names, output_file):
    """
    筛选出 JSONL 中 task 包含在给定 task_names 序列中的数据，并将其保存到新的 JSONL 文件中。

    参数:
    - input_files: 包含多个 JSONL 文件路径的 list。
    - task_names: 包含多个任务名称的列表。
    - output_file: 结果保存的 JSONL 文件路径。
    """
    filtered_data = []

    # 遍历每个输入的 JSONL 文件
    for file_path in input_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line.strip())
                    task = data.get("task", "")
                    # 如果数据中的 task 匹配任意一个任务名，则添加到结果列表中
                    if any(task_name in task for task_name in task_names):
                        filtered_data.append(data)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    # 将筛选的数据保存到新的 JSONL 文件中
    with open(output_file, "w", encoding="utf-8") as f:
        for item in filtered_data:
            f.write(json.dumps(item) + "\n")

    print(f"Filtered data saved to {output_file}.")


def count_task_in_jsonl(input_files, task_names):
    """
    统计输入的 JSONL 文件中，每个 task 匹配 task_names 序列中的数据量。

    参数:
    - input_files: 包含多个 JSONL 文件路径的 list。
    - task_names: 包含多个任务名称的列表。

    返回:
    - dict: 每个 task 模式对应的数据量
    """
    task_count = defaultdict(int)  # 使用 defaultdict 来统计每个任务的数量

    # 遍历每个输入的 JSONL 文件
    for file_path in input_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line.strip())
                    task = data.get("task", "")
                    # 如果数据中的 task 匹配 task_names 中的任意一个任务名，则计数
                    for task_name in task_names:
                        if task_name in task:
                            task_count[task_name] += 1
                            break  # 一旦匹配上一个任务名，就停止继续检查其他任务名
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    return dict(task_count)  # 将 defaultdict 转换为普通 dict 返回


if __name__ == "__main__":
    # 假设你的两个文件是 file1.jsonl 和 file2.jsonl
    # input_files = ["/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/stage3_train_data/stage3_train_new.jsonl"]
    input_files = ["/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/dev_new.jsonl"]
    # task_names = ["tf-m", "pd-prom_300", "NoncodingRNAFamily-NoncodingRNAFamily", "Modification-Modification", "promoter_enhancer"]  # 筛选的任务名列表（完全匹配）
    # task_names = ["Solubility-Solubility", "antibody_antigen", "rna_protein_interaction"]  # 筛选的任务名列表（完全匹配）
    # task_names = ["cpd-prom_core", "FunctionEC-FunctionEC"]
    # task_names = ["cpd-prom_core", "FunctionEC-FunctionEC",
    #           "Solubility-Solubility", "antibody_antigen",
    #           "rna_protein_interaction", "tf-m", "pd-prom_300",
    #           "NoncodingRNAFamily-NoncodingRNAFamily",
    #           "Modification-Modification", "promoter_enhancer"]  # 需要模糊匹配的任务名
    task_names = ["Solubility-Solubility"]  # 需要模糊匹配的任务名
    output_file = "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/TargetTask0808/Solubility/val_solubility_task_0808.jsonl"
    # output_file = '/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/TargetTask0808/check/train_target_task_0808.jsonl'  # 筛选后的数据保存到这个文件中

    # 1. 筛选数据并保存到新的 JSONL 文件中
    filter_jsonl_by_task(input_files, task_names, output_file)

    # 2. 统计每个任务在所有文件中的数据量
    task_counts = count_task_in_jsonl(input_files, task_names)
    print(f"Task counts: {task_counts}")
