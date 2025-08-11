# pylint: skip-file
"""
统计 dev_new.jsonl 中 task 字段的取值种类并打印
然后计算符合筛选条件任务的数据总量（模糊匹配）
"""

import json
from collections import Counter
from pathlib import Path

file_path = Path("/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/dev_new.jsonl")

# 1. 筛选任务名列表（模糊匹配）
task_names = [
    "cpd-prom_core",
    "FunctionEC-FunctionEC",
    "Solubility-Solubility",
    "antibody_antigen",
    "rna_protein_interaction",
    "tf-m",
    "pd-prom_300",
    "NoncodingRNAFamily-NoncodingRNAFamily",
    "Modification-Modification",
    "promoter_enhancer",
]  # 需要模糊匹配的任务名

# 2. 逐行读取，收集所有任务
tasks = []
with file_path.open(encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        line = line.strip()
        if not line:  # 跳过空行
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[警告] 第 {line_no} 行解析失败：{e}")
            continue
        task = data.get("task")
        if task is not None:
            tasks.append(task)

# 3. 统计所有任务
counter = Counter(tasks)
print(f"共发现 {len(counter)} 种 task：")
for task, cnt in counter.items():
    print(f"  {task} : {cnt} 条")

# 4. 统计符合筛选条件任务的数据总量（模糊匹配）
filtered_count = sum(
    cnt for task, cnt in counter.items() if any(name in task for name in task_names)
)
print(f"\n总共找到 {filtered_count} 条符合筛选条件的任务数据。")
