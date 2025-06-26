"""
data analysis
"""
import os
import json
from typing import List, Dict
from collections import defaultdict

import matplotlib.pyplot as plt 

class JsonlDatasetInspector:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def count_lines(self) -> int:
        """统计 JSONL 文件的总行数"""
        count = 0
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for _ in f:
                count += 1
        return count

    def preview(self, num_lines: int = 10) -> List[dict]:
        """返回并打印前 num_lines 条样本"""
        samples = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                try:
                    data = json.loads(line)
                    samples.append(data)
                except json.JSONDecodeError:
                    print(f"第 {i+1} 行 JSON 解码失败：{line.strip()}")
        for i, sample in enumerate(samples):
            print(f"\n样本 {i+1}:\n{json.dumps(sample, ensure_ascii=False, indent=2)}")
        return samples

    def count_tasks(self, visualize: bool = True, save_path: str = "task_distribution_pie.png") -> Dict[str, int]:
        """
        统计每个 task 的样本数量，并打印总任务数。
        如果 visualize=True，则绘制饼图。
        """
        task_counter = defaultdict(int)
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    task_name = data.get("task", "UNKNOWN")
                    task_counter[task_name] += 1
                except json.JSONDecodeError:
                    print(f"第 {i+1} 行 JSON 解码失败，跳过")

        print("\n任务统计结果：")
        for task, count in task_counter.items():
            print(f"{task}: {count} 条")
        print(f"\n📊 总共 {len(task_counter)} 种任务")

        if visualize:
            labels = list(task_counter.keys())
            sizes = list(task_counter.values())

            # 生成颜色列表以确保 legend 与 wedge 一致
            colors = plt.cm.tab20.colors  # 可换成 tab20b, tab20c, Set3 等

            plt.figure(figsize=(16, 15))
            wedges, texts, autotexts = plt.pie(
                sizes,
                autopct='%1.1f%%',
                startangle=140,
                colors=colors[:len(sizes)],
                textprops={'fontsize': 8},
            )

            # 添加图例 legend 在右边
            plt.legend(
                wedges,
                labels,
                title="Tasks",
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                fontsize=12,
                title_fontsize=13
            )

            plt.title('Sample Distribution by Task', fontsize=16)
            plt.axis('equal')
            plt.tight_layout()

            # 保存图像
            plt.savefig(save_path, bbox_inches="tight")  # 防止图例被截断
            print(f"\n✅ 饼图已保存到: {os.path.abspath(save_path)}")


        return dict(task_counter)

    def visualize_per_task(self, num_samples_per_task: int = 3) -> Dict[str, List[dict]]:
        """
        按 task 抽取指定数量样本，并打印展示
        返回每个 task 对应的样本列表
        """
        task_samples = defaultdict(list)
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    task = data.get("task", "UNKNOWN")
                    if len(task_samples[task]) < num_samples_per_task:
                        task_samples[task].append(data)
                except json.JSONDecodeError:
                    continue

        for task, samples in task_samples.items():
            print(f"\n【任务: {task}】展示前 {len(samples)} 条样本:")
            for i, sample in enumerate(samples):
                print(f"\n  样本 {i+1}:\n{json.dumps(sample, ensure_ascii=False, indent=2)}")

        return dict(task_samples)

if __name__ == "__main__":
    # 初始化
    train_file = "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/train_only_dna.jsonl"
    test_file = "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/dev_only_dna.jsonl"
    inspector = JsonlDatasetInspector(train_file)
    # 统计样本总数
    total = inspector.count_lines()
    print(f"\n总样本数: {total}")
    # 查看前 5 条样本
    inspector.preview(num_lines=3)
    # 统计各个 task 的样本分布
    task_stats = inspector.count_tasks(visualize=True, save_path="train_task_pie.png")
    # 按每个任务打印最多 3 条样本
    # inspector.visualize_per_task(num_samples_per_task=1)


