# pylint: skip-file
import json
import os
import re
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def extract_answer(text: str) -> str:
    """
    从 decoded_output 中抽取 Answer: 后的内容
    规则：
      1. 匹配 "Answer:" 或 "Answer：" 之后直到换行或句尾
      2. 去掉首尾空格，并统一转小写
    """
    if not text:
        return ""
    pattern = r"(?<=Answer[:：])\s*(.*?)(?=\n|$)"
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip().lower()
    return ""


def load_json(path):
    """逐行读取 JSON Lines -> list[dict]"""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                samples.append(json.loads(line))
    return samples


def compute_metrics(y_true, y_pred, labels=None):
    acc = accuracy_score(y_true, y_pred)
    # [None, 'micro', 'macro', 'weighted']
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0, labels=labels
    )
    return {"Accuracy": acc, "Precision": precision, "Recall": recall, "F1-Score": f1}


def plot_cm(y_true, y_pred, labels, save_path=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.ylabel("Ground Truth")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


def main():
    json_path = "/fs-computility/ai4agr/lijinzhe/code/BioMLLM_V2/res/inference/inference_Qwen3_4B_NT_sft_0729_KEGG_checkpoint-700.json"
    if not os.path.exists(json_path):
        print(f"文件不存在：{json_path}")
        sys.exit(1)

    data = load_json(json_path)

    y_true, y_pred = [], []
    for item in data:
        decoded_output = item.get("decoded_output", "")
        gt_label = str(item.get("gt_label", "")).strip().lower()
        pred_label = extract_answer(decoded_output)

        def fuzzy_match(a: str, b: str) -> bool:
            a, b = a.strip(), b.strip()
            return (bool(a) and a != b and a in b) or (bool(b) and b != a and b in a)

        if fuzzy_match(pred_label, gt_label):
            print(f"fuzzy match: {pred_label} -> {gt_label}")
            pred_label = gt_label

        y_true.append(gt_label)
        y_pred.append(pred_label)

    labels = sorted(set(y_true))
    # print(y_true)
    # print(y_pred)
    metrics = compute_metrics(y_true, y_pred, labels=labels)

    print("=" * 50)
    for k, v in metrics.items():
        print(f"{k:<12}: {v:.4f}")
    print("=" * 50)
    print("标签列表:", labels)

    # 可选：保存混淆矩阵
    # plot_cm(y_true, y_pred, labels, save_path="confusion_matrix.png")


if __name__ == "__main__":
    main()
