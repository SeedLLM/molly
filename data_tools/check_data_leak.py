# pylint: skip-file
import re

import pandas as pd

# # 加载 parquet 文件
# val_df = pd.read_parquet('/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/Eval/dna_rna_8k_eval.parquet')
# train_df = pd.read_parquet('/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/rewritten_8k/train_dna_rna.parquet')

# # 检查 val_df 中的 input 是否出现在 train_df 中
# duplicates_in_train = val_df[val_df['input'].isin(train_df['input'])]

# # 显示 val_df 中重复出现在 train_df 的行及其数量
# if not duplicates_in_train.empty:
#     print(f"发现 {len(duplicates_in_train)} 条 val_df 中的 input 在 train_df 中出现：")
#     print(duplicates_in_train)
# else:
#     print("没有发现 val_df 中的 input 在 train_df 中出现。")


# 加载 parquet 文件
df = pd.read_parquet(
    "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/TargetTask0729/protein/train_target_task_protein.parquet"
)

# 检查 input 中是否符合规则
def check_input_format(input_text):
    # 提取 <rna> 和 <protein> 标签之间的内容
    rna_pattern = r"<rna>\s*([ACGTNacgtn]+)\s*<rna>"
    protein_pattern = r"<protein>\s*([ACDEFGHIKLMNPQRSTVWYBXZOU]+)\s*<protein>"

    rna_matches = re.findall(rna_pattern, input_text)
    protein_matches = re.findall(protein_pattern, input_text)

    # 如果有多个 <rna> 标签，且符合规则
    if len(rna_matches) > 1:
        return "rna-rna"

    # 如果有多个 <protein> 标签，且符合规则
    if len(protein_matches) > 1:
        return "protein-protein"

    # 如果是 <protein> 在前，<rna> 在后
    if len(protein_matches) == 1 and len(rna_matches) == 1:
        protein_start = input_text.find("<protein>")
        rna_start = input_text.find("<rna>")
        if protein_start < rna_start:
            return "protein-rna"
        else:
            return "rna-protein"

    # 检查单独的 <rna> 或 <protein>
    if len(rna_matches) == 1:
        return "rna"
    elif len(protein_matches) == 1:
        return "protein"

    return "invalid"


# 检查 `input` 列
df["input_check"] = df["input"].apply(check_input_format)

# 根据 `input` 检查 `kind` 是否匹配
def check_kind(row):
    input_check = row["input_check"]
    kind = row["kind"]

    if input_check == "protein-rna" and kind != "protein-rna":
        return "Mismatch"
    elif input_check == "rna-protein" and kind != "rna-protein":
        return "Mismatch"
    elif input_check == "rna-rna" and kind != "rna-rna":
        return "Mismatch"
    elif input_check == "protein-protein" and kind != "protein-protein":
        return "Mismatch"
    elif input_check == "rna" and kind != "rna":
        return "Mismatch"
    elif input_check == "protein" and kind != "protein":
        return "Mismatch"
    elif input_check == "invalid" and kind != "invalid":
        return "Mismatch"
    return "Match"


# 检查 `kind` 是否与 `input` 匹配
df["kind_check"] = df.apply(check_kind, axis=1)

# 输出结果
print(df[["input", "kind", "input_check", "kind_check"]])

# 只输出不匹配的部分
print("\nMismatch entries:")
print(df[df["kind_check"] == "Mismatch"])
