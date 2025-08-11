# pylint: skip-file
import pandas as pd

# 读取两个 Parquet 文件
# file1 = "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/TargetTask0729/dev_target_task.parquet"
file1 = "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/TargetTask0808/train_10_task.parquet"
# file2 = "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/TargetTask0729/protein/val_target_task_protein.parquet"
file2 = "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/TargetTask0808/all2stage3/stage3_target_task_0808.parquet"

# 使用 pandas 读取 Parquet 文件
df1 = pd.read_parquet(file1)
df2 = pd.read_parquet(file2)

# 合并两个 DataFrame（按行合并）
merged_df = pd.concat([df1, df2], ignore_index=True)

# 将合并后的 DataFrame 保存为新的 Parquet 文件
output_file = "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/TargetTask0808/train_10_task_wos3.parquet"
# output_file = "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/TargetTask0729/training/val_wo_s3_all.parquet"
merged_df.to_parquet(output_file)

print(f"Files merged successfully and saved to {output_file}")
