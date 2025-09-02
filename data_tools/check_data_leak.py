import pandas as pd

# ===== 请在这里填写两个 Parquet 文件的绝对或相对路径 =====
file1_path = '/share/org/YZWL/yzwl_lijz/dataset_res/omics_data/CotExp/COT/TF-M/train_TF-M_cot_merged.parquet'
file2_path = '/share/org/YZWL/yzwl_lijz/dataset_res/omics_data/CotExp/COT/TF-M/val_TF-M_task.parquet'
# ======================================================

# 读取两个文件
df1 = pd.read_parquet(file1_path, columns=['input'])
df2 = pd.read_parquet(file2_path, columns=['input'])

# 去除可能存在的空值
inputs1 = set(df1['input'].dropna())
inputs2 = set(df2['input'].dropna())

# 计算交集
common = inputs1 & inputs2

# 输出结果
if common:
    print("发现 input 完全相同的行！")
    print("共有重复 input 数量：", len(common))
    # 如需查看具体内容，可取消下一行注释
    print("重复 input 示例（前10个）：", list(common)[:10])
else:
    print("两个文件中 input 列没有重复值。")