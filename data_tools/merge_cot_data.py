#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并两个 parquet 文件（列名相同），路径写死在脚本里
"""

import pandas as pd
from pathlib import Path

# /share/org/YZWL/yzwl_lijz/dataset_res/omics_data/CotExp/woCOT/Enhancer-Promoter/val_Enhancer-Promoter_task.parquet 
# /share/org/YZWL/yzwl_lijz/dataset_res/omics_data/CotExp/COT/Enhancer-Promoter/

# 固定路径
Kind = "Enhancer-Promoter"
FILE1 = Path(f"/share/org/YZWL/yzwl_lijz/dataset_res/omics_data/CotExp/woCOT/{Kind}/train_{Kind}_task.parquet")
FILE2 = Path(f"/share/org/YZWL/yzwl_lijz/dataset_res/omics_data/CotExp/COT/{Kind}/cot.parquet")
OUT   = Path(f"/share/org/YZWL/yzwl_lijz/dataset_res/omics_data/CotExp/COT/{Kind}/train_{Kind}_cot_merged.parquet")


def main():
    # 读取
    df1 = pd.read_parquet(FILE1, engine="pyarrow")
    df2 = pd.read_parquet(FILE2, engine="pyarrow")

    # 合并（纵向拼接）
    df = pd.concat([df1, df2], ignore_index=True)

    # 保存
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, engine="pyarrow", index=False)

    # 打印统计
    print("✅ 合并完成")
    print(f"文件1: {FILE1} -> {len(df1)} 行")
    print(f"文件2: {FILE2} -> {len(df2)} 行")
    print(f"输出 : {OUT} -> {len(df)} 行")
    print(f"列名 : {list(df.columns)}")


if __name__ == "__main__":
    main()