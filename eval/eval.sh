#!/usr/bin/env bash
set -euo pipefail
[[ $# -ne 2 ]] && { echo "用法: $0 <model_name> <input_file_path>"; exit 1; }
model_name=$1
input_file_path=$2

python3 eval.py \
  --model_name ${model_name} \
  --OMICS All \
  --input_file_path ${input_file_path}
