#! /bin/bash
# usage:  ./inference_nt_lora.sh  /path/to/checkpoint  /path/to/1B_mini_pack1.jsonl
set -euo pipefail

# 1. 接收外部参数
CHECKPOINT_DIR="${1:?Usage: $0 <checkpoint-dir> <output.jsonl>}"
OUTPUT_JSONL="${2:?Usage: $0 <checkpoint-dir> <output.jsonl>}"

# 2. 检查路径是否存在
[[ -d $CHECKPOINT_DIR ]] || { echo "ERROR: checkpoint dir not found: $CHECKPOINT_DIR"; exit 1; }

# 3. 其余固定选项
readonly TEXT_MODEL="/mnt/shared-storage-user/ai4agr-share/lijinzhe/PreModel/Qwen3-1.7B"
readonly DNA_MODEL="/mnt/shared-storage-user/ai4agr-share/lijinzhe/PreModel/nucleotide-transformer/"
readonly PROTEIN_MODEL="/mnt/shared-storage-user/ai4agr-share/lijinzhe/PreModel/esm2_t33_650M_UR50D"
readonly DATASET="/mnt/shared-storage-user/ai4agr-share/lijinzhe/data/BioMLLM/train-val-test/test_all_task_standard.parquet"

OPTIONS=(
    --text-model-path            "$TEXT_MODEL"
    --dna-rna-model-path         "$DNA_MODEL"
    --dna-rna-k-tokens           1024
    --protein-model-path         "$PROTEIN_MODEL"
    --protein-k-tokens           1024
    --trained-model-path         "$CHECKPOINT_DIR"
    --dataset-path               "$DATASET"
    --max-length                 3072
    --batch-size                 32
    --temperature                0.8
    --top-p                      0.95
    --repetition-penalty         1.1
    --seed                       42
    --json-file                  "$OUTPUT_JSONL"
)

export TRANSFORMERS_VERBOSITY=info
echo "Running inference with LoRA model from $CHECKPOINT_DIR"
echo "Results will be saved to $OUTPUT_JSONL"
echo "Script will be run in 3 seconds later.."
sleep 3
python3 src/inference_lora.py "${OPTIONS[@]}"