#! /bin/bash

# 设置变量，方便切换模型
experiment_name="MOLLM-1.7B"
MODEL_DIR="/mnt/shared-storage-user/ai4agr-share/lijinzhe/TaskRes/BioMLLM/checkpoint/${experiment_name}"
CHECKPOINT="checkpoint-40370"
TIME="1"

options="--text-model-path /mnt/shared-storage-user/ai4agr-share/lijinzhe/PreModel/Qwen3-1.7B \
    --dna-rna-model-path /mnt/shared-storage-user/ai4agr-share/lijinzhe/PreModel/nucleotide-transformer/ \
    --dna-rna-k-tokens 1024 \
    --protein-model-path /mnt/shared-storage-user/ai4agr-share/lijinzhe/PreModel/esm2_t33_650M_UR50D/ \
    --protein-k-tokens 1024 \
    --trained-model-path ${MODEL_DIR} \
    --dataset-path /mnt/shared-storage-user/ai4agr-share/lijinzhe/data/BioMLLM/train-val-test/test_all_task.parquet \
    --max-length 3072 \
    --batch-size 16 \
    --temperature 0.8 \
    --top-p 0.95 \
    --repetition-penalty 1.1 \
    --seed 42 \
    --json-file /mnt/shared-storage-user/ai4agr-share/lijinzhe/TaskRes/MOLLM/InferRes/MOLLM-1.7B_time${TIME}/inference_${experiment_name}_${CHECKPOINT}.json
"

# --use_lora \
export TRANSFORMERS_VERBOSITY=info
echo "Running inference with LoRA model from ${MODEL_DIR}/${CHECKPOINT}"
python src/inference_lora.py $options