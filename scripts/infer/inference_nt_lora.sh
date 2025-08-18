#! /bin/bash

# 设置变量，方便切换模型
experiment_name="Qwen3_4B_Omics_sft_protein_task_exp2"
MODEL_DIR="/tos-bjml-ai4agr/lijinzhe/BioMLLM/RES_Model/${experiment_name}"
CHECKPOINT="checkpoint-15690"

options="--text-model-path /tos-bjml-ai4agr/lijinzhe/BioMLLM/Qwen3-4B \
    --dna-rna-model-path /tos-bjml-ai4agr/lijinzhe/BioModel/nucleotide-transformer/ \
    --dna-rna-k-tokens 1024 \
    --protein-model-path /tos-bjml-ai4agr/lijinzhe/BioMLLM/esm2_t33_650M_UR50D/ \
    --protein-k-tokens 1024 \
    --trained-model-path ${MODEL_DIR}/${CHECKPOINT} \
    --dataset-path /tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/TargetTask0808/val_10_task.parquet \
    --max-length 3072 \
    --batch-size 16 \
    --temperature 0.8 \
    --top-p 0.95 \
    --repetition-penalty 1.1 \
    --json-file /fs-computility/ai4agr/lijinzhe/code/BioMLLM_V2/res/inference/Qwen3_4B_Omics_sft_protein_task_exp2_5epoch/inference_${experiment_name}_${CHECKPOINT}.json
"

# --use_lora \
export TRANSFORMERS_VERBOSITY=info
echo "Running inference with LoRA model from ${MODEL_DIR}/${CHECKPOINT}"
python src/inference_lora.py $options