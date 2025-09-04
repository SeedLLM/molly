#! /bin/bash

# 设置变量，方便切换模型
experiment_name="Qwen3_4B_Omics_sft_0828_Solubility_cot_task_exp2"
MODEL_DIR="/share/org/YZWL/yzwl_lijz/dataset_res/res_model/${experiment_name}"
CHECKPOINT="checkpoint-40370"
TIME="3"

options="--text-model-path /share/org/YZWL/yzwl_lijz/base_llm/Qwen3-4B \
    --dna-rna-model-path /share/org/YZWL/yzwl_lijz/base_llm/nucleotide-transformer/ \
    --dna-rna-k-tokens 1024 \
    --protein-model-path /share/org/YZWL/yzwl_lijz/base_llm/esm2_t33_650M_UR50D/ \
    --protein-k-tokens 1024 \
    --trained-model-path ${MODEL_DIR}/${CHECKPOINT} \
    --dataset-path /share/org/YZWL/yzwl_lijz/dataset_res/omics_data/CotExp/COT/Solubility/val_Solubility_task.parquet \
    --max-length 5120 \
    --batch-size 16 \
    --temperature 0.8 \
    --top-p 0.95 \
    --repetition-penalty 1.1 \
    --seed 4200 \
    --json-file /share/org/YZWL/yzwl_lijz/src/BioMLLM_V2/res/infer/Qwen3_4B_Omics_sft_0828_Solubility_cot_task_exp2_time${TIME}/inference_${experiment_name}_${CHECKPOINT}.json
"

# --use_lora \
export TRANSFORMERS_VERBOSITY=info
echo "Running inference with LoRA model from ${MODEL_DIR}/${CHECKPOINT}"
python src/inference_lora.py $options