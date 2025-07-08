#! /bin/bash

# 设置变量，方便切换模型
MODEL_DIR="/fs-computility/ai4agr/lijinzhe/res_data_model/biomllm_res/0707_qwen3_4b_1k"
CHECKPOINT="checkpoint-50"

options="--batch_size 8 \
    --dataset_path /tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/dev_only_dna.jsonl \
    --text_model_path /tos-bjml-ai4agr/lijinzhe/BioMLLM/Qwen3-4B \
    --bio_model_path /tos-bjml-ai4agr/lijinzhe/BioMLLM/DNABERT-2-117M \
    --trained_model_path ${MODEL_DIR}/${CHECKPOINT} \
    --use_lora \
    --multimodal_k_tokens 128 \
    --max_length 1024 \
    --temperature 0.8 \
    --top_p 0.95 \
    --max_samples 100 \
    --output_path /fs-computility/ai4agr/lijinzhe/res_data_model/biomllm_res/inference_results/0707_lora_results \
"

export TRANSFORMERS_VERBOSITY=info
echo "Running inference with LoRA model from ${MODEL_DIR}/${CHECKPOINT}"
python inference_lora.py $options