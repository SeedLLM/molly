#! /bin/bash

# 设置变量，方便切换模型
experiment_name="Qwen3_4B_NT_sft_exp1_0710"
MODEL_DIR="/fs-computility/ai4agr/lijinzhe/res_data_model/biomllm_res/${experiment_name}"
CHECKPOINT="checkpoint-1500"

options="--batch_size 8 \
    --dataset_path /fs-computility/ai4agr/lijinzhe/code/BioMLLM_V2/src/utils/balanced_dna_test_dataset.jsonl \
    --text_model_path /tos-bjml-ai4agr/lijinzhe/BioMLLM/Qwen3-4B \
    --bio_model_path /tos-bjml-ai4agr/lijinzhe/BioModel/nucleotide-transformer/ \
    --trained_model_path ${MODEL_DIR}/${CHECKPOINT} \
    --multimodal_k_tokens 128 \
    --use_lora \
    --max_length 1024 \
    --temperature 0.8 \
    --top_p 0.95 \
    --output_path /fs-computility/ai4agr/lijinzhe/res_data_model/biomllm_res/inference_results/${experiment_name} \
"

export TRANSFORMERS_VERBOSITY=info
echo "Running inference with LoRA model from ${MODEL_DIR}/${CHECKPOINT}"
python inference_nt_lora.py $options