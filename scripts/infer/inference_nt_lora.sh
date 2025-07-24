#! /bin/bash

# 设置变量，方便切换模型
experiment_name="Qwen3_1.7B_NT_sft_0723_exp1"
MODEL_DIR="/fs-computility/ai4agr/lijinzhe/res_data_model/biomllm_res/${experiment_name}"
CHECKPOINT="checkpoint-7236"

options="--batch_size 16 \
    --dataset_path /tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/rewritten_8k/valid_dna_rna.parquet \
    --text_model_path /tos-bjml-ai4agr/lijinzhe/BioMLLM/Qwen3-1.7B \
    --bio_model_path /tos-bjml-ai4agr/lijinzhe/BioModel/nucleotide-transformer/ \
    --trained_model_path ${MODEL_DIR}/${CHECKPOINT} \
    --multimodal_k_tokens 128 \
    --use_lora \
    --max_length 1024 \
    --temperature 0.8 \
    --top_p 0.95 \
    --output_path /fs-computility/ai4agr/lijinzhe/res_data_model/biomllm_res/inference_results/${experiment_name} \
    --json_file /fs-computility/ai4agr/lijinzhe/code/BioMLLM_V2/data_tools/sample/inference_${experiment_name}.json
"

export TRANSFORMERS_VERBOSITY=info
echo "Running inference with LoRA model from ${MODEL_DIR}/${CHECKPOINT}"
python inference_nt_lora.py $options