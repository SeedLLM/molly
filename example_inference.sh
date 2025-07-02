#! /bin/bash

options="--batch_size 8 \
    --dataset_path /tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/dev_only_dna.jsonl \
    --text_model_path /fs-computility/ai4agr/lijinzhe/basemodel/Qwen3-0.6B \
    --bio_model_path /tos-bjml-ai4agr/lijinzhe/BioMLLM/DNABERT-2-117M \
    --trained_model_path /fs-computility/ai4agr/lijinzhe/res_data_model/0630_qwen3_200k/checkpoint-500/pytorch_model.bin \
    --multimodal_k_tokens 128 \
    --max_length 1024 \
    --temperature 0.8 \
    --top_p 0.95 \
    --max_samples 100 \
    --output_path /fs-computility/ai4agr/lijinzhe/res_data_model/inference_results \
"

export TRANSFORMERS_VERBOSITY=info
python new_inference.py $options