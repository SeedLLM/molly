#! /bin/bash

# 设置变量，方便切换模型
experiment_name="Qwen3_4B_NT_untrain"

options="--batch_size 16 \
    --dataset_path /tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/stage3_train_data/val_nt_dna_rna.parquet \
    --text_model_path /tos-bjml-ai4agr/lijinzhe/BioMLLM/Qwen3-4B \
    --bio_model_path /tos-bjml-ai4agr/lijinzhe/BioModel/nucleotide-transformer/ \
    --multimodal_k_tokens 128 \
    --max_length 1024 \
    --temperature 0.8 \
    --top_p 0.95 \
    --output_path /fs-computility/ai4agr/lijinzhe/res_data_model/biomllm_res/inference_results/${experiment_name} \
    --json_file /fs-computility/ai4agr/lijinzhe/code/BioMLLM_V2/data_tools/sample/sample_inference_${experiment_name}.json
"

export TRANSFORMERS_VERBOSITY=info
python inference_untrain.py $options