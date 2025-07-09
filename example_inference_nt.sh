#! /bin/bash

options="--batch_size 1 \
    --dataset_path /tos-bjml-ai4agr/mazhe/data/BioMLLM_1000/dev_only_dna.jsonl \
    --text_model_path /tos-bjml-ai4agr/mazhe/model_dir/Qwen3-0.6B \
    --bio_model_path /tos-bjml-ai4agr/lijinzhe/BioModel/nucleotide-transformer/ \
    --trained_model_path /fs-computility/ai4agr/mazhe/BioMLLM_V2/mazhe_utils/checkpoint-300/pytorch_model.bin \
    --multimodal_k_tokens 128 \
    --max_length 1024 \
    --temperature 0.8 \
    --top_p 0.95 \
    --max_samples 100 \
    --output_path /fs-computility/ai4agr/mazhe/BioMLLM_V2/mazhe_utils \
"

export TRANSFORMERS_VERBOSITY=info
python -m pdb new_inference_nt.py $options