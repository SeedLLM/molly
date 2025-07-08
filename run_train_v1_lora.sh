enable_list="multimodal model.model.embed_tokens model.model.layers model.lm_head"

options="--experiment-name Qwen_DNABERT_sft_exp3_lora_4b_ \
--use-lora \
--swanlab \
--swanlab-team BioMLLM_report \
--swanlab-project BioMLLM_LoRA \
--text-model-path /tos-bjml-ai4agr/lijinzhe/BioMLLM/Qwen3-B \
--bio-model-path /tos-bjml-ai4agr/lijinzhe/BioMLLM/DNABERT-2-117M \
--multimodal-k-tokens 128 \
--device cuda \
--load-pretrained \
--freeze-dna-bert \
--train-dataset-path /tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/train_only_dna.jsonl \
--eval-dataset-path /tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/dev_only_dna.jsonl \
--max-len 1024 \
--max-src-len 1024 \
--eval-max-len 1024 \
--eval-max-src-len 1024 \
--mode sft \
--batch-size-per-gpu 4 \
--eval-batch-size-per-gpu 4 \
--read-nums 50000 \
--eval-read-nums 100 \
--epochs 1 \
--lr 1.0e-5 \
--ds-config-path /fs-computility/ai4agr/lijinzhe/code/BioMLLM_V2/zero2_config.json \
--enable-list $enable_list \
--output-path /fs-computility/ai4agr/lijinzhe/res_data_model/biomllm_res/0708_exp3_qwen3_4b_50k \
--save-interval 2500 \
--eval-interval 100 \
--show_avg_loss_step 10 \
--save_trainable False \
--save-total-limit 50 \
" 

deepspeed --include localhost:0,1,2,3 \
train_v1_lora.py \
$options