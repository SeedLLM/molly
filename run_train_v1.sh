enable_list="multimodal model.model.embed_tokens model.model.layers model.lm_head"

options="--experiment-name Qwen_DNABERT_sft_exp_2_8b_ \
--test-code \
--swanlab \
--swanlab-team BioMLLM_report \
--swanlab-project BioMLLM_wandb \
--text-model-path /fs-computility/ai4agr/lijinzhe/basemodel/Qwen3-0.6B \
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
--eval-batch-size-per-gpu 8 \
--read-nums 50000 \
--eval-read-nums 500 \
--epochs 1 \
--lr 1.0e-5 \
--ds-config-path /fs-computility/ai4agr/lijinzhe/code/BioMLLM_V2/zero2_config.json \
--enable-list $enable_list \
--output-path /fs-computility/ai4agr/lijinzhe/res_data_model/0630_qwen3_50k \
--save-interval 50 \
--eval-interval 100 \
--show_avg_loss_step 20 \
--save_trainable True \
" 

deepspeed --include localhost:0,1 \
train_v1.py \
$options