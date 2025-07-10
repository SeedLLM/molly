enable_list="multimodal model.model.embed_tokens model.model.layers model.lm_head"
experiment_name="Qwen_NT_sft_exp_20250710_all"
output_path="/fs-computility/ai4agr/mazhe/BioMLLM_V2/output/${experiment_name}"
tb_log_dir="/fs-computility/ai4agr/mazhe/BioMLLM_V2/tensorboard/${experiment_name}"

options="--experiment-name $experiment_name \
--test-code \
--text-model-path /tos-bjml-ai4agr/mazhe/model_dir/Qwen3-0.6B \
--bio-model-path /tos-bjml-ai4agr/lijinzhe/BioModel/nucleotide-transformer/  \
--multimodal-k-tokens 128 \
--device cuda \
--load-pretrained \
--freeze-nt \
--train-dataset-path /tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/train_only_dna.jsonl \
--eval-dataset-path /tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/dev_only_dna.jsonl \
--max-len 1024 \
--max-src-len 1024 \
--eval-max-len 1024 \
--eval-max-src-len 1024 \
--mode sft \
--batch-size-per-gpu 2 \
--eval-batch-size-per-gpu 4 \
--read-nums 50000 \
--eval-read-nums 500 \
--epochs 1 \
--lr 1.0e-5 \
--ds-config-path /fs-computility/ai4agr/mazhe/BioMLLM_V2/mazhe_utils/zero2_config.json \
--enable-list $enable_list \
--output-path $output_path \
--save-interval 50 \
--eval-interval 100 \
--show_avg_loss_step 20 \
--save_trainable True \
--tensorboard \
--tb-log-dir $tb_log_dir \
" 

deepspeed --include localhost:0,1,2,3,4,5,6,7 \
train_v1_nt.py \
$options

"--swanlab \
--swanlab-team BioMLLM_report \
--swanlab-project BioMLLM_wandb \"