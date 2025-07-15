enable_list="multimodal model.model.embed_tokens model.model.layers model.lm_head"
experiment_name="Qwen3_1.7B_NT_sft_exp1_0715"
output_path="/fs-computility/ai4agr/lijinzhe/res_data_model/biomllm_res/${experiment_name}"
tb_log_dir="/fs-computility/ai4agr/mazhe/BioMLLM_V2/tensorboard/${experiment_name}"

options="--experiment-name $experiment_name \
--use-lora \
--text-model-path /tos-bjml-ai4agr/lijinzhe/BioMLLM/Qwen3-1.7B \
--bio-model-path /tos-bjml-ai4agr/lijinzhe/BioModel/nucleotide-transformer/  \
--multimodal-k-tokens 128 \
--device cuda \
--load-pretrained \
--freeze-nt \
--train-dataset-path /tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/stage3_train_data/stage3_train_nt.parquet \
--eval-dataset-path /tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/stage3_train_data/val_nt.parquet \
--max-len 1024 \
--max-src-len 1024 \
--eval-max-len 1024 \
--eval-max-src-len 1024 \
--mode sft \
--batch-size-per-gpu 2 \
--eval-batch-size-per-gpu 4 \
--read-nums 16500 \
--eval-read-nums 500 \
--epochs 3 \
--lr 1.0e-5 \
--ds-config-path /fs-computility/ai4agr/lijinzhe/code/BioMLLM_V2/zero2_config.json \
--enable-list $enable_list \
--output-path $output_path \
--save-interval 500 \
--eval-interval 100 \
--show_avg_loss_step 20 \
--save_trainable False \
--save-total-limit 50 \
--swanlab \
--swanlab-team BioMLLM_report \
--swanlab-project BioMLLM_NT \
" 

deepspeed --include localhost:0,1,2,3 \
train_v2_nt_lora.py \
$options