enable_list="multimodal model.model.embed_tokens model.model.layers model.lm_head"
experiment_name="Qwen3_4B_Omics_sft_0805_exp1_lora"
output_path="/tos-bjml-ai4agr/lijinzhe/BioMLLM/RES_Model/${experiment_name}"

options="--experiment-name $experiment_name \
--output_dir $output_path \
--text-model-path /tos-bjml-ai4agr/lijinzhe/BioMLLM/Qwen3-4B \
--dna-rna-model-path /tos-bjml-ai4agr/lijinzhe/BioModel/nucleotide-transformer/  \
--dna-rna-k-tokens 128 \
--protein-model-path /tos-bjml-ai4agr/lijinzhe/BioMLLM/esm2_t33_650M_UR50D/ \
--protein-k-tokens 128 \
--device cuda \
--load-pretrained \
--freeze-bio \
--train-dataset-path /tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/TargetTask0729/training/train_wo_s3_all.parquet \
--eval-dataset-path /tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/TargetTask0729/training/val_wo_s3_all.parquet \
--max-len 1024 \
--max-src-len 1024 \
--eval-max-len 1024 \
--eval-max-src-len 1024 \
--dataloader_num_workers 4 \
--mode sft \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 4 \
--read-nums 600083 \
--eval-read-nums 37707 \
--num_train_epochs 2 \
--learning_rate 3e-5 \
--bf16 \
--enable-list $enable_list \
--save_strategy steps \
--save_steps 1000 \
--eval_steps 500 \
--eval_strategy steps \
--logging_strategy steps \
--logging_steps 20 \
--save_trainable False \
--save-total-limit 50 \
--swanlab \
--swanlab-team BioMLLM_report \
--swanlab-project BioMLLM \
--report_to swanlab \
--warmup_ratio 0.1 \
--use-lora \
" 
# --load_best_model_at_end \
# --save_safetensors \
# --greater_is_better \
# --use-lora

deepspeed --include localhost:0,1,2,3,4,5,6,7 \
src/train.py \
--deepspeed_config src/configs/zero3_config.json \
$options