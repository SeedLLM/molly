enable_list="multimodal model.model.embed_tokens model.model.layers model.lm_head"
experiment_name="Qwen3_1.7B_Omics_sft_1003_all_task_exp1"
output_path="${experiment_name}"

options="--experiment-name $experiment_name \
--output_dir $output_path \
--text-model-path /mnt/shared-storage-user/ai4agr-share/lijinzhe/PreModel/Qwen3-1.7B \
--dna-rna-model-path /mnt/shared-storage-user/ai4agr-share/lijinzhe/PreModel/nucleotide-transformer/  \
--dna-rna-k-tokens 1024 \
--protein-model-path /mnt/shared-storage-user/ai4agr-share/lijinzhe/PreModel/esm2_t33_650M_UR50D/ \
--protein-k-tokens 1024 \
--device cuda \
--freeze-bio \
--train-dataset-path /mnt/shared-storage-user/ai4agr-share/lijinzhe/data/BioMLLM/train-val-test/train_all_task.parquet \
--eval-dataset-path /mnt/shared-storage-user/ai4agr-share/lijinzhe/data/BioMLLM/train-val-test/dev_all_task.parquet \
--max-len 3072 \
--max-src-len 3072 \
--eval-max-len 3072 \
--eval-max-src-len 3072 \
--mode sft \
--per_device_train_batch_size 6 \
--per_device_eval_batch_size 4 \
--read-nums 8192 \
--eval-read-nums 8192 \
--num_train_epochs 2 \
--learning_rate 3e-5 \
--bf16 \
--enable-list $enable_list \
--save_strategy steps \
--save_steps 50000 \
--eval_steps 25000 \
--eval_strategy steps \
--logging_strategy steps \
--logging_steps 20 \
--save_trainable False \
--save-total-limit 500 \
--warmup_ratio 0.1 \
--early-stopping-patience 1000000000 \
--gradient-accumulation-steps 2 \
--save_only_model \
--attn_impl sdpa \
--use_liger False \
" 
# --load_best_model_at_end \
# --save_safetensors \
# --greater_is_better \
# --use-lora
# --load-pretrained \

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
deepspeed \
--include localhost:0 \
src/train.py \
--deepspeed_config src/configs/ds_z0_config.json \
$options
