enable_list="multimodal model.model.embed_tokens model.model.layers model.lm_head"
experiment_name="Qwen3_1.7B_Omics_sft_1003_all_task_exp1_test"
output_path="{experiment_name}"

# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_TIMEOUT=3600

options="--experiment-name $experiment_name \
--output_dir $output_path \
--text-model-path /mnt/shared-storage-user/ai4agr-share/lijinzhe/PreModel/Qwen3-1.7B \
--dna-rna-model-path /mnt/shared-storage-user/ai4agr-share/lijinzhe/PreModel/nucleotide-transformer/  \
--dna-rna-k-tokens 1024 \
--protein-model-path /mnt/shared-storage-user/ai4agr-share/lijinzhe/PreModel/esm2_t33_650M_UR50D/ \
--protein-k-tokens 1024 \
--device cuda \
--train-mlp \
--train-llm \
--train-dataset-path /mnt/shared-storage-user/ai4agr-share/lijinzhe/data/BioMLLM/train-val-test/train_all_task_standard.parquet \
--eval-dataset-path /mnt/shared-storage-user/ai4agr-share/lijinzhe/data/BioMLLM/train-val-test/dev_all_task_standard.parquet \
--max-len 3072 \
--max-src-len 3072 \
--eval-max-len 3072 \
--eval-max-src-len 3072 \
--mode sft \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 2 \
--read-nums 10240000000 \
--eval-read-nums 1024000000 \
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
--gradient-accumulation-steps 4 \
--save_only_model \
--attn_impl flash_attention_2 \
--use_liger True \
--swanlab \
--swanlab-mode local \
--swanlab-team BioMLLM_report \
--swanlab-project BioMLLM \
--seed 42 \
"
# --load_best_model_at_end \
# --save_safetensors \
# --greater_is_better \
# --use-lora
# --load-pretrained \

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
deepspeed \
--include localhost:0,1,2,3,4,5,6,7 \
src/train.py \
--deepspeed_config src/configs/ds_z0_config.json \
$options
