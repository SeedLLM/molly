enable_list="multimodal model.model.embed_tokens model.model.layers model.lm_head"

# 去掉路径，去掉最后一个 .xxx 后缀
experiment=$(basename "$0")
output_path=${experiment%.*}

echo "输出路径" "$output_path"

sleep 3
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600
export NCCL_ASYNC_ERROR_HANDLING=1   # 超时不会直接 crash，可日志报警
export NCCL_DEBUG=INFO               # 方便定位

# rlaunch 需要 96cpu
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

options="--experiment-name $experiment \
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
--max-len 8192 \
--eval-max-len 8192 \
--mode sft \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 1 \
--read-nums 12800000 \
--eval-read-nums 12800000 \
--num_train_epochs 1 \
--learning_rate 3e-5 \
--bf16 \
--enable-list $enable_list \
--save_strategy steps \
--save_steps 3000 \
--eval_steps 25000 \
--eval_strategy steps \
--logging_strategy steps \
--logging_steps 2 \
--save_trainable False \
--save-total-limit 500 \
--warmup_ratio 0.1 \
--early-stopping-patience 1000000000 \
--gradient-accumulation-steps 4 \
--save_only_model \
--attn_impl flash_attention_2 \
--swanlab \
--swanlab-mode local \
--swanlab-team BioMLLM_report \
--swanlab-project BioMLLM \
--seed 42 \
--use_dem_sft False \
--use_liger True \
--packing True
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
