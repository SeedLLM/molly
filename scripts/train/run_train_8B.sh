enable_list="multimodal model.model.embed_tokens model.model.layers model.lm_head"
experiment_name="Qwen3_1B_Omics_sft_0814_10_task_test"
output_path="/share/org/YZWL/yzwl_lijz/dataset_res/res_model/${experiment_name}"

# export NCCL_DEBUG=INFO
# export NCCL_P2P_DISABLE=1    
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_LEVEL=NVL 
# export NCCL_LL_THRESHOLD=0


options="--experiment-name $experiment_name \
--output_dir $output_path \
--text-model-path /share/org/YZWL/yzwl_lijz/base_llm/Qwen3-1B \
--dna-rna-model-path /share/org/YZWL/yzwl_lijz/base_llm/nucleotide-transformer/  \
--dna-rna-k-tokens 1024 \
--protein-model-path /share/org/YZWL/yzwl_lijz/base_llm/esm2_t33_650M_UR50D/ \
--protein-k-tokens 1024 \
--device cuda \
--freeze-bio \
--train-dataset-path /share/org/YZWL/yzwl_lijz/dataset_res/omics_data/TargetTask0808/train_10_task.parquet \
--eval-dataset-path /share/org/YZWL/yzwl_lijz/dataset_res/omics_data/TargetTask0808/val_10_task.parquet \
--max-len 3072 \
--max-src-len 3072 \
--eval-max-len 3072 \
--eval-max-src-len 3072 \
--mode sft \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 4 \
--read-nums 50000 \
--eval-read-nums 51276 \
--num_train_epochs 5 \
--learning_rate 3e-5 \
--bf16 \
--enable-list $enable_list \
--save_strategy steps \
--save_steps 64000 \
--eval_steps 64000 \
--eval_strategy steps \
--logging_strategy steps \
--logging_steps 20 \
--save_trainable False \
--save-total-limit 50 \
--swanlab \
--swanlab-mode local \
--swanlab-team BioMLLM_report \
--swanlab-project BioMLLM \
--report_to swanlab \
--warmup_ratio 0.1 \
--early-stopping-patience 1000000000 \
--gradient-accumulation-steps 2 \
--save_only_model \
--attn_impl flash_attention_2 \
" 
# --load_best_model_at_end \
# --save_safetensors \
# --greater_is_better \
# --use-lora
# --load-pretrained \

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
deepspeed --include localhost:0,1,2,3,4,5,6,7 \
src/train.py \
--deepspeed_config src/configs/ds_z2_config.json \
$options