enable_list="multimodal model.model.embed_tokens model.model.layers model.lm_head"
experiment_name="Qwen3_8B_Omics_sft_0811_solubility_task_test"
output_path="/tos-bjml-ai4agr/lijinzhe/BioMLLM/RES_Model/${experiment_name}"

options="--experiment-name $experiment_name \
--output_dir $output_path \
--text-model-path /tos-bjml-ai4agr/lijinzhe/BioMLLM/Qwen3-8B \
--dna-rna-model-path /tos-bjml-ai4agr/lijinzhe/BioModel/nucleotide-transformer/  \
--dna-rna-k-tokens 1 \
--protein-model-path /tos-bjml-ai4agr/lijinzhe/BioMLLM/esm2_t33_650M_UR50D/ \
--protein-k-tokens 1 \
--device cuda \
--load-pretrained \
--freeze-bio \
--train-dataset-path /tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/TargetTask0808/Solubility/train_solubility_task_0808.parquet \
--eval-dataset-path /tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/TargetTask0808/Solubility/val_solubility_task_0808.parquet \
--max-len 5 \
--max-src-len 5 \
--eval-max-len 5 \
--eval-max-src-len 5 \
--dataloader_num_workers 8 \
--mode sft \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 4 \
--read-nums 62478 \
--eval-read-nums 6942 \
--num_train_epochs 5 \
--learning_rate 3e-5 \
--bf16 \
--enable-list $enable_list \
--save_strategy steps \
--save_steps 500 \
--eval_steps 50 \
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
--early-stopping-patience 1000000000 \
--gradient-accumulation-steps 1 \
" 
# --load_best_model_at_end \
# --save_safetensors \
# --greater_is_better \
# --use-lora

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
deepspeed --include localhost:0,1,2,3 \
src/train.py \
--deepspeed_config src/configs/ds_z2_config.json \
$options