enable_list="multimodal model.model.embed_tokens model.model.layers model.lm_head"
experiment_name="Qwen3_0.6B_NT_sft_exp1_0718"
output_path="/fs-computility/ai4agr/chenzihong/res_data_model/biomllm_res/${experiment_name}"

options="--experiment-name $experiment_name \
--output_dir $output_path \
--use-lora \
--text-model-path /tos-bjml-ai4agr/lijinzhe/BioMLLM/Qwen3-0.6B \
--bio-model-path /tos-bjml-ai4agr/lijinzhe/BioModel/nucleotide-transformer/  \
--multimodal-k-tokens 128 \
--device cuda \
--load-pretrained \
--load_best_model_at_end True \
--freeze-nt \
--train-dataset-path /tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/stage3_train_data/stage3_train_nt_dna_rna.parquet \
--eval-dataset-path /tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/stage3_train_data/val_nt_dna_rna.parquet \
--max-len 1024 \
--max-src-len 1024 \
--eval-max-len 1024 \
--eval-max-src-len 1024 \
--dataloader_num_workers 4 \
--mode sft \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 4 \
--read-nums 16500 \
--eval-read-nums 500 \
--num_train_epochs 3 \
--learning_rate 1.0e-5 \
--bf16 \
--enable-list $enable_list \
--save_strategy steps \
--save_steps 1000 \
--eval_strategy steps \
--eval_steps 100 \
--logging_strategy steps \
--logging_steps 20 \
--save_trainable False \
--save-total-limit 50 \
--swanlab \
--swanlab-team BioMLLM_report \
--swanlab-project BioMLLM_NT \
--report_to swanlab \
--greater_is_better False \
--warmup_ratio 0.1 \
" 

deepspeed --include localhost:0,1 \
src/train_lora.py \
--deepspeed_config /tos-bjml-ai4agr/chenzihong/code/BioMLLM_V2/src/configs/zero2_config.json \
$options