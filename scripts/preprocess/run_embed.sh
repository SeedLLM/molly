output_path="/share/org/YZWL/yzwl_chenzh/work_dir/data/output/${experiment_name}"
text_model_path="/share/appspace_data/shared_groups/yzwl_chenzh_ChenZihong_Share/base_llm/Qwen3-Embedding-4B"
dna_rna_model_path="/share/appspace_data/shared_groups/yzwl_chenzh_ChenZihong_Share/base_llm/nucleotide-transformer/"
protein_model_path="/share/appspace_data/shared_groups/yzwl_chenzh_ChenZihong_Share/base_llm/esm2_t33_650M_UR50D/"

dataset_path="/share/org/YZWL/yzwl_chenzh/work_dir/data/biollm/train_target_task_0819.parquet"

mkdir -p "$output_path"

options="--text-model-path $text_model_path \
--dna-rna-model-path $dna_rna_model_path \
--dna-rna-k-tokens 1024 \
--protein-model-path $protein_model_path \
--protein-k-tokens 1024 \
--device cuda \
--train-dataset-path $dataset_path \
--output_dir $output_path \
--max-len 3072 \
--max-src-len 3072 \
--dataloader-num-workers 32 \
--read-nums 2000000 \
--mode sft \
--skip-eval \
"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python src/embed_text.py $options