output_path="/tos-bjml-ai4agr/lijinzhe/BioMLLM/RES_Embed/${experiment_name}"
text_model_path="/tos-bjml-ai4agr/lijinzhe/BioMLLM/Qwen3-Embedding-4B"
dna_rna_model_path="/tos-bjml-ai4agr/lijinzhe/BioModel/nucleotide-transformer/"
protein_model_path="/tos-bjml-ai4agr/lijinzhe/BioMLLM/esm2_t33_650M_UR50D/"

dataset_path="/tos-bjml-ai4agr/lijinzhe/move2H/BioMLLM/Data/BioMLLM/Cluster/train_target_task_0819.parquet"

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
--dataloader-num-workers 8 \
--read-nums 20000000 \
--mode sft \
--skip-eval \
"


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python src/embed_text.py $options