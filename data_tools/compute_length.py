import pandas as pd
import re

from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
)

# 安装所需的库
# pip install pyarrow

# 读取parquet文件
file_path = "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/TargetTask0808/train_10_task_wos3.parquet"
data = pd.read_parquet(file_path)

dna_rna_tokenizer = AutoTokenizer.from_pretrained("/tos-bjml-ai4agr/lijinzhe/BioModel/nucleotide-transformer/",
                                                    trust_remote_code=True)
protein_tokenizer = AutoTokenizer.from_pretrained("/tos-bjml-ai4agr/lijinzhe/BioMLLM/esm2_t33_650M_UR50D/",
                                                    trust_remote_code=True)

# 提取每行数据中的生物序列，并计算长度
def extract_lengths(sequence_text):
    # 匹配DNA, RNA和蛋白质序列的正则表达式
    dna_pattern = r"<dna>(.*?)<dna>"
    rna_pattern = r"<rna>(.*?)<rna>"
    protein_pattern = r"<protein>(.*?)<protein>"
    
    dna_length = []
    rna_length = []
    protein_length = []
    # 获取所有匹配的序列
    for seq in re.findall(dna_pattern, sequence_text):
        encoding = dna_rna_tokenizer(seq)
        dna_len = len(encoding['input_ids'])
        dna_length.append(dna_len)
        if dna_len == 0 or dna_len > 128:
            print(dna_len)
    for seq in re.findall(rna_pattern, sequence_text):
        encoding = dna_rna_tokenizer(seq)
        rna_len = len(encoding['input_ids'])
        rna_length.append(rna_len)
        if rna_len == 0 or rna_len > 128:
            print(rna_len)
    for seq in re.findall(protein_pattern, sequence_text):
        encoding = protein_tokenizer(seq)
        protein_len = len(encoding['input_ids'])
        protein_length.append(protein_len)
        # if protein_len == 0 or protein_len > 128:
        #     print(protein_len, "protein")
    # rna_lengths = [len(seq) ]
    # protein_lengths = [len(seq) ]
    
    return dna_length, rna_length, protein_length

# 存储每行的DNA、RNA、Protein的长度
dna_lengths = []
rna_lengths = []
protein_lengths = []

# 对每一行的input列进行操作
for text in data['input']:
    extract_lengths(text)
    dna, rna, protein = extract_lengths(text)
    dna_lengths.extend(dna)
    rna_lengths.extend(rna)
    protein_lengths.extend(protein)

# 计算均值
mean_dna_length = sum(dna_lengths) / len(dna_lengths) if dna_lengths else 0
mean_rna_length = sum(rna_lengths) / len(rna_lengths) if rna_lengths else 0
mean_protein_length = sum(protein_lengths) / len(protein_lengths) if protein_lengths else 0

if protein_lengths:
    max_protein_length = max(protein_lengths)
    min_protein_length = min(protein_lengths)
    print(f"最大Protein序列长度: {max_protein_length}")
    print(f"最小Protein序列长度: {min_protein_length}")

# 输出结果
print(f"平均DNA序列长度: {mean_dna_length}")
print(f"平均RNA序列长度: {mean_rna_length}")
print(f"平均Protein序列长度: {mean_protein_length}")
