import json
import random
import re

# 检查一个字符串是否只包含AGCT序列
def is_dna_sequence(output):
    return bool(re.match('^[AGCT]+$', output))  # 检查是否全是AGCT

# 读取 JSON 数据
def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = []
        for line in file:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding line: {e}")
        return data

# 按照每个task划分为 8:2
def split_data_by_task(data):
    task_dict = {}
    
    # 根据task字段将数据分类
    for entry in data:
        task = entry['task']
        if task not in task_dict:
            task_dict[task] = []
        task_dict[task].append(entry)

    train_data = []
    valid_data = []

    # 为每个task分类并划分为 8:2
    for task, entries in task_dict.items():
        random.shuffle(entries)  # 打乱数据
        split_index = int(0.8 * len(entries))  # 80%的数据用于训练
        train_data.extend(entries[:split_index])
        valid_data.extend(entries[split_index:])
    
    return train_data, valid_data

# 过滤掉不符合要求的output数据
def filter_data(data):
    filtered_data = []
    for entry in data:
        output = entry['output']
        
        # 如果output为空或是基因序列（AGCT序列），则跳过
        if not output or is_dna_sequence(output):
            continue
        
        filtered_data.append(entry)
    
    return filtered_data

# 逐条写入数据到 JSON 文件
def save_json_file(data, file_path):
    with open(file_path, 'w') as file:
        for entry in data:
            json.dump(entry, file, ensure_ascii=False)
            file.write("\n")  # 每个 JSON 对象写入新的一行

# 主函数
def process_data(input_file, output_train_file, output_valid_file):
    # 加载原始数据
    data = load_json_file(input_file)

    # 过滤掉不符合要求的output数据
    filtered_data = filter_data(data)

    # 按照task划分为train和valid数据
    train_data, valid_data = split_data_by_task(filtered_data)

    # 保存train和valid数据
    save_json_file(train_data, output_train_file)
    save_json_file(valid_data, output_valid_file)

    print(f"Data successfully split. Train data: {len(train_data)}, Valid data: {len(valid_data)}")

# 文件路径示例
input_file = '/fs-computility/ai4agr/lijinzhe/code/BioMLLM_V2/data_tools/sample/rewritten_output_0717.json'
output_train_file = '/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/rewritten_8k/train_data.json'
output_valid_file = '/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/rewritten_8k/valid_data.json'

# 执行处理
process_data(input_file, output_train_file, output_valid_file)
