# pylint: skip-file
import json

# 定义文件路径
file_path = "/fs-computility/ai4agr/lijinzhe/code/BioMLLM_V2/res/inference/deepseek-r1/output_2.json"

# 初始化一个列表来存储所有的 completion_tokens
completion_tokens = []

# 逐行读取 JSON 文件
with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        try:
            # 尝试解析每一行作为独立的 JSON 对象
            data = json.loads(line)

            # 假设每个条目都有 'completion_tokens' 字段
            if "completion_tokens" in data:
                completion_tokens.append(data["completion_tokens"])
        except json.JSONDecodeError as e:
            print(f"解析失败: {e}")

# 计算 completion_tokens 的均值
if completion_tokens:
    average_tokens = sum(completion_tokens) / len(completion_tokens)
    print(f"均值: {average_tokens}")
else:
    print("没有找到 completion_tokens 字段")
