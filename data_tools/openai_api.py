import requests
import json
import os
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import json
import requests
import re
import time
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 调用方式与openai官网一致，仅需修改baseurl
def claudeshop_api(prompt):
   Baseurl = "https://api.claudeshop.top"
   Skey = ""
   payload = json.dumps({
      "model": "deepseek-r1",
      "messages": [
         {
            "role": "system",
            "content": "You are a helpful assistant."
         },
         {
            "role": "user",
            "content": "<rna>CCCCCTCCTCCCGGACGAGGCGGCCGGGGTGGTAGCAGAGCTCGGAATCTTCCTCTTCCTCCACCACCACCACCTAGAGGGGGGTAAGTTGTTTTTTTTGT<rna> What are the RNA modifications predicted by the analysis of the sequence?"
         }
      ]
   })
   url = Baseurl + "/v1/chat/completions"
   headers = {
      'Accept': 'application/json',
      'Authorization': f'Bearer {Skey}',
      'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
      'Content-Type': 'application/json'
   }

   response = requests.request("POST", url, headers=headers, data=payload)
   print(response)

   # 解析 JSON 数据为 Python 字典
   data = response.json()

   # 获取 content 字段的值
   content = data

   print(content)

# 读取Parquet文件
def read_parquet(input_file):
    df = pd.read_parquet(input_file)
    return df

# deepseek-r1-0528
# https://dashscope.aliyuncs.com/compatible-mode/v1
# 定义ali_api函数
def ali_api(input_text, Key):
    try:
        client = OpenAI(
            api_key=Key,  # 使用正确的API Key
            base_url="https://180.163.156.43:21020/dsr1/v1/",
        )
        system_message = 'You are a helpful assistant.'
        completion = client.chat.completions.create(
            model="deepseek",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=[
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': input_text}
            ],
        )
        outputs = json.loads(completion.model_dump_json())
    
        # 提取 content, reasoning_content 和 completion_tokens
        content = outputs['choices'][0]['message']['content']
        reasoning_content = outputs['choices'][0]['message']['reasoning_content']
        completion_tokens = outputs['usage']['completion_tokens']

        return content, reasoning_content, completion_tokens
    except Exception as e:
        # 捕获所有异常，并打印错误信息
        print(f"Error occurred while calling ali_api: {e}")


def lab_api(input_text, Key):
    try:
        name ='deepseek'
        model_url = 'https://180.163.156.43:21020/dsr1/v1/chat/completions'
        api_key = Key
        system_message = 'You are a helpful assistant.'
        payload = {
                    "model": name,
                    "messages":[
                        {'role': 'system', 'content': 'You are a helpful assistant.'},
                        {'role': 'user', 'content': input_text}
                    ],
                    "stream": False
        }
        response = requests.post(
                model_url,
                json=payload,
                verify=False,
                headers={"Authorization": f"Bearer {api_key}"}
        )
        outputs = json.loads(response.content)
        raw_content = outputs['choices'][0]['message']['content']
        reasoning_content = None
        content = raw_content

        # 如果 raw_content 中包含 <think> 标签，提取其中的内容
        match = re.search(r"<think>(.*?)</think>", raw_content, re.DOTALL)
        if match:
            reasoning_content = match.group(1)  # 提取 <think> 和 </think> 之间的内容
            content = raw_content.replace(match.group(0), "")  # 去除 <think> 和 </think> 标签部分

        # 获取 completion_tokens
        completion_tokens = outputs['usage']['completion_tokens']

        return content, reasoning_content, completion_tokens
    except Exception as e:
        # 捕获所有异常，并打印错误信息
        print(f"Error occurred while calling ali_api: {e}")


# 读取文件中已存在的ID
def read_existing_ids(output_file):
    existing_ids = set()
    try:
        with open(output_file, 'r') as f:
            for line in f:
                result = json.loads(line)
                existing_ids.add(result['id'])
    except FileNotFoundError:
        # 如果文件不存在，返回一个空集合
        return existing_ids
    return existing_ids


# 顺序处理多个请求并保存结果，增加并发处理
def process_data(df, output_file, Key, max_workers):
    existing_ids = read_existing_ids(output_file)  # 读取已存在的ID
    with open(output_file, 'a') as f:  # 使用追加模式打开文件
        # 创建一个线程池
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 使用 tqdm 显示进度条
            futures = []
            for index, row in df.iterrows():
                id = index + 1  # 为每个请求生成唯一的id，顺序从1开始
                if id in existing_ids:
                    continue  # 如果ID已存在，跳过该行
                # 提交异步任务
                futures.append(executor.submit(process_row, row, Key, f, id))

            # 使用tqdm更新进度
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing", ncols=100):
                pass 


def process_row(row, Key, f, id):
    input_text = row['input']  # 获取输入
    task = row['task']  # 获取 task
    kind = row['kind']  # 获取 kind
    label = row['label']  # 获取 label
    gt_output = row['output']

    # 调用 ali_api 获取结果
    # content, reasoning_content, completion_tokens = ali_api(input_text, Key)
    # content, reasoning_content, completion_tokens = lab_api(input_text, Key)
    content, reasoning_content, completion_tokens = test_inetrn_api(input_text, Key)

    # 创建结果字典
    result = {
        'id': id,  # 为每个输出添加唯一的 id
        'task': task,  # 填充 task
        'input': row['input'],  # 获取原始 input
        'decoded_output': content,
        'think': reasoning_content,
        'kind': kind,  # 填充 kind
        'gt_label': label,  # 保留原数据的 label
        'gt_output': gt_output,
        'completion_tokens': completion_tokens
    }

    # 写入结果到文件
    f.write(json.dumps(result) + '\n')
    f.flush()  # 刷新缓冲区，确保写入

def test_ali_api(input_text, Key):
    try:
        name ='deepseek'
        model_url = 'https://180.163.156.43:21020/dsr1/v1/chat/completions'
        api_key = Key
        system_message = 'You are a helpful assistant.'
        payload = {
                    "model": name,
                    "messages":[
                        {'role': 'system', 'content': 'You are a helpful assistant.'},
                        {'role': 'user', 'content': input_text}
                    ],
                    "stream": False
        }
        response = requests.post(
                model_url,
                json=payload,
                verify=False,
                headers={"Authorization": f"Bearer {api_key}"}
        )
        outputs = json.loads(response.content)
        raw_content = outputs['choices'][0]['message']['content']
        reasoning_content = None
        content = raw_content

        # 如果 raw_content 中包含 <think> 标签，提取其中的内容
        match = re.search(r"<think>(.*?)</think>", raw_content, re.DOTALL)
        if match:
            reasoning_content = match.group(1)  # 提取 <think> 和 </think> 之间的内容
            content = raw_content.replace(match.group(0), "")  # 去除 <think> 和 </think> 标签部分

        # 获取 completion_tokens
        completion_tokens = outputs['usage']['completion_tokens']
        return content, reasoning_content, completion_tokens
    except Exception as e:
        # 捕获所有异常，并打印错误信息
        print(f"Error occurred while calling ali_api: {e}")


def test_inetrn_api(input_text, Key):
    try:
        client = OpenAI(
            api_key=Key,  # 此处传token，不带Bearer
            base_url="https://chat.intern-ai.org.cn/api/v1/",
        )

        chat_rsp = client.chat.completions.create(
            model="intern-s1",
            messages=[{"role": "user", "content": input_text}],
            extra_body=dict(thinking_mode=True)
        )
        outputs = json.loads(chat_rsp.model_dump_json())
        content = outputs['choices'][0]['message']['content']
        completion_tokens = outputs['usage']['completion_tokens']
        reasoning_content = outputs['choices'][0]['message']['reasoning_content']
    except Exception as e:
        # 捕获所有异常，并打印错误信息
        print(f"Error occurred while calling ali_api: {e}")

    return content, reasoning_content, completion_tokens


if __name__ == "__main__":
   part = "FEC2cpd"
   # part = "protein"
   input_file_path = f"/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/TargetTask0808/eval/part_{part}.parquet"
   # input_file_path = "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/TargetTask0729/protein/val_target_task_protein.parquet"
   output_file_path = f"/fs-computility/ai4agr/lijinzhe/code/BioMLLM_V2/res/inference/intern-s1/output_{part}.json"
   # output_file_path = f"/fs-computility/ai4agr/lijinzhe/code/BioMLLM_V2/res/inference/deepseek-r1/output_{part}.json"
   Key = "xxx"
   # Key = os.getenv('API_KEY')
   max_workers = 22
   df = pd.read_parquet(input_file_path)
   process_data(df, output_file_path, Key, max_workers)
   # print(test_ali_api("hello", Key))
   # test_inetrn_api("hello, can you think?", Key)
