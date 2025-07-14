import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Union, Dict, List

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

def sample_jsonl_by_task(
    input_dir: Union[str, Path],
    output_path: Union[str, Path],
    sample_num_per_task: int = 1
) -> None:
    """
    从 input_dir 下所有 .jsonl 文件中按 task 字段分组采样，每个 task 随机抽 sample_num_per_task 条写入 output_path。

    参数:
        input_dir (str | Path): 输入目录，包含多个 jsonl 文件
        output_path (str | Path): 输出 jsonl 文件路径
        sample_num_per_task (int): 每个 task 抽样的数量（默认 1 条）
    example:
        sample_jsonl_by_task(
        input_dir="xxx",
        output_path="sampled_by_task.jsonl",
        sample_num_per_task=3  # 每个任务抽取3条
    )
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    task_to_items = defaultdict(list)

    for file_path in input_dir.glob("*.jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    task = obj.get("task", "Unknown")
                    task_to_items[task].append(obj)
                except json.JSONDecodeError:
                    continue

    sampled_data = []
    for task, items in task_to_items.items():
        sampled = random.sample(items, min(sample_num_per_task, len(items)))
        sampled_data.extend(sampled)

    with open(output_path, "w", encoding="utf-8") as out_file:
        for item in sampled_data:
            out_file.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ 总共采样 {len(sampled_data)} 条，来自 {len(task_to_items)} 个 task，已写入：{output_path}")


# 基于llm的评估
def evaluate_label_match_qwen_chat(
    input_file: str,
    model_path: str,
    output_file: str = None,
    max_eval: int = None,
    max_new_tokens: int = 128
):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    results = []
    with open(input_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, desc="Evaluating")):
            if max_eval and idx >= max_eval:
                break

            obj = json.loads(line)
            label = obj["label"]
            llm_output = obj["pre_output"]
            raw_input = obj.get("input", "")
            raw_output = obj.get("output", "")

            # === Updated prompt with input ===
            prompt = (
                "You are a professional evaluator. Your task is to determine whether the following LLM output "
                "correctly and explicitly conveys the meaning of the given label for the given question.\n\n"
                f"Question: {raw_input}\n"
                f"Label: {label}\n"
                f"LLM Output: {llm_output}\n\n"
                "Answer strictly with \"Yes\" or \"No\" only. Do not explain."
            )

            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            inputs = tokenizer([text], return_tensors="pt").to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id
                )

            output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()

            try:
                think_end = len(output_ids) - output_ids[::-1].index(151668)  # </think>
            except ValueError:
                think_end = 0

            thinking_text = tokenizer.decode(output_ids[:think_end], skip_special_tokens=True).strip()
            final_text = tokenizer.decode(output_ids[think_end:], skip_special_tokens=True).strip()

            answer_clean = final_text.strip().lower()
            correct = answer_clean.startswith("yes")

            result = {
                "task": obj.get("task"),
                "input": raw_input,
                "output": raw_output,
                "label": label,
                "llm_output": llm_output,
                "answer_raw": final_text,
                "thinking": thinking_text,
                "correct": correct
            }
            results.append(result)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f_out:
            for r in results:
                f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"✅ Results saved to {output_file}")

    if results:
        # 计算总体准确率
        acc = sum(r["correct"] for r in results)
        print(f"✅ Overall Accuracy: {acc}/{len(results)} = {acc / len(results):.2%}")
        
        # 按任务统计准确率
        task_stats = calculate_accuracy_by_task(results)
        print("\n📊 各任务准确率统计:")
        print("-" * 60)
        print(f"{'Task Name':<30} | {'Correct':<8} | {'Total':<8} | {'Accuracy':<10}")
        print("-" * 60)
        
        for task, (correct, total) in sorted(task_stats.items(), key=lambda x: x[1][1], reverse=True):
            accuracy = correct / total if total > 0 else 0
            print(f"{task[:30]:<30} | {correct:<8} | {total:<8} | {accuracy:.2%}")
        
        print("-" * 60)
        
        # 保存任务统计结果
        if output_file:
            stats_output = output_file.replace('.jsonl', '_task_stats.json')
            task_stats_json = {
                task: {"correct": correct, "total": total, "accuracy": correct / total if total > 0 else 0}
                for task, (correct, total) in task_stats.items()
            }
            with open(stats_output, "w", encoding="utf-8") as f_stats:
                json.dump(task_stats_json, f_stats, ensure_ascii=False, indent=2)
            print(f"✅ Task statistics saved to {stats_output}")

    return results

def calculate_accuracy_by_task(results: List[Dict]) -> Dict[str, tuple]:
    """
    按任务统计准确率
    
    Args:
        results: 评估结果列表
        
    Returns:
        Dict[str, tuple]: 任务名称到(正确数, 总数)的映射
    """
    task_stats = defaultdict(lambda: [0, 0])  # [correct_count, total_count]
    
    for result in results:
        task = result.get("task", "Unknown")
        task_stats[task][1] += 1  # 总数+1
        if result.get("correct", False):
            task_stats[task][0] += 1  # 正确数+1
    
    # 将列表转换为元组
    return {task: tuple(counts) for task, counts in task_stats.items()}


if __name__ == "__main__":
    sample_file_path = "/fs-computility/ai4agr/lijinzhe/res_data_model/biomllm_res/inference_results/Qwen3_4B_NT_sft_exp1_0710/predictions_300_samples_step500.jsonl"
    llm_path="/fs-computility/ai4agr/lijinzhe/basemodel/Qwen3-14B"
    # sample_jsonl_by_task(
    #     input_dir="/fs-computility/ai4agr/lijinzhe/res_data_model/0621_test",
    #     output_path="/fs-computility/ai4agr/lijinzhe/res_data_model/0621_test/sampled_by_task.jsonl",
    #     sample_num_per_task=20
    # )
    evaluate_label_match_qwen_chat(
        input_file=sample_file_path,
        model_path=llm_path,
        output_file="Qwen3_4B_NT_sft_exp1_0710_step500.jsonl",
        max_new_tokens=512)

