import json
import os
from typing import List, Dict

from vllm import LLM, SamplingParams

# ------------------------------------------------------------------
# 常量
# ------------------------------------------------------------------
MODEL_PATH = "/fs-computility/ai4agr/shared/Qwen3-32B"
DATA_PATH  = "/fs-computility/ai4agr/lijinzhe/code/BioMLLM_V2/data_tools/sample/filter_dna_rna_test_dataset.jsonl"

SYSTEM_PROMPT = (
    "You are a precise bioinformatics assistant. "
    "The user has provided a DNA/RNA sequence and a question. "
    "Your job is to generate a detailed, fluent paragraph that:\n"
    "  1) restates the question in your own words,\n"
    "  2) briefly analyses any relevant motifs or regions,\n"
    "  3) gives biological reasoning in 2–3 sentences,\n"
    "  4) ends with a single, clear sentence that exactly matches the ground-truth label.\n"
    "Do NOT contradict the ground-truth label."
)

SAMPLING_KWARGS = dict(
    temperature=0.7,  # Increased temperature for diversity
    top_p=0.95,      # Increase top_p for more diverse output
    max_tokens=1024,  # Output length of 1k tokens
    stop=["<|im_end|>"]
)

# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------
def _build_prompt(sample: Dict) -> str:
    """构造 chat 模板 prompt，包含原始的 output"""
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}\n"
        "Ground-truth label (you must agree): "
        f"{sample['label']}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{sample['input']}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "Here is the original output:\n"
        f"{sample.get('output', 'No previous output available.')}\n"
        "Now, expand the analysis while maintaining the original conclusion and providing a detailed explanation.\n"
    )

def _load_jsonl(path: str, max_samples: int = None) -> List[Dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_samples is not None and idx >= max_samples:
                break
            samples.append(json.loads(line))
    return samples

def chunk_prompts(prompts: List[str], batch_size: int) -> List[List[str]]:
    """将 prompts 切分成小批次"""
    return [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]

# ------------------------------------------------------------------
# 主函数
# ------------------------------------------------------------------
def run_vllm_rewrite(
        max_samples: int = None,
        out_path: str = "rewritten_outputs.jsonl",
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 8,
        batch_size: int = 128,  # 每个批次的大小
        **sampling_overrides
    ) -> None:
    """
    Parameters
    ----------
    max_samples : int | None
        如果给定，只处理前 N 条；None 则全部。
    out_path : str
        结果 jsonl 保存路径。
    gpu_memory_utilization, tensor_parallel_size : vLLM 参数
    batch_size : int
        每个批次的大小
    sampling_overrides : 额外采样参数
    """
    # 1. 读数据
    samples = _load_jsonl(DATA_PATH, max_samples)
    if not samples:
        print("No samples loaded. Exiting.")
        return

    # 2. 初始化 vLLM
    llm = LLM(
        model=MODEL_PATH,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True
    )
    sp = SamplingParams(**{**SAMPLING_KWARGS, **sampling_overrides})

    # 3. 构造 prompts
    prompts = [_build_prompt(s) for s in samples]

    # 4. 分批处理 prompts
    chunked_prompts = chunk_prompts(prompts, batch_size)

    # 5. 推理 (每批次处理)
    outputs = []
    from tqdm import tqdm

    # Wrap the loop with tqdm to show progress
    for prompt_batch in tqdm(chunked_prompts, desc="Processing Prompts"):
        batch_outputs = llm.generate(prompt_batch, sp)
        outputs.extend(batch_outputs)

    # 6. 组装新样本
    new_samples = []
    for s, out in zip(samples, outputs):
        rewritten_text = out.outputs[0].text.strip()

        # 生成新的样本
        new_samples.append({
            **s,                    # 保持所有旧字段
            "output": rewritten_text,  # 新的 output，包含原始的 output
            "old_output": s.get("output", "")  # 保留原始的 output
        })

    # 7. 写回
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in new_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Rewritten {len(new_samples)} samples -> {out_path}")

if __name__ == "__main__":
    run_vllm_rewrite(out_path="/fs-computility/ai4agr/lijinzhe/code/BioMLLM_V2/data_tools/sample/rewritten_output_0717.jsonl", batch_size=128)
