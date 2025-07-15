import json, re
from collections import Counter
from pathlib import Path
from typing import Iterable, Tuple, Set, Dict, FrozenSet


# ---------------- 通用过滤 ----------------
def filter_by_markers(
    input_path: str | Path,
    output_path: str | Path,
    *,
    require: Iterable[str] = ("<dna>",),
    forbid: Iterable[str] | None = None,
    mode: str = "all",                # "all" | "any"
) -> int:
    """
    写出满足条件的样本：
      mode="all" : 同时包含所有 require 中的标识
      mode="any" : 至少包含一个 require 中的标识
      forbid     : 若出现其中任意标识则剔除
    返回写出的行数
    """
    require = set(require)
    forbid  = set(forbid or [])
    kept = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for raw in fin:
            raw = raw.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue

            text = item.get("input", "")

            # --- 判断是否保留 ---
            if mode == "all":
                ok = all(m in text for m in require)
            elif mode == "any":
                ok = any(m in text for m in require)
            else:
                raise ValueError("mode 必须是 'all' 或 'any'")

            if ok and not any(m in text for m in forbid):
                json.dump(item, fout, ensure_ascii=False)
                fout.write("\n")
                kept += 1

    print(f"[filter] 写出满足条件的样本: {kept}")
    return kept


# ---------------- 替换特殊标识 ----------------
def replace_reserved_tokens(
    input_path: str | Path,
    output_path: str | Path,
    *,
    replace_in_fields: Tuple[str, ...] = ("input", "output"),
    verbose: bool = True,
) -> Dict[str, int]:
    """
    替换文本中的特殊标识：
      <|reserved_special_token_1|> -> <dna>
      <|reserved_special_token_2|> -> <rna>
      <|reserved_special_token_3|> -> <protein>
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        replace_in_fields: 需要替换的字段名称，默认为input和output
        verbose: 是否打印详细信息
    
    Returns:
        包含处理统计信息的字典：
        - processed: 处理的总行数
        - replaced_dna: 替换的DNA标识数量
        - replaced_rna: 替换的RNA标识数量
        - replaced_protein: 替换的Protein标识数量
    """
    # 定义替换映射
    token_map = {
        "<|reserved_special_token_1|>": "<dna>",
        "<|reserved_special_token_2|>": "<rna>",
        "<|reserved_special_token_3|>": "<protein>"
    }
    
    # 统计信息
    stats = {
        "processed": 0,
        "replaced_dna": 0,
        "replaced_rna": 0,
        "replaced_protein": 0,
        "files_with_replacements": 0
    }
    
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line_num, raw in enumerate(fin, 1):
            raw = raw.strip()
            if not raw:
                continue
            
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                if verbose:
                    print(f"[警告] 第{line_num}行JSON解析失败，跳过")
                continue
            
            file_has_replacements = False
            
            # 对每个指定字段进行替换
            for field in replace_in_fields:
                if field in item:
                    text = item[field]
                    
                    # 只处理字符串类型的字段
                    if isinstance(text, str):
                        # 记录原始文本以检测变化
                        original_text = text
                        
                        # 执行替换
                        for token, replacement in token_map.items():
                            count = text.count(token)
                            if count > 0:
                                text = text.replace(token, replacement)
                                
                                # 更新统计信息
                                if replacement == "<dna>":
                                    stats["replaced_dna"] += count
                                elif replacement == "<rna>":
                                    stats["replaced_rna"] += count
                                elif replacement == "<protein>":
                                    stats["replaced_protein"] += count
                                
                                file_has_replacements = True
                        
                        # 如果文本有变化，更新字段
                        if text != original_text:
                            item[field] = text
            
            # 写出修改后的JSON
            json.dump(item, fout, ensure_ascii=False)
            fout.write("\n")
            stats["processed"] += 1
            
            if file_has_replacements:
                stats["files_with_replacements"] += 1
    
    # 打印统计信息
    if verbose:
        print(f"\n[替换] 共处理 {stats['processed']} 行数据")
        print(f"[替换] 替换了 {stats['replaced_dna']} 个 DNA 标识")
        print(f"[替换] 替换了 {stats['replaced_rna']} 个 RNA 标识")
        print(f"[替换] 替换了 {stats['replaced_protein']} 个 Protein 标识")
        print(f"[替换] 有替换操作的行数: {stats['files_with_replacements']}")
    
    return stats


# ---------------- 组合与未知标识统计（保持不变） ----------------
def marker_statistics(
    input_path: str | Path,
    known_markers: Tuple[str, ...] = ("<dna>", "<rna>", "<protein>"),
    *,
    min_combo_len: int = 2,
) -> Dict[str, object]:
    pat_any_marker = re.compile(r"<[^>]+>")
    combo_counter: Counter[FrozenSet[str]] = Counter()
    other_marker_set: Set[str] = set()
    other_marker_count = 0

    with open(input_path, "r", encoding="utf-8") as fin:
        for raw in fin:
            raw = raw.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue

            text = item.get("input", "")

            present = frozenset(m for m in known_markers if m in text)
            if len(present) >= min_combo_len:
                combo_counter[present] += 1

            found = set(pat_any_marker.findall(text))
            unknown = found.difference(known_markers)
            if unknown:
                other_marker_count += 1
                other_marker_set.update(unknown)

    # ----- 打印 -----
    print(f"\n[stats] 行内含 ≥{min_combo_len} 个已知标识的组合计 {sum(combo_counter.values())} 行")
    for combo, freq in combo_counter.most_common():
        print(f"  {' + '.join(sorted(combo)):<30} : {freq}")
    print(f"[stats] 含未知 <...> 标识的行数          : {other_marker_count}")
    print(f"[stats] 未知标识集合                 : {', '.join(sorted(other_marker_set)) or 'None'}")

    return {
        "combo_counter": combo_counter,
        "other_marker_count": other_marker_count,
        "other_marker_set": other_marker_set,
    }


# ---------------- 示例 ----------------
if __name__ == "__main__":
    input_file = "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/dev_new.jsonl"
    out_file = "/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/dev_dna_rna.jsonl"

    # 过滤：只要包含 <dna> 或 <rna>（或同时包含二者），并排除 <protein>
    filter_by_markers(
        input_path=input_file,
        output_path=out_file,
        require=("<dna>", "<rna>"),
        forbid=("<protein>",),
        mode="any",
    )

    # 统计示例
    # marker_statistics(input_file)
    
    # # 替换特殊标识示例
    # replace_reserved_tokens(
    #     input_path="/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/dev.jsonl",
    #     output_path="/tos-bjml-ai4agr/lijinzhe/dataset/BioMLLM/dev_new.jsonl",
    # )
