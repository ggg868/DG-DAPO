import json
import re
import math
from collections import Counter, defaultdict
import argparse

def last_boxed_only_string(string: str):
    """从 verl 借用的提取最后 \boxed{} 的精确函数，支持嵌套括号"""
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    for i in range(idx, len(string)):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None

def remove_boxed(s: str) -> str:
    left = "\\boxed{"
    return s[len(left) : -1]

def extract_aime_answer(response_text):
    """
    结合 verl 括号匹配与你的 AIME 专属提取逻辑
    """
    boxed_str = last_boxed_only_string(response_text)
    if boxed_str:
        ans_candidate = remove_boxed(boxed_str)
        ans_candidate = ans_candidate.replace(",", "").replace("$", "").replace("\\text{", "").replace("}", "").strip()
        try:
            val = float(ans_candidate)
            if val.is_integer() and 0 <= val <= 999:
                return str(int(val))
        except ValueError:
            pass 

    numbers = re.findall(r"(?<![a-zA-Z\.\-])\d+(?!\.\d+)", response_text.replace(",", ""))
    if numbers:
        for num_str in reversed(numbers):
            try:
                val = int(num_str)
                if 0 <= val <= 999:
                    return str(val)
            except ValueError:
                continue
                
    return None

def is_equivalent_aime(model_ans, gt_ans):
    """AIME 答案比对"""
    if model_ans is None or gt_ans is None:
        return False
    return str(model_ans).strip() == str(gt_ans).strip()

def pass_at_k_expected(n, c, k):
    """无偏计算 pass@k 期望"""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="vLLM生成的AIME结果文件(960行)")
    parser.add_argument("--debug_file", type=str, default="aime_debug_log.jsonl", help="判定日志输出路径")
    # 如果你的单次生成文本字段不叫 "response"，你可以通过参数传入，比如叫 "generated_text"
    parser.add_argument("--response_key", type=str, default="response", help="JSON中代表模型输出的字段名")
    args = parser.parse_args()

    total_questions = 0
    total_pass_1_prob = 0.0
    pass_32_count = 0
    maj_32_count = 0
    
    debug_logs = []
    
    # 🌟 新增：数据聚合字典
    # 键是题目文本，值是一个包含 ground_truth 和 32次回答的字典
    grouped_data = {}

    print("正在读取并聚合数据...")
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            prompt = data.get("problem", "Unknown problem") 
            gt_ans = str(data.get("answer", "")).strip()
            
            # 获取当前行的这 1 次回答，注意替换为你的真实字段名
            response_text = data.get(args.response_key, "") 
            
            # 如果是列表，比如 {"responses": ["单个回答"]}，提取第一个
            if isinstance(response_text, list) and len(response_text) > 0:
                response_text = response_text[0]
            
            if prompt not in grouped_data:
                grouped_data[prompt] = {
                    "problem": prompt,
                    "answer": gt_ans,
                    "responses": []
                }
            
            # 将这条回答加入到该题目的列表中
            grouped_data[prompt]["responses"].append(response_text)

    print(f"聚合完成，共发现 {len(grouped_data)} 道独立题目。开始计算指标...")

    # 对聚合后的每一道题进行评估（现在每道题都有 32 个回答了）
    for prompt, data in grouped_data.items():
        responses = data["responses"] # list of 32 strings
        gt_ans = data["answer"]
        
        extracted_answers = [extract_aime_answer(r) for r in responses]
        correctness = [is_equivalent_aime(ans, gt_ans) for ans in extracted_answers]
        
        c = sum(correctness) # 正确的个数
        n = len(responses)   # 应该是 32
        if n == 0: continue
        
        total_questions += 1
        
        # 1. 计算期望 Pass@1
        total_pass_1_prob += pass_at_k_expected(n, c, 1)
        
        # 2. 计算 Pass@32
        if c > 0: pass_32_count += 1
            
        # 3. majority@32 (多数投票)
        valid_answers = [ans for ans in extracted_answers if ans is not None]
        maj_ans = None
        is_maj_correct = False
        
        if valid_answers:
            maj_ans = Counter(valid_answers).most_common(1)[0][0]
            if is_equivalent_aime(maj_ans, gt_ans):
                maj_32_count += 1
                is_maj_correct = True
        
        debug_logs.append({
            "problem": prompt[:100] + "...",
            "ground_truth": gt_ans,
            "majority_answer": maj_ans,
            "is_maj_correct": is_maj_correct,
            "correct_count_in_32": c,
            "total_sampled_for_this_question": n, # 记录一下实际聚合了多少次，正常是 32
            "first_extracted_answer": extracted_answers[0]
        })

    with open(args.debug_file, "w", encoding="utf-8") as df:
        for log in debug_logs:
            df.write(json.dumps(log, ensure_ascii=False) + "\n")

    print("="*40)
    print(f"Dataset Size : {total_questions} (Independent Questions)")
    print(f"Pass@1       : {(total_pass_1_prob / total_questions) * 100:.2f}%")
    print(f"Pass@32      : {(pass_32_count / total_questions) * 100:.2f}%")
    print(f"Maj@32       : {(maj_32_count / total_questions) * 100:.2f}%")
    print(f"审计日志已保存至: {args.debug_file}")
    print("="*40)

if __name__ == "__main__":
    main()