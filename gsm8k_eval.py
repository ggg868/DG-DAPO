import json
import re
import math
from collections import Counter
import argparse

def extract_ground_truth(gt_text):
    """提取原始 GSM8K 的真实答案 (处理掉千分位)"""
    if "#### " in gt_text:
        ans = gt_text.split("#### ")[-1].strip()
    else:
        ans = gt_text.strip()
    return ans.replace(",", "")

def extract_model_answer(response_text):
    """双重正则提取：优先找 \boxed{}，找不到再找最后一段数字"""
    # 1. 优先尝试提取 \boxed{} 里的内容
    boxed_matches = re.findall(r"\\boxed\{([^{}]+)\}", response_text)
    if boxed_matches:
        ans = boxed_matches[-1] # 取最后一个框
        return ans.replace(",", "").replace("$", "").strip()
    
    # 2. 兜底策略：提取整段文本的最后一个数字（带负号和小数点）
    numbers = re.findall(r"(?<![a-zA-Z])-?\d+(?:\.\d+)?", response_text.replace(",", ""))
    if numbers:
        return numbers[-1]
    
    return None

def is_equivalent(model_ans, gt_ans):
    """数值安全的等价判断"""
    if model_ans is None or gt_ans is None:
        return False
    try:
        # 转为浮点数，解决 400 == 400.0 的问题
        return abs(float(model_ans) - float(gt_ans)) < 1e-5
    except ValueError:
        return str(model_ans).strip() == str(gt_ans).strip()

def pass_at_k_expected(n, c, k):
    """无偏计算 pass@k 期望"""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="vLLM生成的结果文件")
    parser.add_argument("--debug_file", type=str, default="eval_debug_log.jsonl", help="判定日志输出路径")
    args = parser.parse_args()

    total_questions = 0
    total_pass_1_prob = 0.0
    pass_32_count = 0
    maj_32_count = 0
    
    debug_logs = []

    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            prompt = data.get("prompt", "Unknown prompt")
            responses = data["responses"] # list of 32 strings
            gt_text = data["answer"]      # 原始答案字段
            
            gt_ans = extract_ground_truth(gt_text)
            
            # 解析 32 个输出
            extracted_answers = [extract_model_answer(r) for r in responses]
            correctness = [is_equivalent(ans, gt_ans) for ans in extracted_answers]
            
            c = sum(correctness) # 32个里正确的个数
            n = len(responses)   # 32
            if n == 0: continue
            
            total_questions += 1
            
            # --- 指标计算核心 ---
            # 1. Pass@1 (无偏期望值，即正确率的平均)
            p1_expected = pass_at_k_expected(n, c, 1)
            total_pass_1_prob += p1_expected
            
            # 2. Pass@32 (至少对 1 个就算对)
            if c > 0: pass_32_count += 1
                
            # 3. Majority@32 (多数投票)
            valid_answers = [ans for ans in extracted_answers if ans is not None]
            maj_ans = None
            is_maj_correct = False
            if valid_answers:
                maj_ans = Counter(valid_answers).most_common(1)[0][0]
                if is_equivalent(maj_ans, gt_ans):
                    maj_32_count += 1
                    is_maj_correct = True
            
            # --- 记录审查日志，用于确保提取正确 ---
            debug_logs.append({
                "question": prompt[:100] + "...", # 截断一点防止过长
                "ground_truth": gt_ans,
                "majority_answer": maj_ans,
                "is_maj_correct": is_maj_correct,
                "correct_count_in_32": c,
                "first_extracted_answer": extracted_answers[0] # 只看第一个用于 debug
            })

    # 将审查日志写入文件
    with open(args.debug_file, "w", encoding="utf-8") as df:
        for log in debug_logs:
            df.write(json.dumps(log, ensure_ascii=False) + "\n")

    # 打印最终所有指标
    print("="*40)
    print(f"Dataset Size : {total_questions}")
    print(f"Pass@1       : {(total_pass_1_prob / total_questions) * 100:.2f}%")
    print(f"Pass@32      : {(pass_32_count / total_questions) * 100:.2f}%")
    print(f"Maj@32       : {(maj_32_count / total_questions) * 100:.2f}%")
    print(f"审查日志已保存至: {args.debug_file}")
    print("请务必抽查日志中的 ground_truth 和 majority_answer 是否提取正确！")
    print("="*40)

if __name__ == "__main__":
    main()