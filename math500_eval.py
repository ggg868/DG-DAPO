import json
import re
import math
import argparse
from math_verify import parse, verify

def extract_model_answer(response_text):
    """提取 \boxed{} 里的内容"""
    boxed_matches = re.findall(r"\\boxed\{((?:[^{}]|(?:\{[^{}]*\}))*)\}", response_text)
    if boxed_matches:
        return boxed_matches[-1].strip()
    return None

def pass_at_k_expected(n, c, k):
    """无偏计算 pass@k 期望"""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="vLLM生成的MATH结果文件")
    parser.add_argument("--debug_file", type=str, default="math500_debug_log.jsonl", help="判定日志输出路径")
    args = parser.parse_args()

    total_questions = 0
    total_pass_1_prob = 0.0
    pass_32_count = 0
    maj_32_count = 0
    
    debug_logs = []

    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            prompt = data.get("problem", "Unknown problem") 
            responses = data["responses"] # list of 32 strings
            gt_ans = data["answer"]       # 原始正确答案
            
            # 1. 解析真实答案
            try:
                gt_parsed = parse(gt_ans)
            except Exception:
                gt_parsed = None # 极少数极端格式可能会触发
            
            # 2. 提取所有模型生成的答案
            extracted_answers = [extract_model_answer(r) for r in responses]
            
            # 3. 判定每个答案的正确性
            correctness = []
            parsed_valid = [] # 用于保存 (原字符串, 解析后的对象) 供后续投票使用
            
            for ans in extracted_answers:
                if ans is None or gt_parsed is None:
                    correctness.append(False)
                else:
                    try:
                        ans_parsed = parse(ans)
                        # 调用 sympy 引擎进行严谨的代数等价判定
                        is_correct = verify(ans_parsed, gt_parsed)
                        correctness.append(is_correct)
                        parsed_valid.append((ans, ans_parsed))
                    except Exception:
                        # 如果模型生成了乱码导致解析失败，直接判错
                        correctness.append(False)
            
            c = sum(correctness)
            n = len(responses)
            if n == 0: continue
            
            total_questions += 1
            
            # 记录 Pass@1 和 Pass@32
            total_pass_1_prob += pass_at_k_expected(n, c, 1)
            if c > 0: pass_32_count += 1
                
            # 4. 语义级多数投票 (Semantic Majority Voting)
            maj_ans_raw = None
            is_maj_correct = False
            
            if parsed_valid:
                best_count = 0
                maj_parsed = None
                
                # O(N^2) 寻找等价次数最多的答案 (N最多32，纯CPU计算极快)
                for i in range(len(parsed_valid)):
                    raw_i, parsed_i = parsed_valid[i]
                    current_count = 0
                    
                    for j in range(len(parsed_valid)):
                        _, parsed_j = parsed_valid[j]
                        try:
                            # 如果两个答案在代数上等价，则算作同一票
                            if verify(parsed_i, parsed_j):
                                current_count += 1
                        except Exception:
                            pass
                            
                    if current_count > best_count:
                        best_count = current_count
                        maj_ans_raw = raw_i
                        maj_parsed = parsed_i
                        
                # 检查最高票数的答案是否正确
                if maj_ans_raw is not None and gt_parsed is not None:
                    try:
                        is_maj_correct = verify(maj_parsed, gt_parsed)
                        if is_maj_correct:
                            maj_32_count += 1
                    except Exception:
                        pass
            
            # 5. 记录日志
            debug_logs.append({
                "problem": prompt[:100] + "...",
                "ground_truth": gt_ans,
                "majority_answer": maj_ans_raw,
                "is_maj_correct": is_maj_correct,
                "correct_count_in_32": c,
                "first_extracted_answer": extracted_answers[0]
            })

    # 写入日志文件
    with open(args.debug_file, "w", encoding="utf-8") as df:
        for log in debug_logs:
            df.write(json.dumps(log, ensure_ascii=False) + "\n")

    # 打印最终所有指标
    print("="*40)
    print(f"Dataset Size : {total_questions}")
    print(f"Pass@1       : {(total_pass_1_prob / total_questions) * 100:.2f}%")
    print(f"Pass@32      : {(pass_32_count / total_questions) * 100:.2f}%")
    print(f"Maj@32       : {(maj_32_count / total_questions) * 100:.2f}%")
    print(f"审计日志已保存至: {args.debug_file}")
    print("="*40)

if __name__ == "__main__":
    main()