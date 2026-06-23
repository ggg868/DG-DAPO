import os
import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Dict
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

def make_step_rewards_robust(logits, token_masks):
    #  将 Logits 转换为概率分布
    probabilities = F.softmax(logits, dim=-1)
    all_scores_res = []
    
    #  遍历 Batch 中的每一个样本
    for i in range(probabilities.size(0)):
        # 直接利用布尔掩码，取出当前样本中 <extra_0> 位置的概率分布
        # 结果 shape: (num_valid_steps, num_labels)
        step_probs = probabilities[i][token_masks[i]]
        
        # 提取类别 1（即 "Good" 或 "正确"）的概率
        if step_probs.size(0) > 0:
            good_probs = step_probs[:, 1].cpu().tolist()
        else:
            good_probs = []
            
        all_scores_res.append(good_probs)
        
    return all_scores_res


class PRMScorer:
    """Process Reward Model Scorer optimized for MDPO framework"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Math-PRM-7B",
        system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{}.",
        aggregation_method: str = "log_mean_prob",
        clip_epsilon: float = 1e-4,
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.aggregation_method = aggregation_method
        self.clip_epsilon = clip_epsilon
        
        logger.info(f"[PRMScorer] Loading PRM model from {model_name}...")
        
        # 初始化 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # 开启左侧 Padding，确保生成的最后一步对齐
        self.tokenizer.padding_side = "left" 
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 初始化模型
        self.model = AutoModel.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_cache=False,
        ).eval()
        
        # 提取特殊符 ID
        extra_0_ids = self.tokenizer.encode("<extra_0>", add_special_tokens=False)
        self.step_sep_id = extra_0_ids[0]
        logger.info(f"[PRMScorer] Model loaded. <extra_0> token ID: {self.step_sep_id}")


    def split_steps_and_inject(self, response_text: str) -> str:
        """步骤切分与标识符注入：将 Actor 的输出转化为 PRM 可读格式"""
        if not response_text or not response_text.strip():
            return ""
        steps = [s.strip() for s in response_text.split("\n\n") if s.strip()]
        if len(steps) < 2:
            steps =[s.strip() for s in response_text.replace(". ", ".\n\n").split("\n\n") if s.strip()]
        return "<extra_0>".join(steps) + "<extra_0>" if steps else ""


    def score_batch(self, prompts: List[str], responses: List[str]) -> Dict[str, any]:
        """核心批处理评分管道：一次性处理一个大 Batch"""
        conversations = []
        for p, r in zip(prompts, responses):
            assistant_content = self.split_steps_and_inject(r)
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": p},
                {"role": "assistant", "content": assistant_content}
            ]
            conv_str = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            conversations.append(conv_str)

        # 批处理 Tokenize
        inputs = self.tokenizer(conversations, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        # 全局单次 Forward 推理
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            del outputs
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
        
        # 生成目标位置掩码
        token_masks = (input_ids == self.step_sep_id)
        
        # 张量掩码提取步级分数
        batch_step_rewards = make_step_rewards_robust(logits, token_masks)
        
        # 分数聚合与降维
        trajectory_scores = []
        for scores in batch_step_rewards:
            if not scores:
                trajectory_scores.append(-5.0)
                continue
                
            # 安全截断，防止 log(0) 导致梯度爆炸
            arr = np.clip(np.array(scores, dtype=np.float32), self.clip_epsilon, 1.0 - self.clip_epsilon)
            
            # MDPO-Seq 核心：对数联合概率（Log-Geometric Aggregation）
            if self.aggregation_method == "log_mean_prob":
                trajectory_scores.append(float(np.mean(np.log(arr))))
            else:
                trajectory_scores.append(float(np.mean(arr)))

        return {
            "trajectory_scores": trajectory_scores, 
            "step_scores_list": batch_step_rewards
        }