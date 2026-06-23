from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("dapo")
class DAPORewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

    # 增加 *args, **kwargs
    def __call__(self, data: DataProto, return_dict: bool = False, *args, **kwargs):
        """We will expand this function gradually based on the available datasets"""

        # 保留新分支的 rm_scores 提取逻辑
        # if "rm_scores" in data.batch.keys():
        #     if return_dict:
        #         reward_extra_keys = data.meta_info.get("reward_extra_keys",[])
        #         reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
        #         return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
        #     else:
        #         return data.batch["rm_scores"]

        #  废弃单一的 reward_tensor，初始化解耦的 Tensors
        # has_real_prm = "rm_scores" in data.batch.keys()

        # # --- 方案二核心修改：仅在第一个 Batch 打印警告横幅 ---
        # if not hasattr(self, '_already_logged_mode'):
        #     print("\n" + "!"*60)
        #     if has_real_prm:
        #         print("🚀 [运行模式确认] 完整 DAPO 模式：PRM 模型已挂载！")
        #     else:
        #         print("⚠️ [消融模式确认] PRM 已物理断电！当前仅运行: 结果正确性 + 长度门控")
        #     print("!"*60 + "\n")
        #     self._already_logged_mode = True # 加上这个标记，防止每个 batch 都刷屏

        has_real_prm = False
        # GPU 传过来的分数可能是 rm_scores，也可能是 verl 默认的 reward_tensor
        if "rm_scores" in data.batch:
            real_prm_tensor = data.batch["rm_scores"]
            has_real_prm = True
        elif "reward_tensor" in data.batch:
            real_prm_tensor = data.batch["reward_tensor"]
            has_real_prm = True

        if has_real_prm:
            # 成功抢救！克隆一份，防止后面被覆盖
            prm_tensor = real_prm_tensor.clone() 
        else:
            prm_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        # 纯净版日志
        if not hasattr(self, '_already_logged_mode'):
            print("\n" + "🛡️"*20)
            if has_real_prm:
                print("🚀 [DAPO 结算中心] 已成功捕获 GPU 传回的 PRM 动态分数！SwanLab 记录开启！")
            else:
                print("⚠️ [DAPO 结算中心] 未检测到 PRM 分数，当前为物理断电模式！")
            print("🛡️"*20 + "\n")
            self._already_logged_mode = True
        
        correctness_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        length_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", {})

            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})

            extra_info["rollout_reward_scores"] = rollout_reward_scores

            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            score: float
            if isinstance(result, dict):
                score = result["score"]
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result
                reward_extra_info["acc"].append(score)

            #  不再执行相加 (reward += overlong_reward)，纯粹拆解
            overlong_reward = 0.0 # 默认长度分为 0
            if self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                
                # 算惩罚
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)

                if score <= 0:
                     # 门控：outcome错误时强行赋底线惩罚 -1.0
                     overlong_reward = -1.0

                if getattr(self.overlong_buffer_cfg, 'log', False):
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)
            
            # if has_real_prm and score <= 0:
            #     prm_tensor[i, valid_response_length - 1] = -10.0

            # 分别写入解耦的张量
            correctness_tensor[i, valid_response_length - 1] = score
            length_tensor[i, valid_response_length - 1] = overlong_reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        # 保留总分兜底，防止原框架其他日志代码报错
        total_reward_tensor = correctness_tensor + length_tensor + prm_tensor

        # 直接塞入 batch，跳过 extract_reward 的束缚
        data.batch["token_level_scores_correctness"] = correctness_tensor
        data.batch["token_level_scores_length"] = length_tensor
        data.batch["token_level_scores_prm"] = prm_tensor

        # 返回框架原本期待的字典结构
        if return_dict:
            #reward_extra_info["mean_prm"] = prm_tensor.sum(dim=-1).mean().item()
            mean_prm_val = prm_tensor.sum(dim=-1).mean().item()
            if has_real_prm:
                # 正常跑的时候，标签叫 ACTIVE
                reward_extra_info["mean_prm/ACTIVE"] = [mean_prm_val] * len(data)
            else:
                # 消融模式下，标签直接带上 DISABLED 警告，且值恒为 0
                reward_extra_info["mean_prm/DISABLED_ZERO"] = [0.0] * len(data)
            reward_extra_info["mean_prm"] = [mean_prm_val] * len(data)
            return {
                "reward_tensor": total_reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return total_reward_tensor

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# from collections import defaultdict

# import torch

# from verl import DataProto
# from verl.utils.reward_score import default_compute_score
# from verl.workers.reward_manager import register
# from verl.workers.reward_manager.abstract import AbstractRewardManager


# @register("dapo")
# class DAPORewardManager(AbstractRewardManager):
#     """The reward manager."""

#     def __init__(
#         self,
#         tokenizer,
#         num_examine,
#         compute_score=None,
#         reward_fn_key="data_source",
#         max_resp_len=None,
#         overlong_buffer_cfg=None,
#     ) -> None:
#         self.tokenizer = tokenizer
#         self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
#         self.compute_score = compute_score or default_compute_score
#         self.reward_fn_key = reward_fn_key
#         self.overlong_buffer_cfg = overlong_buffer_cfg
#         self.max_resp_len = max_resp_len

#         if self.overlong_buffer_cfg is not None:
#             assert self.max_resp_len is not None, (
#                 f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
#             )
#             assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
#                 "max_resp_len must be larger than overlong_buffer.len"
#             )

#     def __call__(self, data: DataProto, return_dict: bool = False):
#         """We will expand this function gradually based on the available datasets"""

#         # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
#         if "rm_scores" in data.batch.keys():
#             if return_dict:
#                 reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
#                 reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
#                 return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
#             else:
#                 return data.batch["rm_scores"]

#         reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
#         reward_extra_info = defaultdict(list)

#         already_print_data_sources = {}

#         for i in range(len(data)):
#             data_item = data[i]  # DataProtoItem

#             prompt_ids = data_item.batch["prompts"]

#             prompt_length = prompt_ids.shape[-1]

#             valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
#             valid_prompt_ids = prompt_ids[-valid_prompt_length:]

#             response_ids = data_item.batch["responses"]
#             valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
#             valid_response_ids = response_ids[:valid_response_length]

#             # decode
#             prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
#             response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
#             eos_token = self.tokenizer.eos_token
#             if response_str.endswith(eos_token):
#                 response_str = response_str[: -len(eos_token)]

#             ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

#             data_source = data_item.non_tensor_batch[self.reward_fn_key]

#             extra_info = data_item.non_tensor_batch.get("extra_info", {})

#             rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})

#             extra_info["rollout_reward_scores"] = rollout_reward_scores

#             result = self.compute_score(
#                 data_source=data_source,
#                 solution_str=response_str,
#                 ground_truth=ground_truth,
#                 extra_info=extra_info,
#             )

#             score: float
#             if isinstance(result, dict):
#                 score = result["score"]
#                 # Store the information including original reward
#                 for key, value in result.items():
#                     reward_extra_info[key].append(value)
#             else:
#                 score = result
#                 reward_extra_info["acc"].append(score)

#             reward = score

#             if self.overlong_buffer_cfg.enable:
#                 overlong_buffer_len = self.overlong_buffer_cfg.len
#                 expected_len = self.max_resp_len - overlong_buffer_len
#                 exceed_len = valid_response_length - expected_len
#                 overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
#                 overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
#                 reward += overlong_reward
#                 if self.overlong_buffer_cfg.log:
#                     reward_extra_info["overlong_reward"].append(overlong_reward)
#                     reward_extra_info["overlong"].append(overlong_reward < 0)

#             reward_tensor[i, valid_response_length - 1] = reward

#             if data_source not in already_print_data_sources:
#                 already_print_data_sources[data_source] = 0

#             if already_print_data_sources[data_source] < self.num_examine:
#                 already_print_data_sources[data_source] += 1
#                 print("[prompt]", prompt_str)
#                 print("[response]", response_str)
#                 print("[ground_truth]", ground_truth)
#                 if isinstance(result, dict):
#                     for key, value in result.items():
#                         print(f"[{key}]", value)
#                 else:
#                     print("[score]", score)

#         if return_dict:
#             return {
#                 "reward_tensor": reward_tensor,
#                 "reward_extra_info": reward_extra_info,
#             }
#         else:
#             return reward_tensor

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.