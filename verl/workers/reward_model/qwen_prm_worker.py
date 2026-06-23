import os
import warnings
import torch
import torch.nn.functional as F
import numpy as np

from transformers import AutoConfig, AutoModelForTokenClassification
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import CPUOffload

from verl import DataProto
# from verl.workers.fsdp_workers import RewardModelWorker
# from verl.utils.hf_accelerate import copy_to_local
# from verl.utils.tokenizer import hf_tokenizer
from verl.workers.fsdp_workers import RewardModelWorker, get_sharding_strategy
from verl.utils.fs import copy_to_local
from verl.utils import hf_tokenizer
from verl.utils.fsdp_utils import get_init_weight_context_manager, get_fsdp_wrap_policy, init_fn
# from verl.utils.fsdp_utils import get_init_weight_context_manager, get_fsdp_wrap_policy, get_sharding_strategy, init_fn
# from verl.models.registry import apply_monkey_patch
from verl.models.transformers.monkey_patch import apply_monkey_patch
#from verl.utils.torch_functional import get_device_id, compute_position_id_with_mask
from verl.utils.device import get_device_id
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F

try:
    from torch.distributed.fsdp import CPUOffloadPolicy
    from verl.utils.fsdp_utils import apply_fsdp2, fsdp2_load_full_state_dict, get_shard_placement_fn
except ImportError:
    pass

class QwenPRMFSDPWorker(RewardModelWorker):
    """
    继承官方 FSDP Worker，重写底层以支持 Qwen-PRM 的步级概率提取与对数几何聚合
    """

    def _build_model(self, config):
        """使用 AutoModel 替换 AutoModelForTokenClassification"""
        use_shm = config.model.get("use_shm", False)
        local_path = copy_to_local(config.model.path, use_shm=use_shm)

        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_to_local(config.model.input_tokenizer, use_shm=use_shm)
            self.input_tokenizer = hf_tokenizer(
                input_tokenizer_local_path, trust_remote_code=config.model.get("trust_remote_code", False)
            )
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get("trust_remote_code", False))
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            # 提取并缓存 step_sep_id
            self.step_sep_id = self.tokenizer.encode("<extra_0>", add_special_tokens=False)[0]

        trust_remote_code = config.model.get("trust_remote_code", False)
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        
        # 输出维度为 2 的 logits
        model_config.num_labels = 2 

        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_config.classifier_dropout = 0.0
            
            # 使用 AutoModel 加载
            # reward_module = AutoModel.from_pretrained(
            #     pretrained_model_name_or_path=local_path,
            #     config=model_config,
            #     torch_dtype=torch.bfloat16,
            #     attn_implementation="flash_attention_2",
            #     trust_remote_code=trust_remote_code,
            # )
            reward_module = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name_or_path=local_path,
                config=model_config,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            apply_monkey_patch(
                model=reward_module,
                use_remove_padding=config.model.get("use_remove_padding", False),
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
            )
            reward_module.to(torch.bfloat16)

        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)
        sharding_strategy = get_sharding_strategy(self.device_mesh)

        if config.strategy == "fsdp":
            reward_module = FSDP(
                reward_module,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,
                sync_module_states=True,
                cpu_offload=CPUOffload(offload_params=True), # 救命的卸载
                forward_prefetch=self.config.model.fsdp_config.forward_prefetch,
                device_mesh=self.device_mesh,
            )
        elif config.strategy == "fsdp2":
            cpu_offload = CPUOffloadPolicy(pin_memory=True)
            fsdp_kwargs = {
                "mesh": self.device_mesh,
                "offload_policy": cpu_offload,
                "reshard_after_forward": config.model.fsdp_config.reshard_after_forward,
                "shard_placement_fn": get_shard_placement_fn(fsdp_size=self.device_mesh.shape[-1]),
            }
            full_state = reward_module.state_dict()
            apply_fsdp2(reward_module, fsdp_kwargs, config.model.fsdp_config)
            fsdp2_load_full_state_dict(reward_module, full_state, self.device_mesh, cpu_offload)
        return reward_module

    # def _switch_chat_template(self, data: DataProto):
    #     """在转换模板时，动态注入 <extra_0> 标签"""
    #     src_max_length = data.batch["attention_mask"].shape[-1]
    #     src_tokenizer = self.input_tokenizer
    #     target_tokenizer = self.tokenizer

    #     rm_input_ids =[]
    #     rm_attention_mask = []

    #     for i in range(data.batch.batch_size[0]):
    #         chat: list = list(data.non_tensor_batch["raw_prompt"][i])

    #         response_ids = data.batch["responses"][i]
    #         response_length = response_ids.shape[-1]
    #         valid_response_length = data.batch["attention_mask"][i][-response_length:].sum()
    #         valid_response_ids = response_ids[:valid_response_length]

    #         response = src_tokenizer.decode(valid_response_ids)
    #         response = response.replace(src_tokenizer.eos_token, "")
            
    #         steps = [s.strip() for s in response.split("\n\n") if s.strip()]
    #         # 容错防御：如果模型完全没分步（一长段到底），才退化为按句号强行切分
    #         if len(steps) < 2:
    #             steps = [s.strip() for s in response.replace(". ", ".\n\n").split("\n\n") if s.strip()]
    #         assistant_content = "<extra_0>".join(steps) + "<extra_0>" if steps else ""

    #         # 强行纠正系统提示词
    #         # if chat[0]["role"] == "system":
    #         #     chat[0]["content"] = "Please reason step by step, and put your final answer within \\boxed{}."
    #         if len(chat) > 0 and chat[0]["role"] == "system":
    #             # 创建一个全新的字典，覆盖原来的引用
    #             chat[0] = {**chat[0], "content": "Please reason step by step, and put your final answer within \\boxed{}."}
    #         else:
    #             chat.insert(0, {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."})

    #         chat.append({"role": "assistant", "content": assistant_content})

    #         prompt_with_chat_template = target_tokenizer.apply_chat_template(chat, add_generation_prompt=False, tokenize=False)
    #         max_length = self.config.get("max_length", src_max_length) or src_max_length

    #         model_inputs = target_tokenizer(prompt_with_chat_template, return_tensors="pt", add_special_tokens=False)
    #         input_ids, attention_mask = verl_F.postprocess_data(
    #             input_ids=model_inputs["input_ids"],
    #             attention_mask=model_inputs["attention_mask"],
    #             max_length=max_length,
    #             pad_token_id=target_tokenizer.pad_token_id,
    #             left_pad=False,
    #             truncation=self.config.get("truncation", "right"),
    #         )

    #         rm_input_ids.append(input_ids)
    #         rm_attention_mask.append(attention_mask)

    #     rm_input_ids = torch.cat(rm_input_ids, dim=0)
    #     rm_attention_mask = torch.cat(rm_attention_mask, dim=0)
    #     rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

    #     rm_inputs = {"input_ids": rm_input_ids, "attention_mask": rm_attention_mask, "position_ids": rm_position_ids}
    #     return DataProto.from_dict(rm_inputs)
    
    def _switch_chat_template(self, data: DataProto):
        """重写：直接从张量解码 Prompt，动态注入 <extra_0> 标签，彻底摆脱 KeyError"""
        src_max_length = data.batch["attention_mask"].shape[-1]
        src_tokenizer = self.input_tokenizer if hasattr(self, 'input_tokenizer') else self.tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            # 🚀 1. 直接从张量解码 Prompt，不再依赖 non_tensor_batch！
            # prompt_ids = data.batch["prompts"][i]
            # p_len = prompt_ids.shape[-1]
            # v_p_len = data.batch["attention_mask"][i, :p_len].sum().item()
            # # 提取真实有效的 prompt token 并解码
            # prompt_str = src_tokenizer.decode(prompt_ids[-v_p_len:], skip_special_tokens=True)
            # 🚀 1. 动态安全获取纯净的 Prompt，拒绝 Tensor 反解码造成的格式污染！
            prompt_str = ""
            available_keys = list(data.non_tensor_batch.keys())
            
            # 依次尝试命中各种可能的 prompt 键名（你在脚本里设了 source_prompt，这里一定能命中！）
            for k in ["source_prompt", "prompt", "raw_prompt", "question", "query"]:
                if k in available_keys:
                    raw_item = data.non_tensor_batch[k][i]
                    if isinstance(raw_item, str):
                        prompt_str = raw_item
                    elif isinstance(raw_item, list):
                        # 如果存的是对话列表，精准提取 user 发送的内容
                        prompt_str = next((msg["content"] for msg in raw_item if msg.get("role") == "user"), str(raw_item))
                    elif isinstance(raw_item, dict) and "content" in raw_item:
                        prompt_str = raw_item["content"]
                    break
            
            # 终极兜底，哪怕天塌下来模型也能看到正常的文字
            if not prompt_str:
                prompt_str = "Please solve the problem step by step."
            # 2. 解码 Response
            response_ids = data.batch["responses"][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            response_str = src_tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            response_str = response_str.replace(src_tokenizer.eos_token, "")
            
            # 3. 注入 <extra_0>
            steps = [s.strip() for s in response_str.split("\n\n") if s.strip()]
            if len(steps) < 2:
                steps = [s.strip() for s in response_str.replace(". ", ".\n\n").split("\n\n") if s.strip()]
            assistant_content = "<extra_0>".join(steps) + "<extra_0>" if steps else ""

            # 🚀 4. 手动组装纯净的 Chat 模板，彻底无视原来的结构
            chat = [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": prompt_str},
                {"role": "assistant", "content": assistant_content}
            ]

            prompt_with_chat_template = target_tokenizer.apply_chat_template(chat, add_generation_prompt=False, tokenize=False)
            max_length = self.config.get("max_length", src_max_length) or src_max_length

            model_inputs = target_tokenizer(prompt_with_chat_template, return_tensors="pt", add_special_tokens=False)
            
            import verl.utils.torch_functional as verl_F
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,
                truncation=self.config.get("truncation", "right"),
            )

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)
        
        from verl.utils.model import compute_position_id_with_mask
        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        rm_inputs = {"input_ids": rm_input_ids, "attention_mask": rm_attention_mask, "position_ids": rm_position_ids}
        return DataProto.from_dict(rm_inputs)

    def _forward_micro_batch(self, micro_batch):
        """提取 Logits，利用布尔掩码计算对数几何平均"""
        from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
        from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                # 兼容 remove_padding 加速
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                if position_ids.dim() == 3:
                    position_ids_rmpad = (index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1))
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
                    )

                output = self.reward_module(input_ids=input_ids_rmpad, attention_mask=None, position_ids=position_ids_rmpad, use_cache=False)
                
                # 提取 Logits [total_nnz, 2]
                # logits_rmpad = output[0] if isinstance(output, tuple) else output.logits
                if hasattr(output, 'logits'):
                    logits_rmpad = output.logits
                elif isinstance(output, (list, tuple)):
                    logits_rmpad = output[0]
                else:
                    logits_rmpad = getattr(output, 'last_hidden_state', None)
        
                logits_rmpad = logits_rmpad.squeeze(0)

                if self.ulysses_sequence_parallel_size > 1:
                    logits_rmpad = gather_outputs_and_unpad(logits_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # 还原 Padding: [B, S, 2]
                logits = pad_input(logits_rmpad, indices=indices, batch=batch_size, seqlen=seqlen)
            else:
                output = self.reward_module(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
                # logits = output[0] if isinstance(output, tuple) else output.logits
                if hasattr(output, 'logits'):
                    logits = output.logits
                elif isinstance(output, (list, tuple)):
                    logits = output[0]
                else:
                    # 最后的保底手段，尝试从对象中直接抓取
                    logits = getattr(output, 'last_hidden_state', None)

            probabilities = F.softmax(logits, dim=-1)
            token_masks = (input_ids == getattr(self, "step_sep_id", 151646)) 

            # trajectory_scores =[]
            # for i in range(batch_size):
            #     step_probs = probabilities[i][token_masks[i]]
            #     if step_probs.size(0) > 0:
            #         good_probs = step_probs[:, 1].cpu().numpy()
            #         arr = np.clip(good_probs, 1e-4, 1.0)
            #         final_score = float(np.mean(np.log(arr)))
            #         final_score = max(final_score, -50.0)
            #     else:
            #         # 空步骤重罚，直接在 GPU 上生成标量
            #         final_score = torch.tensor(-50.0, dtype=torch.float32, device=probabilities.device)
            #     trajectory_scores.append(final_score)
            
            # # 返回形状为 (B,) 的标量，给外层的 _expand_to_token_level 处理
            # # 外层会自动把这个标量放到句子的最后一个 Token 上！
            # return torch.tensor(trajectory_scores, dtype=torch.float32, device=input_ids.device)
            
            mask_float = token_masks.float()
            good_probs = probabilities[:, :, 1]
            arr = torch.clamp(good_probs, min=1e-4, max=1.0)
            log_probs = torch.log(arr)
            masked_log_probs = log_probs * mask_float         
            sum_log_probs = masked_log_probs.sum(dim=-1)
            valid_counts = mask_float.sum(dim=-1)
            mean_log_probs = sum_log_probs / torch.clamp(valid_counts, min=1.0)
            final_scores = torch.where(
                valid_counts > 0, 
                mean_log_probs, 
                torch.tensor(-50.0, dtype=mean_log_probs.dtype, device=mean_log_probs.device)
            )
            final_scores = torch.clamp(final_scores, min=-50.0)

            # --- 🚀 [PRM 实时监控日志] ---
            # 只有在第一张卡的第一个微批次打印，防止刷屏
            if self.rank == 0:
                # 随机抓取当前 batch 中的第 0 条数据看一眼
                sample_idx = 0
                sample_valid_count = int(valid_counts[sample_idx].item())
                sample_final_score = final_scores[sample_idx].item()
                print(f"✅ 当前 Sample 抓取到步骤标记 (<extra_0>) 数量: {sample_valid_count}")
                # 如果抓到了分数，打印出来看一眼趋势
                if sample_valid_count > 0:
                    print(f"📝 对数几何平均分 (Log-Mean Score): {sample_final_score:.4f}")
                    # 转换回概率值直观感受一下（Exp后）
                    print(f"📈 换算为原始平均概率: {np.exp(sample_final_score):.2%}")
                else:
                    print("⚠️ 警告：当前样本未抓取到任何有效步骤，请检查 Chat Template 是否注入成功！")
                
                # 打印整个 Batch 的统计信息
                print(f"🌐 Batch 统计: 均值={final_scores.mean():.4f}, 最大={final_scores.max():.4f}, 最小={final_scores.min():.4f}")
                print("🔍" * 30 + "\n")
            # ------------------------# --- 🚀 [PRM 深度 X光扫描日志] ---
            # 只有在第一张卡的第一个微批次打印，防止刷屏
            # if self.rank == 0 and not hasattr(self, '_already_printed_prm_xray'):
            #     sample_idx = 0
            #     sample_valid_count = int(valid_counts[sample_idx].item())
            #     sample_final_score = final_scores[sample_idx].item()
                
            #     if sample_valid_count > 0:
            #         print(f"📉 对数几何平均分: {sample_final_score:.4f} (概率: {np.exp(sample_final_score):.2%})")
            #         # 扒出第一个有效步骤的原始 Logits
            #         try:
            #             first_step_idx = (token_masks[sample_idx]).nonzero(as_tuple=True)[0][0]
            #             raw_logit = logits[sample_idx, first_step_idx].tolist()
            #             print(f"🧮 [第 1 步的原始 Logits] 类别 0 (Bad): {raw_logit[0]:.4f}, 类别 1 (Good): {raw_logit[1]:.4f}")
            #         except Exception as e:
            #             pass
                
            #     print(f"🌐 Batch 统计: 均值={final_scores.mean():.4f}, 最大={final_scores.max():.4f}, 最小={final_scores.min():.4f}")
                # self._already_printed_prm_xray = True
            # ------------------------------

            return final_scores.to(torch.float32)