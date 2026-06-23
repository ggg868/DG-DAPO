#!/usr/bin/env bash

# [Original] 开启严格模式 (报错即退出，变量未定义即退出)
# set -xeuo pipefail

# [Modified] 仅开启执行打印 (-x)，去掉 -e 和 -u 以便在调试环境容错，防止因非关键错误退出
set -x

export NAS_HOME="/data/maoyan.gan"
export VERL_HOME="${NAS_HOME}/verl"
export RAY_TMPDIR="${NAS_HOME}/tmp/ray"
export SWANLAB_RESUME=must
export SWANLAB_RUN_ID="rshsxplmvcpxq8xqigow1"
project_name='DAPO'

exp_name='DAPO_Qwen2.5_3b_instruct_with_prm_5_7_nogate'

export CUDA_VISIBLE_DEVICES=0,4
GPUS_PER_NODE=2

ray stop --force || true
pkill -f ray || true
sleep 3
mkdir -p "${RAY_TMPDIR}"
chmod 700 "${RAY_TMPDIR}"
ray start --head --port=6379 --num-gpus=2 --disable-usage-stats --temp-dir=$RAY_TMPDIR
rm -rf /tmp/ray 2>/dev/null || true
export RAY_ADDRESS="192.168.25.37:6379"
python -c "import ray; ray.init(address='192.168.25.37:6379'); r=ray.available_resources(); print(r); assert r.get('GPU',0)>=2"

adv_estimator=gdpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

# [Original] Clip 范围 (DAPO 论文设置)
# clip_ratio_low=0.2
# clip_ratio_high=0.28
# [Modified] 保持不变 (这是 DAPO 的核心 Trick，不需要改)
clip_ratio_low=0.2
clip_ratio_high=0.28


max_prompt_length=$((1024 * 2)) 
max_response_length=$((1024 * 8))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=True

filter_groups_metric=acc

max_num_gen_batches=10


train_prompt_bsz=128 
gen_prompt_bsz=128
n_resp_per_prompt=16     
train_prompt_mini_bsz=64 

NNODES=1

MODEL_PATH="/data/maoyan.gan/my_verl/models/Qwen2.5-3B-Instruct"
PRM_MODEL_PATH="/data/maoyan.gan/.cache/modelscope/hub/models/Qwen/Qwen2.5-Math-PRM-7B"
CKPTS_DIR="${VERL_HOME}/checkpoints/${project_name}/${exp_name}"
TRAIN_FILE="${VERL_HOME}/data/dapo_math_17k.parquet"
TEST_FILE="${VERL_HOME}/data/aime_2024.parquet"

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 
val_top_p=0.7

sp_size=1

use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=False

gen_tp=1

# ========================================================================
# 6. 启动命令 (Execution Command)
# ========================================================================

# [Original] Ray 提交模式 (适合集群)
# ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
#     --working-dir "${WORKING_DIR}" \
#     -- python3 -m recipe.dapo.main_dapo \
#     ... (参数略) ...

# [Modified] Python 直连模式 (适合单机调试，更稳定)
python3 -m recipe.dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=source_prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    +trainer.mixed_precision=bf16 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((3 *(max_prompt_length + max_response_length))) \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((2 *(max_prompt_length + max_response_length))) \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((2 *(max_prompt_length + max_response_length))) \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=131072 \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    reward_model.n_gpus_per_node=2 \
    reward_model.enable=True \
    reward_model.strategy=fsdp \
    reward_model.model.path="${PRM_MODEL_PATH}" \
    reward_model.model.fsdp_config.fsdp_size=2 \
    +reward_model.model.fsdp_config.model_dtype=bfloat16 \
    reward_model.micro_batch_size_per_gpu=8 \
    reward_model.model.use_remove_padding=False \
    reward_model.enable_resource_pool=False \
    trainer.logger='["console", "swanlab"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.balance_batch=False \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node=2 \
    trainer.val_before_train=True \
    trainer.test_freq=20 \
    trainer.save_freq=20 \
    trainer.total_epochs=8 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    trainer.device=cuda
    