# DG-DAPO

**DG-DAPO** is a process-guided reinforcement learning framework for improving the mathematical reasoning ability of large language models. This repository is built on top of the `verl` framework and modifies the original DAPO training pipeline by introducing **decoupled advantage estimation** and a **dual outcome-gated reward mechanism**.

The goal of DG-DAPO is to reduce reward interference among heterogeneous reward signals, such as outcome correctness, response length, and process-level rewards, while mitigating reward hacking caused by misaligned auxiliary rewards.

## Overview

Recent reinforcement learning methods for large language model reasoning often combine multiple reward signals, including final-answer correctness, length constraints, and process rewards. However, directly aggregating these heterogeneous rewards may introduce several problems:

* reward scale mismatch between different reward types;
* verbose reasoning caused by process reward exploitation;
* incorrect trajectories being reinforced by locally reasonable process rewards.

DG-DAPO addresses these issues through two main components:

1. **Decoupled Advantage Estimation**

   DG-DAPO computes advantages for different reward dimensions separately, including correctness reward, length reward, and process reward. These advantages are normalized independently before being combined, which reduces scale interference among heterogeneous reward signals.

2. **Dual Outcome-Gated Mechanism**

   DG-DAPO conditions both dense process rewards and soft length penalties on final-answer correctness. Auxiliary rewards are encouraged to guide optimization only when they are aligned with correct outcomes.

   * For correct responses, the model receives normal process rewards and soft length penalties.
   * For incorrect responses, the process reward is suppressed and the length penalty is strengthened.

## Method

Given a response sampled from the policy model, DG-DAPO decomposes the reward into three parts:

* `R_correctness`: final-answer correctness reward;
* `R_length`: soft length control reward;
* `R_prm`: process reward produced by a Process Reward Model.

Instead of directly summing raw rewards, DG-DAPO estimates advantages separately:

```text
A_correctness = Normalize(R_correctness)
A_length      = Normalize(R_length)
A_prm         = Normalize(R_prm)
```

Then the final advantage is computed as:

```text
A_DG-DAPO = A_correctness + A_length + A_prm
```

An optional masked whitening operation is applied to stabilize optimization:

```text
A_final = MaskedWhiten(A_DG-DAPO)
```

The process reward is computed using a PRM model. In this implementation, the PRM scores each reasoning step and aggregates step-level scores into a trajectory-level process reward.

## Main Features

* Based on the `verl` reinforcement learning framework.
* Built upon the DAPO training pipeline.
* Supports decoupled reward and advantage computation.
* Supports correctness reward, length reward, and PRM-based process reward.
* Supports outcome-gated process reward.
* Supports outcome-gated length penalty.
* Supports PRM scoring with Qwen2.5-Math-PRM style models.
* Designed for mathematical reasoning tasks.

## Repository Structure

```text
DG-DAPO/
├── verl/                         # Modified verl framework
├── recipe/                       # Training recipes and configuration files
├── examples/                     # Example training scripts
├── scripts/                      # Launch scripts
├── dapo.py                       # Modified DAPO reward processing logic
├── core_algos.py                 # Advantage estimation algorithms
├── prm_scorer.py                 # Process Reward Model scoring module
├── README.md
└── requirements.txt
```

> Note: The actual file names may differ depending on your local implementation. Please refer to the corresponding scripts and configuration files in this repository.

## Installation

Clone this repository:

```bash
git clone https://github.com/ggg868/DG-DAPO.git
cd DG-DAPO
```

Create a Python environment:

```bash
conda create -n dg-dapo python=3.10
conda activate dg-dapo
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the package in editable mode:

```bash
pip install -e .
```

## Model Preparation

DG-DAPO requires a policy model and, optionally, a Process Reward Model.

Example policy models:

```text
Qwen2.5-7B-Insturct
Qwen2.5-3B-Instruct
```

Example PRM model:

```text
Qwen/Qwen2.5-Math-PRM-7B
```

Please download the models in advance or configure the model paths in the training script.

## Data Preparation

Prepare mathematical reasoning datasets in the format required by `verl`. The training data should contain prompts and reference answers for reward computation.

A sample data item may look like:

```json
{
  "prompt": "Solve the problem step by step...",
  "answer": "..."
}
```

## Training

An example training command is shown below:

```bash
bash scripts/train_dg_dapo.sh
```

Or launch training manually:

```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=dg-dapo \
    reward_model.enable_prm=True \
    data.train_files=PATH_TO_TRAIN_DATA \
    data.val_files=PATH_TO_VALIDATION_DATA \
    actor_rollout_ref.model.path=PATH_TO_POLICY_MODEL \
    reward_model.model.path=PATH_TO_PRM_MODEL \
    trainer.project_name=DG-DAPO \
    trainer.experiment_name=dg_dapo_math
```

Please adjust the configuration according to your hardware environment and model size.

## Key Configuration

Important configuration options include:

```yaml
algorithm:
  adv_estimator: dg-dapo

reward_model:
  enable_prm: true
  prm_model_path: Qwen/Qwen2.5-Math-PRM-7B
  aggregation_method: log_mean_prob
  clip_epsilon: 1e-4

dg_dapo:
  use_decoupled_advantage: true
  use_outcome_gated_prm: true
  use_outcome_gated_length_penalty: true
  prm_clip_min: -5.0
  prm_clip_max: 0.0
```

## Evaluation

After training, evaluate the model on mathematical reasoning benchmarks:

```bash
bash scripts/eval_math.sh
```

Example evaluation metrics:

* accuracy;
* average response length;
* PRM score;
* training reward curve;
* policy loss curve;
* KL divergence;


## Notes

This repository is a research implementation. The current implementation is mainly designed for mathematical reasoning tasks and may require additional adaptation for other domains.

The PRM reward is trajectory-level after aggregation. If it is used in token-level reward computation, it should be placed on the final valid response token rather than expanded to all tokens, in order to avoid introducing length bias.

## Acknowledgements

This project is based on the following open-source projects and research directions:

* `verl`: a flexible reinforcement learning framework for large language models;
* DAPO-style reinforcement learning for mathematical reasoning;
* process reward models for step-by-step reasoning supervision.

