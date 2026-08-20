#!/bin/bash
# =============================================================================
# Smoke test for Qwen3 MaxRL training.
# Derived from qwen3_experiments/run_qwen3_training.sh — same python command,
# same parameter order; ONLY the following were changed:
#   - placeholder dataset/model/checkpoint paths -> real local paths
#   - scaled down for a single-GPU smoke run:
#       FULL_BATCH_SIZE             256 -> 8
#       PPO_MINI_BATCH_SIZE         256 -> 4
#       PER_GPU_MINI_BATCH_SIZE     4   -> 1
#       NUM_PER_PROMPT_ROLLOUTS     16  -> 2
#       NUM_PER_PROMPT_ROLLOUTS_VALIDATION 32 -> 2
#       MAX_PROMPT_LENGTH           1024 -> 512
#       MAX_RESPONSE_LENGTH         4096 -> 512
#       MAX_MODEL_LEN               32000 -> 2048
#       MAX_NUM_BATCHED_TOKENS      32000 -> 2048
#       MAX_TRAJECTORY_LENGTH       3900 -> 1024
#       TOTAL_EPOCHS                5    -> 1
#       trainer.n_gpus_per_node     8    -> 1
#       trainer.nnodes              4    -> 1
#       trainer.test_freq           50   -> 5
#   - Ray/SLURM cluster preamble omitted (run in Ray local mode)
#   - WANDB_MODE=offline so the smoke run does not require wandb login
# The SLURM part of the original script is cluster-specific and unchanged in
# run_qwen3_training.sh.
# =============================================================================

# Use the maxrl conda env (python 3.10)
source /Users/shekuigang/anaconda3/etc/profile.d/conda.sh
conda activate maxrl

export FULL_BATCH_SIZE=8
export PPO_MINI_BATCH_SIZE=4

# Number of rollouts
export NUM_PER_PROMPT_ROLLOUTS=2

# prompt and response length cutoff
export MAX_RESPONSE_LENGTH=512
export MAX_PROMPT_LENGTH=512

# Other hyperparameters
export LEARNING_RATE=1e-6

# RL with ground truth hyperparams
export REWARD_MANAGER='multi_thread'

PER_GPU_MINI_BATCH_SIZE=1
NUM_PER_PROMPT_ROLLOUTS_VALIDATION=2
MAX_MODEL_LEN=2048
MAX_NUM_BATCHED_TOKENS=2048
MAX_TRAJECTORY_LENGTH=1024
TENSOR_MODEL_PARALLEL_SIZE=1

PPO_EPOCHS=1

CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.2
GRAD_CLIP=0.3

# KL coefficient (set to 0.0 since use_kl_loss=False)
KL_COEFF=0.0

echo "This is the per-GPU mini batch size: $PER_GPU_MINI_BATCH_SIZE"
echo "This is the Maximum response length: $MAX_RESPONSE_LENGTH"

# Smoke-test slices (16 train / 4+4 val samples), see code_note.md
TRAIN_DATASET_PATH=/Users/shekuigang/Projects/maxrl/data/polaris_smoke/train.parquet
TEST_DATASET_PATH="['/Users/shekuigang/Projects/maxrl/data/aime25_smoke/test.parquet','/Users/shekuigang/Projects/maxrl/data/math_smoke/test.parquet']"

TOTAL_EPOCHS=1

# Set model path (local copy of Qwen/Qwen3-1.7B-Base)
MODEL_PATH=/Users/shekuigang/Projects/maxrl/models/Qwen3-1.7B-Base
MODEL_NAME=Qwen3-1.7B-Base

ADVANTAGE_ESTIMATOR=maxrl

PROJECT_NAME=Qwen3_MaxRL_Experiments
EXPERIMENT_NAME=${ADVANTAGE_ESTIMATOR}_${MODEL_NAME}_smoke

CHECKPOINT_SAVE_PATH=/Users/shekuigang/Projects/maxrl/checkpoints/${EXPERIMENT_NAME}

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export SEED=79
export WANDB_MODE=offline

python3 -W ignore -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$ADVANTAGE_ESTIMATOR \
    data.train_files=$TRAIN_DATASET_PATH \
    data.val_files=$TEST_DATASET_PATH \
    data.train_batch_size=$FULL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PER_GPU_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEFF \
    actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO_LOW \
    actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
    actor_rollout_ref.actor.grad_clip=$GRAD_CLIP \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.ppo_epochs=$PPO_EPOCHS \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$PER_GPU_MINI_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=$NUM_PER_PROMPT_ROLLOUTS \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$PER_GPU_MINI_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.val_kwargs.n=$NUM_PER_PROMPT_ROLLOUTS_VALIDATION \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_penalty=low_var_kl \
    algorithm.kl_ctrl.kl_coef=$KL_COEFF \
    reward_model.reward_manager=$REWARD_MANAGER \
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.val_only=False \
    trainer.val_on_last_step=True \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CHECKPOINT_SAVE_PATH \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.max_actor_ckpt_to_keep=400 \
    trainer.max_critic_ckpt_to_keep=400 \
    trainer.test_freq=5 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    ray_init.ray_dir=/tmp/ray $@
