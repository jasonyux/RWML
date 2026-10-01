#!/usr/bin/env bash
# Policy RL (task-success reward) on ALFWorld starting from an RWML checkpoint
# Requires verl-agent with our environments plugged in (see rwml/envs/README.md).
# Run from the root of that verl-agent checkout, e.g.
#   MODEL_PATH=/path/to/rwml/checkpoint/actor_hf bash /path/to/rwml/scripts/alfworld/run_policy_rl.sh
# Extra arguments are passed to verl-agent as Hydra overrides.
set -xeuo pipefail

MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the RWML checkpoint in Hugging Face format}
ENGINE=${ENGINE:-vllm}
N_GPUS=${N_GPUS:-2}
DATA_DIR=${DATA_DIR:-$HOME/data/rwml}

PROJECT_NAME=${PROJECT_NAME:-rwml}
EXP_NAME=${EXP_NAME:-alfworld-qwen2.5-7b-rwml-policy_rl}

train_data_size=8
val_data_size=64
group_size=8

python3 -m rwml.envs.prepare_data \
    --env alfworld \
    --train_size $train_data_size \
    --val_size $val_data_size \
    --output_dir "$DATA_DIR"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$DATA_DIR/alfworld/train.parquet" \
    data.val_files="$DATA_DIR/alfworld/test.parquet" \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=15 \
    env.history_length=2 \
    env.rollout.n=$group_size \
    env.alfworld.eval_dataset=eval_in_distribution \
    +env.randomize_reset_seed=True \
    trainer.critic_warmup=0 \
    trainer.logger='[console,wandb]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.total_epochs=300 \
    trainer.save_freq=150 \
    trainer.test_freq=20 \
    trainer.val_before_train=True \
    "$@"
