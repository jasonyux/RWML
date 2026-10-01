#!/usr/bin/env bash
# Policy RL (task-success reward) on τ²-Bench starting from an RWML checkpoint
# Requires verl-agent with our environments plugged in, and tau2-bench (see rwml/envs/README.md).
# Run from the root of that verl-agent checkout, e.g.
#   MODEL_PATH=/path/to/rwml/checkpoint/actor_hf USER_LLM_API_BASE=http://<host>:<port>/v1 \
#       bash /path/to/rwml/scripts/tau2bench/run_policy_rl.sh
# Extra arguments are passed to verl-agent as Hydra overrides.
set -xeuo pipefail

MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the RWML checkpoint in Hugging Face format}
ENGINE=${ENGINE:-vllm}
N_GPUS=${N_GPUS:-8}
DATA_DIR=${DATA_DIR:-$HOME/data/rwml}

# User simulator, called through LiteLLM (e.g. Qwen3-235B-A22B-Instruct-2507 served with vLLM).
USER_LLM=${USER_LLM:-hosted_vllm/Qwen3-235B-A22B-Instruct-2507}
USER_LLM_API_BASE=${USER_LLM_API_BASE:?set USER_LLM_API_BASE to the user simulator endpoint}
USER_LLM_API_KEY=${USER_LLM_API_KEY:-EMPTY}

PROJECT_NAME=${PROJECT_NAME:-rwml}
EXP_NAME=${EXP_NAME:-tau2bench-qwen3-8b-rwml-policy_rl}

train_data_size=8
val_data_size=64
group_size=8
max_prompt_length=16384
max_response_length=8192

python3 -m rwml.envs.prepare_data \
    --env tau2bench \
    --train_size $train_data_size \
    --val_size $val_data_size \
    --output_dir "$DATA_DIR"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$DATA_DIR/tau2bench/train.parquet" \
    data.val_files="$DATA_DIR/tau2bench/test.parquet" \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
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
    actor_rollout_ref.rollout.max_num_batched_tokens=$((16 * (max_prompt_length + max_response_length))) \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    env.env_name=tau2bench \
    env.seed=0 \
    env.max_steps=30 \
    env.history_length=5 \
    env.rollout.n=$group_size \
    +env.randomize_reset_seed=True \
    +env.tau2bench.user_llm="$USER_LLM" \
    +env.tau2bench.user_llm_args.api_base="$USER_LLM_API_BASE" \
    +env.tau2bench.user_llm_args.api_key="$USER_LLM_API_KEY" \
    +env.tau2bench.user_llm_args.temperature=0.0 \
    +env.tau2bench.user_llm_args.max_completion_tokens=512 \
    +env.tau2bench.task_set_name=all \
    +env.tau2bench.task_split_name=train \
    +env.tau2bench.test_task_set_name=all \
    +env.tau2bench.test_task_split_name=test \
    +env.tau2bench.max_concurrency=8 \
    trainer.critic_warmup=0 \
    trainer.logger='[console,wandb]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.total_epochs=200 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.val_before_train=True \
    "$@"
