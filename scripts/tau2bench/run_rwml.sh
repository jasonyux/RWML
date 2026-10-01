#!/usr/bin/env bash
# RWML on τ²-Bench with Qwen3-8B on 4 GPUs
# Requires an embedding server for the reward, see scripts/serve_embedding.sh.
# Extra arguments are passed to verl as Hydra overrides, e.g. trainer.n_gpus_per_node=8
set -xeuo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

TRAIN_FILE=${TRAIN_FILE:-$REPO_DIR/data/tau2bench/train.parquet}
VAL_FILE=${VAL_FILE:-$REPO_DIR/data/tau2bench/val.parquet}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
N_GPUS=${N_GPUS:-4}

EMBED_API_BASE=${EMBED_API_BASE:-http://localhost:8000/v1}
EMBED_API_KEY=${EMBED_API_KEY:-EMPTY}
EMBED_MODEL=${EMBED_MODEL:-qwen-embedding-8b}

# REWARD=embedding is RWML. For ablations only (Table 4): REWARD=judge or REWARD=judge_binary uses an LLM judge
# with a real-valued or binary score instead, served at JUDGE_API_BASE (we used Qwen3-235B-A22B-Instruct-2507).
REWARD=${REWARD:-embedding}
case "$REWARD" in
    embedding)
        exp_suffix=""
        reward_args=(
            custom_reward_function.path="$REPO_DIR/rwml/rewards/tau2bench.py"
            +custom_reward_function.reward_kwargs.embed_api_base="$EMBED_API_BASE"
            +custom_reward_function.reward_kwargs.embed_api_key="$EMBED_API_KEY"
            +custom_reward_function.reward_kwargs.embed_model="$EMBED_MODEL"
            +custom_reward_function.reward_kwargs.sim_threshold=0.6
            +custom_reward_function.reward_kwargs.num_workers=4
        ) ;;
    judge | judge_binary)
        exp_suffix="-$REWARD"
        reward_args=(
            custom_reward_function.path="$REPO_DIR/rwml/rewards/judge.py"
            +custom_reward_function.reward_kwargs.judge_api_base="${JUDGE_API_BASE:?set JUDGE_API_BASE for the judge reward}"
            +custom_reward_function.reward_kwargs.judge_api_key="${JUDGE_API_KEY:-EMPTY}"
            +custom_reward_function.reward_kwargs.judge_model="${JUDGE_MODEL:-Qwen3-235B-A22B-Instruct-2507}"
            +custom_reward_function.reward_kwargs.binary=$([ "$REWARD" = judge_binary ] && echo True || echo False)
            +custom_reward_function.reward_kwargs.max_completion_tokens=4096
            +custom_reward_function.reward_kwargs.num_workers=8
        ) ;;
    *)
        echo "Unknown REWARD=$REWARD (expected embedding, judge, or judge_binary)" >&2
        exit 1 ;;
esac

PROJECT_NAME=${PROJECT_NAME:-rwml}
EXP_NAME=${EXP_NAME:-tau2bench-qwen3-8b-rwml$exp_suffix}
CKPT_DIR=${CKPT_DIR:-$REPO_DIR/checkpoints/$PROJECT_NAME/$EXP_NAME}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.custom_cls.path="$REPO_DIR/rwml/dataset.py" \
    data.custom_cls.name=TextOnlyRLHFDataset \
    data.train_batch_size=32 \
    data.max_prompt_length=10240 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    reward_model.reward_manager=batch \
    custom_reward_function.name=compute_score_batched \
    "${reward_args[@]}" \
    +custom_reward_function.reward_kwargs.max_prediction_tokens=512 \
    trainer.critic_warmup=0 \
    trainer.logger='[console,wandb]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.total_epochs=2 \
    trainer.save_freq=500 \
    trainer.test_freq=20 \
    trainer.log_val_generations=10 \
    trainer.default_local_dir="$CKPT_DIR" \
    "$@"

bash "$REPO_DIR/scripts/merge_fsdp_checkpoints.sh" "$CKPT_DIR"
