# Reinforcement World Model Learning

Code for [**Reinforcement World Model Learning for LLM-based Agents**](https://arxiv.org/abs/2602.05842).

RWML is a self-supervised method that teaches an LLM agent an action-conditioned world model before policy RL.
Given the interaction history $s_{\le t}$ and an action $a_t$, the model reasons and then predicts the next state $\hat{s}_{t+1}$.
GRPO then rewards predictions that are close to the real next state $s_{t+1}$ in a pre-trained embedding space (a sim-to-real gap reward).
No expert data, stronger LLMs, or task-success rewards are needed.

## Repository structure

```
rwml/
├── rewards/
│   ├── common.py            # next-state parsing, embedding similarity, batching
│   ├── alfworld.py          # ALFWorld reward
│   ├── tau2bench.py         # τ²-Bench reward
│   └── judge.py             # LLM-as-a-judge reward (ablation)
├── dataset.py               # text-only dataset class for verl
└── envs/                    # ALFWorld and τ²-Bench environments for policy RL with verl-agent
scripts/
├── serve_embedding.sh       # Qwen3-Embedding-8B server used by the rewards
├── alfworld/
│   ├── run_rwml.sh          # RWML on ALFWorld
│   └── run_policy_rl.sh     # policy RL on ALFWorld with verl-agent
├── tau2bench/
│   ├── run_rwml.sh          # RWML on τ²-Bench
│   └── run_policy_rl.sh     # policy RL on τ²-Bench with verl-agent
└── merge_fsdp_checkpoints.sh
tests/
└── test_rewards.py
```

## Installation

RWML training runs on [verl](https://github.com/volcengine/verl) v0.5.0 with vLLM rollouts.

```bash
conda create -n rwml python=3.10 -y && conda activate rwml

# verl v0.5.0, see https://verl.readthedocs.io/en/latest/start/install.html
git clone https://github.com/volcengine/verl.git
cd verl && git checkout v0.5.0 && pip install -e . && cd ..
# reference versions we used on B200 GPUs:
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install vllm==0.9.2 flash-attn==2.7.4.post1 transformers==4.51.1

# this repository
git clone <this-repo-url> rwml && cd rwml && pip install -e .
```

The `rwml` package must be installed (`pip install -e .`), because verl loads the reward files by path and they import `rwml.rewards.common`.

## Data format

Each RWML training sample is one $\langle s_{\le t}, a_t, s_{t+1}\rangle$ triplet, stored in verl's parquet format. The rewards rely on these fields:

| Column | Content |
|---|---|
| `prompt` | One user message containing the task, the interaction history $s_{\le t}$, the action $a_t$, and the instruction to reason in `<think> </think>` and to predict the next state in `<next_state> </next_state>` (prompts in Tables A4 and A6 of the paper). |
| `reward_model.ground_truth` | The real next state $s_{t+1}$, ending with the task status: `Task is not yet completed.`, `Task is successfully completed!`, or (τ²-Bench only) `Task is failed.`. On τ²-Bench the status is on its own last line. |
| `extra_info.metadata.response_type` | τ²-Bench only: `tool` or `user`, the type of the next state. |
| `data_source` | Dataset name, e.g. `alfworld` or `tau2bench`. |

By default, the scripts read `data/{alfworld,tau2bench}/{train,val}.parquet`. Set `TRAIN_FILE` and `VAL_FILE` to use other files.

## Training

### 1. Serve the embedding model

The rewards query an OpenAI-compatible embedding endpoint. Start one on GPUs that are not used for training:

```bash
bash scripts/serve_embedding.sh   # Qwen3-Embedding-8B on port 8000, served as "qwen-embedding-8b"
```

### 2. RWML

```bash
EMBED_API_BASE=http://<embedding-host>:8000/v1 bash scripts/alfworld/run_rwml.sh    # Qwen2.5-7B-Instruct, 2 GPUs
EMBED_API_BASE=http://<embedding-host>:8000/v1 bash scripts/tau2bench/run_rwml.sh   # Qwen3-8B, 4 GPUs
```

The scripts are configured through environment variables: `MODEL_PATH`, `TRAIN_FILE`, `VAL_FILE`, `N_GPUS`, `EMBED_API_BASE`, `EMBED_MODEL`, and `CKPT_DIR`.
Any extra arguments are passed to verl as Hydra overrides, e.g. `bash scripts/tau2bench/run_rwml.sh trainer.n_gpus_per_node=8`.
When training finishes, every checkpoint is converted to Hugging Face format at `<CKPT_DIR>/global_step_<N>/actor_hf`.

### 3. Policy RL

RWML is a mid-training stage. Afterwards, train the RWML checkpoint with task-success rewards using standard policy RL with [verl-agent](https://github.com/langfengQ/verl-agent).
The policy RL setup is identical to running policy RL directly on the base model; only the starting checkpoint changes.

Our ALFWorld and τ²-Bench environment implementations for verl-agent are in [`rwml/envs`](rwml/envs/README.md), together with how to plug them into verl-agent. Upstream verl-agent does not include τ²-Bench.
With them in place, run the policy RL scripts from the root of the verl-agent checkout:

```bash
MODEL_PATH=/path/to/rwml/checkpoints/rwml/alfworld-qwen2.5-7b-rwml/global_step_<N>/actor_hf \
    bash /path/to/rwml/scripts/alfworld/run_policy_rl.sh
MODEL_PATH=/path/to/rwml/checkpoints/rwml/tau2bench-qwen3-8b-rwml/global_step_<N>/actor_hf \
USER_LLM_API_BASE=http://<user-simulator-host>:<port>/v1 \
    bash /path/to/rwml/scripts/tau2bench/run_policy_rl.sh
```

## Ablations

The defaults are RWML as in the paper. The following toggles are only there to study the ablations from the paper:

| Toggle | Ablation |
|---|---|
| `REWARD=judge` or `REWARD=judge_binary` for `scripts/*/run_rwml.sh` | An LLM judge with a real-valued or binary score replaces the embedding reward (Table 4). Set `JUDGE_API_BASE` to an OpenAI-compatible endpoint serving the judge (we used Qwen3-235B-A22B-Instruct-2507). |
| `+custom_reward_function.reward_kwargs.think_in_prompt=True` for `scripts/tau2bench/run_rwml.sh` | Base models whose chat template already opens `<think>`, e.g. Qwen3-30B-A3B-Thinking-2507 (Figure 4). |

## Citation

```bibtex
@misc{yu2026reinforcementworldmodellearning,
      title={Reinforcement World Model Learning for LLM-based Agents}, 
      author={Xiao Yu and Baolin Peng and Ruize Xu and Yelong Shen and Pengcheng He and Suman Nath and Nikhil Singh and Jiangfeng Gao and Zhou Yu},
      year={2026},
      eprint={2602.05842},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.05842}, 
}
```

## Acknowledgements

This code builds on [verl](https://github.com/volcengine/verl) and [verl-agent](https://github.com/langfengQ/verl-agent), and uses the [ALFWorld](https://github.com/alfworld/alfworld) and [τ²-Bench](https://github.com/sierra-research/tau2-bench) environments.
