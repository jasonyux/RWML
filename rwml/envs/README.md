# Environments for policy RL

The ALFWorld and τ²-Bench environments we used for policy RL with [verl-agent](https://github.com/langfengQ/verl-agent).
They also record full trajectories (`render_trajectories()`), which is how we collected the rollouts for RWML.

```
rwml/envs/
├── alfworld/
│   ├── envs.py          # vectorized text ALFWorld, task-success reward
│   ├── projection.py    # response -> admissible action
│   ├── prompts.py       # prompt templates
│   └── env_manager.py   # verl-agent environment manager + make_envs(config)
├── tau2bench/
│   ├── envs.py          # vectorized τ²-Bench on tau2-bench's Gymnasium interface, task-success reward
│   ├── projection.py    # response -> tool call or message to the user
│   └── env_manager.py   # verl-agent environment manager + make_envs(config)
├── prepare_data.py      # verl-agent dataset files with the task seed of each environment
└── utils.py
```

Compared to verl-agent's own ALFWorld environment, ours uses a different prompt template, only accepts admissible actions as valid, and gives -0.1 for every step that does not solve the task.
verl-agent does not include τ²-Bench.

## Setup

1. Install [verl-agent](https://github.com/langfengQ/verl-agent) and this repository (`pip install -e .`). The environments import verl-agent's `agent_system` package.
2. For τ²-Bench, install [tau2-bench](https://github.com/sierra-research/tau2-bench) at the commit we used:
   ```bash
   git clone https://github.com/sierra-research/tau2-bench.git
   cd tau2-bench && git checkout a969a0c && pip install -e .
   ```
3. Use our environments in verl-agent's `make_envs` (`agent_system/environments/env_manager.py`), before its own ALFWorld branch:
   ```python
   def make_envs(config):
       if "tau2bench" in config.env.env_name.lower():
           from rwml.envs.tau2bench import make_envs as make_tau2bench_envs
           return make_tau2bench_envs(config)
       if "alfworld" in config.env.env_name.lower():
           from rwml.envs.alfworld import make_envs as make_alfworld_envs
           return make_alfworld_envs(config)
       ...
   ```
4. For τ²-Bench, the environment renders the whole conversation (with tools) with the chat template itself and marks it with `[NO_APPLY_CHAT_TEMPLATE]`.
   In verl-agent's rollout loop (`preprocess_single_sample` in `agent_system/multi_turn_rollout/rollout_loop.py`), use such observations as the prompt as is:
   ```python
   from rwml.envs.utils import CHAT_TEMPLATE_APPLIED_PREFIX

   if obs_content.startswith(CHAT_TEMPLATE_APPLIED_PREFIX):
       prompt_with_chat_template = obs_content[len(CHAT_TEMPLATE_APPLIED_PREFIX):]
   else:
       prompt_with_chat_template = self.tokenizer.apply_chat_template(...)  # unchanged
   ```
5. Write the dataset files, which set the number of environments and their task seeds:
   ```bash
   python -m rwml.envs.prepare_data --env alfworld --train_size 8 --val_size 64
   python -m rwml.envs.prepare_data --env tau2bench --train_size 8 --val_size 64
   ```

`scripts/alfworld/run_policy_rl.sh` and `scripts/tau2bench/run_policy_rl.sh` then run policy RL from verl-agent.

## Configuration

Besides verl-agent's `env` options (`env_name`, `seed`, `max_steps`, `history_length`, `rollout.n`), the environments read:

| Option | Description |
|---|---|
| `env.randomize_reset_seed` | Sample new training tasks at every step instead of reusing the dataset seeds. We used `True`. |
| `env.alfworld.eval_dataset` | ALFWorld validation split: `eval_in_distribution` or `eval_out_of_distribution`. |
| `env.tau2bench.user_llm`, `env.tau2bench.user_llm_args` | User simulator, called through LiteLLM (model name, `api_base`, `api_key`, `temperature`, `max_completion_tokens`). |
| `env.tau2bench.task_set_name`, `env.tau2bench.task_split_name` | Training tasks, e.g. `all` (telecom, airline, retail) and `train`. |
| `env.tau2bench.test_task_set_name`, `env.tau2bench.test_task_split_name` | Validation tasks, e.g. `all` and `test`. |
| `env.tau2bench.max_concurrency` | Number of τ²-Bench episodes stepped in parallel. |

Options that verl-agent's config does not define are passed with a `+` (e.g. `+env.randomize_reset_seed=True`), as in the scripts.
