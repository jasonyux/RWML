"""τ²-Bench environment manager for verl-agent."""

from typing import List

from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from rwml.envs.tau2bench.envs import build_tau2bench_envs
from rwml.envs.tau2bench.projection import tau2bench_projection
from rwml.envs.utils import CHAT_TEMPLATE_APPLIED_PREFIX, get_reset_seeds

CHAT_TEMPLATE_TOKENIZER = "Qwen/Qwen2.5-7B-Instruct"


class Tau2BenchEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config, is_train):
        # EnvironmentManagerBase's constructor differs across verl-agent versions, so set attributes directly
        self.envs = envs
        self.projection_f = projection_f
        self.config = config
        self.is_train = is_train
        self.max_history_length = config.env.history_length
        self.tokenizer = AutoTokenizer.from_pretrained(CHAT_TEMPLATE_TOKENIZER)
        self._num_resets = 0

    def reset(self, kwargs=None, seeds=None):
        seeds = get_reset_seeds(self.config, self.is_train, self._num_resets, kwargs, seeds)
        self._num_resets += 1
        text_obs, image_obs, infos = self.envs.reset(seeds)

        states = self.envs.get_states()
        self.tasks = [state["task_purpose"] for state in states]
        prompts, build_metadatas = self.build_text_obs(states)
        for info, task, build_metadata in zip(infos, self.tasks, build_metadatas):
            info["task_description"] = task
            info["build_metadata"] = build_metadata

        observations = {"text": prompts, "image": image_obs, "anchor": text_obs}
        self.envs.record_obs(observations, infos)
        return observations, infos

    def step(self, text_actions: List[str]):
        raw_actions = text_actions.copy()
        actions, valids = self.projection_f(text_actions)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions, raw_actions)

        prompts, build_metadatas = self.build_text_obs(self.envs.get_states())
        for info, valid, task, build_metadata in zip(infos, valids, self.tasks, build_metadatas):
            info["is_action_valid"] = to_numpy(valid)
            info["action_is_effective"] = valid == 1
            info["task_description"] = task
            info["build_metadata"] = build_metadata

        next_observations = {"text": prompts, "image": image_obs, "anchor": text_obs}
        self.envs.record_obs(next_observations, infos, rewards=rewards, dones=dones)
        return next_observations, to_numpy(rewards), to_numpy(dones), infos

    @staticmethod
    def truncate_messages(messages: list[dict], n: int) -> list[dict]:
        """Keeps the last `n` assistant turns and what follows them."""
        assistant_turns = [i for i, message in enumerate(messages) if message["role"] == "assistant"]
        if len(assistant_turns) <= n:
            return messages
        return messages[assistant_turns[-n] :]

    def build_text_obs(self, states: List[dict]):
        """Renders each conversation (system prompt + last turns, with tools) into a prompt with the chat template."""
        prompts, build_metadatas = [], []
        for state in states:
            messages = state["messages"]
            if self.max_history_length > 0:
                messages = messages[:1] + self.truncate_messages(messages[1:], self.max_history_length)
            prompt = self.tokenizer.apply_chat_template(
                messages, tools=state["tools"], tokenize=False, add_generation_prompt=True
            )
            prompts.append(CHAT_TEMPLATE_APPLIED_PREFIX + prompt)
            build_metadatas.append(
                {
                    "full_messages": state["messages"],
                    "messages": messages,
                    "tools": state["tools"],
                    "tool_choice": state["tool_choice"],
                }
            )
        return prompts, build_metadatas

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # success of the last active step, overall and per domain
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            if total_batch_list[batch_idx][i]["active_masks"]:
                info = total_infos[batch_idx][i]
                won = float(info["won"])
                success["success_rate"].append(won)
                domain = info.get("task", {}).get("user_scenario", {}).get("instructions", {}).get("domain")
                if domain:
                    success[f"{domain}_success_rate"].append(won)
                return


def make_envs(config):
    """Builds the training and validation environment managers from a verl-agent config."""
    tau2_config = config.env.tau2bench
    env_kwargs = {
        "max_steps": config.env.max_steps * 2,  # tau2-bench counts both user and agent turns
        "user_llm": tau2_config.user_llm,
        "user_llm_args": OmegaConf.to_container(tau2_config.user_llm_args, resolve=True),
    }
    envs = build_tau2bench_envs(
        tau2_config.task_set_name,
        tau2_config.task_split_name,
        tau2_config.max_concurrency,
        config.env.seed,
        config.data.train_batch_size,
        max(config.env.rollout.n, 1),
        env_kwargs,
    )
    val_envs = build_tau2bench_envs(
        tau2_config.test_task_set_name,
        tau2_config.test_task_split_name,
        tau2_config.max_concurrency,
        config.env.seed + 1000,
        config.data.val_batch_size,
        1,
        env_kwargs,
    )
    return (
        Tau2BenchEnvironmentManager(envs, tau2bench_projection, config, is_train=True),
        Tau2BenchEnvironmentManager(val_envs, tau2bench_projection, config, is_train=False),
    )
