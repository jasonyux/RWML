"""ALFWorld environment manager for verl-agent."""

from typing import List

from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory

from rwml.envs.alfworld.envs import build_alfworld_envs, default_alfworld_config_path
from rwml.envs.alfworld.projection import alfworld_projection
from rwml.envs.alfworld.prompts import ALFWORLD_TEMPLATE_NO_HIS, ALFWORLD_TEMPLATE_W_REFLECTION
from rwml.envs.utils import get_reset_seeds

TASK_TYPES = [
    "pick_and_place",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]


class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config, is_train):
        # EnvironmentManagerBase's constructor differs across verl-agent versions, so set attributes directly
        self.envs = envs
        self.projection_f = projection_f
        self.config = config
        self.is_train = is_train
        self.memory = SimpleMemory()
        self.max_history_length = config.env.history_length
        self._num_resets = 0

    def reset(self, kwargs=None, seeds=None):
        seeds = get_reset_seeds(self.config, self.is_train, self._num_resets, kwargs, seeds)
        self._num_resets += 1
        text_obs, image_obs, infos = self.envs.reset(seeds)
        self.gamefiles = [info.get("extra.gamefile") for info in infos]
        self.memory.reset(batch_size=len(text_obs))
        self.pre_text_obs = text_obs
        self.tasks = [self.extract_task(obs) for obs in text_obs]

        prompts, build_metadatas = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        for info, task, build_metadata in zip(infos, self.tasks, build_metadatas):
            info["task_description"] = task
            info["admissible_actions"] = build_metadata["reformatted_admissible_actions"]
            info["build_metadata"] = build_metadata

        observations = {"text": prompts, "image": image_obs, "anchor": text_obs}
        self.envs.record_obs(observations, infos)
        return observations, infos

    def step(self, text_actions: List[str]):
        raw_actions = text_actions.copy()
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        self.memory.store({"text_obs": self.pre_text_obs.copy(), "action": actions.copy()})

        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions, raw_actions)
        self.pre_text_obs = text_obs

        prompts, build_metadatas = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        for info, gamefile, valid, task, build_metadata in zip(infos, self.gamefiles, valids, self.tasks, build_metadatas):
            if info.get("extra.gamefile") is None:
                info["extra.gamefile"] = gamefile
            info["is_action_valid"] = to_numpy(valid)
            info["action_is_effective"] = valid == 1
            info["task_description"] = task
            info["admissible_actions"] = build_metadata["reformatted_admissible_actions"]
            info["build_metadata"] = build_metadata

        next_observations = {"text": prompts, "image": image_obs, "anchor": text_obs}
        self.envs.record_obs(next_observations, infos, rewards=rewards, dones=dones)
        return next_observations, to_numpy(rewards), to_numpy(dones), infos

    @staticmethod
    def extract_task(text_obs: str) -> str:
        start = text_obs.find("Your task is to: ")
        if start == -1:
            raise ValueError("Task description not found in text observation.")
        return text_obs[start + len("Your task is to: ") :].strip()

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False):
        """Builds the prompt with the task, the last `max_history_length` steps, and the admissible actions."""
        use_history = not init and self.max_history_length > 0
        if use_history:
            memory_contexts, valid_lens = self.memory.fetch(self.max_history_length, obs_key="text_obs", action_key="action")

        prompts, build_metadatas = [], []
        for i, obs in enumerate(text_obs):
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != "help")
            if use_history:
                build_metadata = {
                    "step_count": len(self.memory[i]),
                    "history_length": valid_lens[i],
                    "action_history": memory_contexts[i],
                    "current_step": len(self.memory[i]) + 1,
                }
                prompt = ALFWORLD_TEMPLATE_W_REFLECTION.format(
                    task_description=self.tasks[i],
                    current_observation=obs,
                    admissible_actions=reformatted_admissible_actions,
                    **build_metadata,
                )
            else:
                build_metadata = {"step_count": 0, "history_length": 0, "action_history": [], "current_step": 1}
                prompt = ALFWORLD_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=obs,
                    admissible_actions=reformatted_admissible_actions,
                )
            build_metadata["reformatted_admissible_actions"] = reformatted_admissible_actions
            prompts.append(prompt)
            build_metadatas.append(build_metadata)
        return prompts, build_metadatas

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # success of the last active step, overall and per task type
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            if total_batch_list[batch_idx][i]["active_masks"]:
                info = total_infos[batch_idx][i]
                won = float(info["won"])
                success["success_rate"].append(won)
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    task_type = next((t for t in TASK_TYPES if t in gamefile), None)
                    if task_type:
                        success[f"{task_type}_success_rate"].append(won)
                return


def make_envs(config):
    """Builds the training and validation environment managers from a verl-agent config."""
    env_kwargs = {
        "max_steps": config.env.max_steps,
        "train_split": "train",
        "test_split": config.env.alfworld.get("eval_dataset", "eval_in_distribution"),
    }
    config_path = default_alfworld_config_path()
    group_n = max(config.env.rollout.n, 1)
    envs = build_alfworld_envs(config_path, config.env.seed, config.data.train_batch_size, group_n, True, env_kwargs)
    val_envs = build_alfworld_envs(config_path, config.env.seed + 1000, config.data.val_batch_size, 1, False, env_kwargs)
    return (
        AlfWorldEnvironmentManager(envs, alfworld_projection, config, is_train=True),
        AlfWorldEnvironmentManager(val_envs, alfworld_projection, config, is_train=False),
    )
