"""Vectorized text ALFWorld environments, using the ALFWorld package bundled with verl-agent."""

import copy
import os

import gymnasium as gym
import yaml
from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment


def compute_reward(info):
    """Task-success reward per step: 10 on success, -0.1 otherwise."""
    return 10.0 if info["won"] else -0.1


class AlfworldWorker:
    """One ALFWorld game at a time."""

    def __init__(self, base_env):
        self.base_env = base_env
        self.env = None
        self.trajectory = [{}]
        self._step_idx = 0
        self.is_done = False
        self.is_won = False

    def reset(self, seed):
        self.trajectory = self.trajectory[:1]
        self._step_idx = 0
        self.is_done = False
        self.is_won = False

        if self.env is not None:
            self.env.close()
        self.env = self.base_env.init_env(batch_size=1)
        self.env.seed(seed)  # the seed picks the game
        obs, infos = self.env.reset()
        infos["observation_text"] = obs
        infos["_text_obs"] = [obs[0]]
        return obs, infos

    def step(self, action, raw_action: str):
        obs, scores, dones, infos = self.env.step([action])
        infos["observation_text"] = obs
        infos["_text_obs"] = [obs[0]]
        done = dones[0] or self.is_done
        won = infos["won"][0] or self.is_won
        infos["won"] = [won]

        self.trajectory.append({"action": action, "raw_action": raw_action, "step_idx": self._step_idx})
        self._step_idx += 1
        self.is_done = done
        self.is_won = won
        return obs, scores, [done], infos

    def record_obs(self, obs, info, reward=0.0, done=False):
        self.trajectory.append({"obs": obs, "info": info, "done": done, "reward": reward})

    def render_trajectory(self):
        return copy.deepcopy(self.trajectory)

    def close(self):
        if self.env is not None:
            self.env.close()


class AlfworldEnvs(gym.Env):
    """`env_num * group_n` ALFWorld (AlfredTWEnv) workers."""

    def __init__(self, alf_config_path, seed=0, env_num=1, group_n=1, is_train=True, **env_kwargs):
        super().__init__()
        with open(alf_config_path) as f:
            config = yaml.safe_load(f)
        config["env"]["expert_type"] = "planner"
        max_steps = env_kwargs.get("max_steps", 50)
        config["rl"]["training"]["max_nb_steps_per_episode"] = max_steps
        config["dagger"]["training"]["max_nb_steps_per_episode"] = max_steps
        split = env_kwargs.get("train_split", "train") if is_train else env_kwargs.get("test_split", "eval_in_distribution")

        self.base_env = get_environment(config["env"]["type"])(config, train_eval=split)
        self.seed = seed
        self.num_processes = env_num * group_n
        self.group_n = group_n
        self.workers = []
        self.prev_admissible_commands = [None] * self.num_processes

    def reset(self, seeds=None):
        if seeds is None:
            seeds = [self.seed + i // self.group_n for i in range(self.num_processes)]
        if not self.workers:
            self.workers = [AlfworldWorker(self.base_env) for _ in range(len(seeds))]

        text_obs, infos = [], []
        for i, (worker, seed) in enumerate(zip(self.workers, seeds)):
            obs, info = worker.reset(seed)
            info = {k: v[0] for k, v in info.items()}
            text_obs.append(obs[0])
            infos.append(info)
            self.prev_admissible_commands[i] = info["admissible_commands"]
        return text_obs, None, infos

    def step(self, actions, raw_actions):
        assert len(actions) == len(self.workers), f"Got {len(actions)} actions for {len(self.workers)} workers"
        text_obs, rewards, dones, infos = [], [], [], []
        for i, (worker, action, raw_action) in enumerate(zip(self.workers, actions, raw_actions)):
            obs, _, done, info = worker.step(action, raw_action)
            info = {k: v[0] for k, v in info.items()}
            text_obs.append(obs[0])
            dones.append(done[0])
            infos.append(info)
            rewards.append(compute_reward(info))
            self.prev_admissible_commands[i] = info["admissible_commands"]
        return text_obs, None, rewards, dones, infos

    @property
    def get_admissible_commands(self):
        return self.prev_admissible_commands

    def record_obs(self, obs, infos, rewards=None, dones=None):
        for i, worker in enumerate(self.workers):
            image = obs["image"][i] if obs["image"] is not None else None
            reward = rewards[i] if rewards is not None else 0.0
            done = dones[i] if dones is not None else False
            worker.record_obs({"text": obs["text"][i], "image": image}, infos[i], reward=reward, done=done)

    def render_trajectories(self):
        return [worker.render_trajectory() for worker in self.workers]

    def close(self):
        for worker in self.workers:
            worker.close()
        self.workers = []


def build_alfworld_envs(alf_config_path, seed, env_num, group_n, is_train=True, env_kwargs=None):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, is_train, **(env_kwargs or {}))


def default_alfworld_config_path():
    """The text-world config bundled with verl-agent."""
    import agent_system.environments

    return os.path.join(os.path.dirname(agent_system.environments.__file__), "env_package/alfworld/configs/config_tw.yaml")
