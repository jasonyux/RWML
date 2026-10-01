"""Vectorized τ²-Bench environments, built on tau2-bench's Gymnasium interface."""

import concurrent.futures
import copy

import gymnasium as gym
from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
from tau2.data_model.message import AssistantMessage, Message, SystemMessage, ToolMessage, UserMessage
from tau2.gym import TAU_BENCH_ENV_ID, register_gym_agent
from tau2.run import get_tasks

DOMAINS = ["telecom", "airline", "retail"]


def compute_reward(info, done):
    """Task-success reward: 10 on success, -1 on failure (given at the end of the episode)."""
    if not done:
        return 0.0
    return 10.0 if info["won"] else -1.0


def get_all_tasks(task_set_name, task_split_name):
    """Tasks of one domain, or of all domains with task_set_name="all"."""
    domains = DOMAINS if task_set_name == "all" else [task_set_name]
    return [task for domain in domains for task in get_tasks(task_set_name=domain, task_split_name=task_split_name)]


def to_chat_messages(messages: list[Message]) -> list[dict]:
    """Converts tau2 messages to chat messages.

    Unlike tau2's `to_litellm_messages`, tool-call arguments stay dicts so that chat templates
    render them as JSON objects.
    """
    chat_messages = []
    for message in messages:
        if isinstance(message, UserMessage):
            chat_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            tool_calls = None
            if message.is_tool_call():
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "function": {"name": tc.name, "arguments": tc.arguments},
                        "type": "function",
                    }
                    for tc in message.tool_calls
                ]
            chat_messages.append({"role": "assistant", "content": message.content, "tool_calls": tool_calls})
        elif isinstance(message, ToolMessage):
            chat_messages.append({"role": "tool", "content": message.content, "tool_call_id": message.id})
        elif isinstance(message, SystemMessage):
            chat_messages.append({"role": "system", "content": message.content})
    return chat_messages


class Tau2BenchWorker:
    """One τ²-Bench episode at a time: the policy is the agent, tau2-bench simulates the user."""

    def __init__(self, **env_kwargs):
        self.env_kwargs = env_kwargs
        self.env = None
        self.task = None
        # tau2-bench counts both user and agent turns, we only count agent turns
        self._max_steps = env_kwargs["max_steps"] // 2
        self.trajectory: list[dict] = [{"env_kwargs": env_kwargs}]
        self._step_idx = 0
        self._last_step_output = None
        self.is_done = False
        self.is_won = False

    def get_state(self):
        """The agent's conversation so far (with the system prompt) and the available tools."""
        env = self.env.unwrapped
        system_prompt = SYSTEM_PROMPT.format(domain_policy=env._get_policy(), agent_instruction=AGENT_INSTRUCTION)
        messages = [SystemMessage(role="system", content=system_prompt)] + env._agent.observation
        tools = env._get_tools()
        tools = [tool.openai_schema for tool in tools] if tools else None
        return {
            "messages": to_chat_messages(messages),
            "tools": tools,
            "tool_choice": "auto" if tools else None,
            "task_purpose": self.task.description.purpose,
        }

    def reset(self, task):
        self.trajectory = self.trajectory[:1]
        self._step_idx = 0
        self._last_step_output = None
        self.is_done = False
        self.is_won = False

        if self.env is not None:
            self.env.close()
        self.env = gym.make(
            TAU_BENCH_ENV_ID,
            domain=task.user_scenario.instructions.domain,
            task_id=task.id,
            **self.env_kwargs,
        )
        self.task = task
        obs, info = self.env.reset()
        info["_text_obs"] = obs
        info["_agent_step_idx"] = 0
        info["won"] = False
        info["task"] = info["task"].model_dump(mode="json")
        info["tools"] = [tool.model_dump(mode="json") for tool in info["tools"]]
        self.trajectory[0]["eval_criteria"] = info["task"]["evaluation_criteria"]
        return obs, info

    def step(self, action, raw_action: str):
        if self.is_done:
            # finished episodes keep returning their last step
            obs, reward, done, info = self._last_step_output
            won = self.is_won
        else:
            obs, reward, done, _, info = self.env.step(action)
            if self._step_idx + 1 >= self._max_steps:
                done = True
            info["_text_obs"] = obs
            info["_agent_step_idx"] = self._step_idx
            won = reward == 1.0
            info["won"] = won
            info["task"] = info["task"].model_dump(mode="json")
            info["tools"] = [tool.model_dump(mode="json") for tool in info["tools"]]

        self.trajectory.append({"action": action, "raw_action": raw_action, "step_idx": self._step_idx})
        self._step_idx += 1
        self._last_step_output = (obs, reward, done, info)
        self.is_done = done
        self.is_won = won
        return obs, reward, done, info

    def record_obs(self, obs, info, reward=0.0, done=False):
        self.trajectory.append({"obs": obs, "info": info, "done": done, "reward": reward})

    def render_trajectory(self):
        return copy.deepcopy(self.trajectory)

    def close(self):
        if self.env is not None:
            self.env.close()


class Tau2BenchEnvs(gym.Env):
    """`env_num * group_n` τ²-Bench workers, stepped concurrently with threads (the user simulator is an API call)."""

    def __init__(self, task_set_name, task_split_name, concurrency, seed=0, env_num=1, group_n=1, **env_kwargs):
        super().__init__()
        self.all_tasks = get_all_tasks(task_set_name, task_split_name)
        self.env_kwargs = copy.deepcopy(env_kwargs)
        self.concurrency = concurrency
        self.seed = seed
        self.num_processes = env_num * group_n
        self.group_n = group_n
        self.workers = []

    def _map(self, fn, *args):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            return list(executor.map(fn, self.workers, *args))

    def reset(self, seeds=None):
        if seeds is None:
            seeds = [self.seed + i // self.group_n for i in range(self.num_processes)]
        if not self.workers:
            self.workers = [Tau2BenchWorker(**copy.deepcopy(self.env_kwargs)) for _ in range(len(seeds))]
        tasks = [self.all_tasks[seed % len(self.all_tasks)] for seed in seeds]
        results = self._map(lambda worker, task: worker.reset(task), tasks)
        text_obs = [obs for obs, _ in results]
        infos = [info for _, info in results]
        return text_obs, None, infos

    def step(self, actions, raw_actions):
        assert len(actions) == len(self.workers), f"Got {len(actions)} actions for {len(self.workers)} workers"
        results = self._map(lambda worker, action, raw_action: worker.step(action, raw_action), actions, raw_actions)
        text_obs, rewards, dones, infos = [], [], [], []
        for obs, _, done, info in results:
            text_obs.append(obs)
            dones.append(done)
            infos.append(info)
            rewards.append(compute_reward(info, done))
        return text_obs, None, rewards, dones, infos

    def get_states(self):
        return [worker.get_state() for worker in self.workers]

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


def build_tau2bench_envs(task_set_name, task_split_name, concurrency, seed, env_num, group_n, env_kwargs):
    register_gym_agent()
    return Tau2BenchEnvs(task_set_name, task_split_name, concurrency, seed, env_num, group_n, **env_kwargs)
