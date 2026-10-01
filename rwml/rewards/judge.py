"""LLM-as-a-judge reward, an ablation of the RWML reward (Table 4 of the paper)

Use with verl's batch reward manager:
    reward_model.reward_manager=batch
    custom_reward_function.path=rwml/rewards/judge.py
    custom_reward_function.name=compute_score_batched
"""

import json

import numpy as np

from rwml.rewards.common import extract_next_state, get_client, has_thinking, num_tokens, run_batched

# "w/ LLM-as-a-judge": real-valued score in [0, 1]
JUDGE_PROMPT = """
{obs_text}

# Action proposed by an AI agent
{action_text}

# Predicted Next observation
After executing the above action, another AI agent predicted the next observation as follows:
{next_obs_desc}

# Actual Next observation
The actual next observation from the environment is as follows:
{actual_next_obs_text}

# Evaluate the predicted next observation
Now, your task is to evaluate how well the predicted next observation matches the actual next observation.
Specifically, you need to judge whether the prediction demonstrates a *genuine* understanding of the environment dynamics relevant to the task, using the actual next observation as reference.

Award points for:
- Specific and accurate environment description in the prediction that matches the actual next observation.
- Correct prediction of task completion status.
Penalize:
- Incorrect environment descriptions in the prediction that differ from the actual next observation, especially those that cold impact (future) task completion. 
- Missing or wrong task completion status in the prediction compared to the actual next observation.
STRONGLY penalize:
- The prediction misses key information from the actual next observation that is important for solving the task.
- Vague or generic descriptions that is consistent with the actual next observation but does not really show genuine understanding of the environment dynamics.

# Your output format
Your task is to output a JSON object in the following format:
<json>
{{
    "positive aspects": "enumerating good aspects of the agent's prediction according to the guidelines above, using the actual next observation as reference.",  # no more than 200 words
    "negative aspects": "enumerating bad aspects of the agent's prediction according to the guidelines above, using the actual next observation as reference.",  # no more than 200 words
    "overall analysis": "overall analysis weighing both aspects, and whether you think the agent has a genuine understanding of the environment dynamics.",  # no more than 50 words
    "score": 0.0-1.0 # overall score summarizing your judgement. higher the better.
}}
</json>
Directly output the JSON object. DO NOT generate anything else.
""".strip()

# "w/ bin(LLM-as-a-judge)": binary score
BINARY_JUDGE_PROMPT = """
{obs_text}

# Action proposed by an AI agent
{action_text}

# Predicted Next observation
After executing the above action, another AI agent predicted the next observation as follows:
{next_obs_desc}

# Actual Next observation
The actual next observation from the environment is as follows:
{actual_next_obs_text}

# Evaluate the predicted next observation
Now, your task is to evaluate how well the predicted next observation matches the actual next observation.
Specifically, you need to judge whether the prediction demonstrates a *genuine* understanding of the environment dynamics relevant to the task, using the actual next observation as reference.
- If all important task-related information in the actual next observation is PRESENT in the predicted next observation AND the task completion status matches the reference in the actual next observation, assign a score of 1.0.
- Otherwise, assign a score of 0.0.

# Your output format
Your task is to output a JSON object in the following format:
<json>
{{
    "analysis": "which important task related content is present/missing in the predicted next observation, and whether the task completion status is correctly predicted.",  # no more than 200 words
    "score": 0.0 or 1.0
}}
</json>
Directly output the JSON object. DO NOT generate anything else.
""".strip()


def has_language_mixing(text: str, threshold: float = 0.1) -> bool:
    """Whether more than `threshold` of the tokens come from non-ASCII words."""
    total = num_tokens(text)
    if total == 0:
        return False
    return num_tokens(" ".join(word for word in text.split() if not word.isascii())) / total > threshold


def judge_score(prompt, api_base, api_key, model, temperature, top_p, max_completion_tokens) -> float:
    """The judge's score clipped to [0, 1], or 0.0 if the request or parsing fails."""
    try:
        response = get_client(api_base, api_key).chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=max_completion_tokens,
        )
        content = response.choices[0].message.content
        for tag in ("<json>", "</json>", "```json", "```"):
            content = content.replace(tag, "")
        return float(np.clip(float(json.loads(content)["score"]), 0.0, 1.0))
    except Exception as e:
        print(f"[rwml.rewards] judge failed, using score 0.0: {e!r}")
        return 0.0


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info,
    *,
    judge_api_base,
    judge_api_key="EMPTY",
    judge_model="Qwen3-235B-A22B-Instruct-2507",
    binary=False,
    max_prediction_tokens=128,
    temperature=0.7,
    top_p=0.95,
    max_completion_tokens=2048,
    language_mixing_penalty=0.2,
    no_thinking_penalty=0.1,
):
    """The judge's score for the predicted next state given the real one (`ground_truth`), minus
    `language_mixing_penalty` for mixed-language responses and `no_thinking_penalty` without a <think> block.
    """
    prediction = extract_next_state(solution_str, max_tokens=max_prediction_tokens)
    prompt = (BINARY_JUDGE_PROMPT if binary else JUDGE_PROMPT).format(
        obs_text=extra_info["obs_text"],
        action_text=extra_info["action_text"],
        next_obs_desc=prediction,
        actual_next_obs_text=ground_truth,
    )
    reward = judge_score(prompt, judge_api_base, judge_api_key, judge_model, temperature, top_p, max_completion_tokens)
    if has_language_mixing(solution_str):
        reward -= language_mixing_penalty
    if not has_thinking(solution_str):
        reward -= no_thinking_penalty
    return reward


def compute_score_batched(data_sources, solution_strs, ground_truths, extra_infos, num_workers=8, **kwargs):
    return run_batched(compute_score, data_sources, solution_strs, ground_truths, extra_infos, num_workers, **kwargs)
