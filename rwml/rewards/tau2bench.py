"""RWML reward for τ²-Bench

Use with verl's batch reward manager:
    reward_model.reward_manager=batch
    custom_reward_function.path=rwml/rewards/tau2bench.py
    custom_reward_function.name=compute_score_batched
"""

from rouge_score import rouge_scorer

from rwml.rewards.common import (
    embedding_similarity,
    extract_next_state,
    has_thinking,
    run_batched,
    task_status_matches,
)


def rouge_f1(reference: str, prediction: str, metric: str = "rougeL") -> float:
    scorer = rouge_scorer.RougeScorer([metric], use_stemmer=True)
    return scorer.score(target=reference, prediction=prediction)[metric].fmeasure


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info,
    *,
    embed_api_base,
    embed_api_key="EMPTY",
    embed_model="qwen-embedding-8b",
    max_prediction_tokens=512,
    sim_threshold=0.6,
    rouge_metric="rougeL",
    rouge_bin=0.2,
    no_thinking_penalty=0.1,
    task_status_penalty=0.5,
    think_in_prompt=False,
):
    """Scores the predicted next state against the real one (`ground_truth`):

    - tool response: ROUGE-L F1, rounded to a multiple of `rouge_bin`
    - user response: 1.0 if close to the real response in embedding space, else 0.0

    minus `no_thinking_penalty` if there is no <think> block and `task_status_penalty` if the
    predicted task status is wrong.
    """
    if think_in_prompt:
        # Chat templates of e.g. Qwen3-*-Thinking-2507 already open <think> in the generation prompt.
        solution_str = "<think>" + solution_str
    prediction = extract_next_state(solution_str, max_tokens=max_prediction_tokens)

    if extra_info["metadata"]["response_type"] == "tool":
        # Tool responses are structured (e.g. JSON schemas), so compare them lexically.
        rouge = rouge_f1(ground_truth, prediction, rouge_metric)
        reward = round(rouge / rouge_bin) * rouge_bin
    else:
        # User responses are free-form text from the user simulator, so compare them semantically.
        similarity = embedding_similarity(ground_truth, prediction, embed_api_base, embed_api_key, embed_model)
        reward = 1.0 if similarity >= sim_threshold else 0.0

    if not has_thinking(solution_str):
        reward -= no_thinking_penalty
    # The last line of the real next state is its task status, e.g. "Task is not yet completed."
    if not task_status_matches(prediction, ground_truth.strip().splitlines()[-1]):
        reward -= task_status_penalty
    return reward


def compute_score_batched(data_sources, solution_strs, ground_truths, extra_infos, num_workers=4, **kwargs):
    return run_batched(compute_score, data_sources, solution_strs, ground_truths, extra_infos, num_workers, **kwargs)
