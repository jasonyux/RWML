"""RWML reward for ALFWorld

Use with verl's batch reward manager:
    reward_model.reward_manager=batch
    custom_reward_function.path=rwml/rewards/alfworld.py
    custom_reward_function.name=compute_score_batched
"""

from rwml.rewards.common import embedding_similarity, extract_next_state, run_batched


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    *,
    embed_api_base,
    embed_api_key="EMPTY",
    embed_model="qwen-embedding-8b",
    max_prediction_tokens=100,
    sim_threshold=0.8,
):
    """1.0 if the predicted next state is close to the real one (`ground_truth`) in embedding space, else 0.0."""
    prediction = extract_next_state(solution_str, max_tokens=max_prediction_tokens)
    similarity = embedding_similarity(ground_truth, prediction, embed_api_base, embed_api_key, embed_model)
    return 1.0 if similarity >= sim_threshold else 0.0


def compute_score_batched(data_sources, solution_strs, ground_truths, extra_infos, num_workers=8, **kwargs):
    return run_batched(compute_score, data_sources, solution_strs, ground_truths, extra_infos, num_workers, **kwargs)
