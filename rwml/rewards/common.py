"""Shared helpers for the RWML sim-to-real gap rewards.

The model reasons inside <think>...</think> and then predicts the next state inside
<next_state>...</next_state>. The prediction is compared against the real next state
observed from the environment (the `ground_truth` of each training sample).
"""

import concurrent.futures
import functools
import threading
import time

import cachetools
import numpy as np
import openai
import tiktoken

NEXT_STATE_TAGS = ("<next_state>", "</next_state>")

# Instruction for Qwen3-Embedding. The real next state is embedded as the query (with this
# instruction) and the predicted next state as the document.
EMBED_QUERY_TEMPLATE = """
Instruct: Given a reference description, retrieve similar descriptions that mentioned all important information in the reference AND correctly described the task completion status specified in the reference.
Reference: {reference}
""".strip()

# Task completion statuses that end every next state in the RWML datasets, e.g.
# "... Task is not yet completed."
TASK_STATUSES = ("successfully completed", "not yet completed", "failed")

# Only used to count and truncate tokens, independent of the policy's tokenizer.
_TOKENIZER = tiktoken.encoding_for_model("gpt-4o")

# All rollouts in a GRPO group share the same reference, so cache its embedding.
_query_embedding_cache = cachetools.LRUCache(maxsize=1000)
_query_embedding_cache_lock = threading.Lock()


def num_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text, disallowed_special=()))


def extract_next_state(response: str, max_tokens: int | None = None) -> str:
    """Returns the content of the last <next_state>...</next_state> block, or "none" if there is none."""
    start_tag, end_tag = NEXT_STATE_TAGS
    start = response.rfind(start_tag)
    end = response.rfind(end_tag)
    if start == -1 or end == -1 or start >= end:
        return "none"
    next_state = response[start + len(start_tag) : end]
    if max_tokens is not None:
        next_state = _TOKENIZER.decode(_TOKENIZER.encode(next_state, disallowed_special=())[:max_tokens])
    return next_state.strip()


def has_thinking(response: str, min_tokens: int = 16) -> bool:
    """Whether the response contains a <think>...</think> block with more than `min_tokens` tokens."""
    start = response.find("<think>")
    end = response.find("</think>")
    if start == -1 or end == -1 or start >= end:
        return False
    return num_tokens(response[start + len("<think>") : end]) > min_tokens


def task_status_matches(prediction: str, reference_status: str) -> bool:
    """Whether the last line of `prediction` states the task status in `reference_status`, and no other."""
    expected = next((status for status in TASK_STATUSES if status in reference_status), None)
    if expected is None:
        raise ValueError(f"Unknown task status: {reference_status!r}")
    lines = prediction.strip().splitlines()
    if not lines:
        return False
    last_line = lines[-1].strip()
    return all((status in last_line) == (status == expected) for status in TASK_STATUSES)


@functools.lru_cache(maxsize=None)
def get_client(api_base: str, api_key: str) -> openai.OpenAI:
    return openai.OpenAI(base_url=api_base, api_key=api_key)


def _embed(texts: list[str], api_base: str, api_key: str, model: str) -> list[np.ndarray]:
    response = get_client(api_base, api_key).embeddings.create(model=model, input=texts, encoding_format="float")
    embeddings = [np.asarray(d.embedding, dtype=np.float64) for d in sorted(response.data, key=lambda d: d.index)]
    return [e / np.linalg.norm(e) for e in embeddings]


def embedding_similarity(reference: str, prediction: str, api_base: str, api_key: str, model: str) -> float:
    """Cosine similarity between E(instruction + reference) and E(prediction), i.e. 1 - d(ŝ, s) in the paper.

    Returns 0.0 if the embedding request fails, so a flaky server does not crash training.
    """
    query = EMBED_QUERY_TEMPLATE.format(reference=reference)
    with _query_embedding_cache_lock:
        query_embedding = _query_embedding_cache.get(query)
    try:
        if query_embedding is None:
            query_embedding, prediction_embedding = _embed([query, prediction], api_base, api_key, model)
            with _query_embedding_cache_lock:
                _query_embedding_cache[query] = query_embedding
        else:
            (prediction_embedding,) = _embed([prediction], api_base, api_key, model)
    except Exception as e:
        print(f"[rwml.rewards] embedding request failed, using similarity 0.0: {e!r}")
        return 0.0
    return float(query_embedding @ prediction_embedding)


def run_batched(score_fn, data_sources, solution_strs, ground_truths, extra_infos, num_workers=8, **kwargs):
    """Scores a batch with a thread pool (the embedding calls are I/O bound). Returns scores in input order."""
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(num_workers)) as executor:
        scores = list(
            executor.map(
                lambda sample: score_fn(*sample, **kwargs),
                zip(data_sources, solution_strs, ground_truths, extra_infos),
            )
        )
    print(f"[rwml.rewards] scored {len(scores)} responses in {(time.time() - start_time) / 60:.2f}m")
    return scores
