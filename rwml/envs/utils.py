import numpy as np

# Marks observations that are already full prompts (chat template applied), see rwml/envs/README.md.
CHAT_TEMPLATE_APPLIED_PREFIX = "[NO_APPLY_CHAT_TEMPLATE]"


def get_reset_seeds(config, is_train, num_resets, kwargs=None, seeds=None):
    """Task seeds for the next episodes, one per environment (shared within a GRPO group).

    With `env.randomize_reset_seed`, training samples new tasks at every step instead of reusing
    the seeds of the dataset rows.
    """
    if seeds is None and kwargs is not None:
        # verl-agent passes the `env_kwargs` of each (already repeated) dataset row
        seeds = [env_kwargs["seed"] for env_kwargs in kwargs]
    if is_train and config.env.get("randomize_reset_seed", False):
        rng = np.random.default_rng(num_resets)
        seeds = np.repeat(rng.integers(2_000, 200_000, size=config.data.train_batch_size), config.env.rollout.n)
    return seeds
