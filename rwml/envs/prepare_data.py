"""Writes the verl-agent dataset files for policy RL on ALFWorld or τ²-Bench.

As in verl-agent, the rows only set the number of environments; prompts come from the environments.
Each row also carries `env_kwargs.seed`, the task each environment is reset to:
  - training: seeds 0..N-1 (re-sampled at every step with `env.randomize_reset_seed=True`)
  - ALFWorld validation: seeds of unique tasks (consecutive seeds can map to the same task)
  - τ²-Bench validation: fixed random seeds (tasks are picked with `seed % num_tasks`)

Usage:
    python -m rwml.envs.prepare_data --env tau2bench --train_size 8 --val_size 64 --output_dir ~/data/rwml
"""

import argparse
import os

import datasets
import numpy as np

ALFWORLD_UNIQUE_SEEDS = {
    "id": [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41,
        42, 43, 44, 45, 46, 47, 49, 51, 53, 54, 56, 60, 61, 63, 64, 65, 68, 70, 71, 72,
        73, 74, 75, 76, 77, 78, 79, 81, 82, 83, 86, 87, 88, 90, 94, 97, 99, 100, 101, 106,
        110, 111, 115, 117, 119, 125, 129, 130, 135, 139, 141, 143, 145, 147, 151, 153, 155, 159, 160, 161,
        165, 174, 184, 186, 191, 193, 212, 213, 214, 215, 216, 221, 225, 230, 233, 239, 249, 250, 257, 262,
        267, 286, 294, 298, 301, 306, 308, 323, 330, 334, 338, 355, 363, 392, 395, 457, 463,
    ],
    "ood": [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 23, 24, 25, 26, 30, 32, 34, 35, 38, 40, 41, 42, 44, 48, 52, 53, 57, 59, 60,
        64, 66, 69, 72, 74, 75, 76, 77, 80, 83, 86, 87, 91, 98, 103, 107, 117, 118, 119, 127,
        130, 132, 137, 143, 147, 149, 156, 162, 166, 169, 173, 200, 205, 211, 234, 275, 482, 793,
    ],
}


def make_rows(split, seeds):
    return [
        {
            "data_source": "text",
            "prompt": [{"role": "user", "content": ""}],
            "ability": "agent",
            "extra_info": {"split": split, "index": idx},
            "env_kwargs": {"seed": int(seed)},
        }
        for idx, seed in enumerate(seeds)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, choices=["alfworld", "tau2bench"])
    parser.add_argument("--train_size", type=int, default=8)
    parser.add_argument("--val_size", type=int, default=64)
    parser.add_argument("--alfworld_val_split", default="id", choices=["id", "ood"])
    parser.add_argument("--output_dir", default="~/data/rwml")
    args = parser.parse_args()

    train_seeds = list(range(args.train_size))
    if args.env == "alfworld":
        val_seeds = ALFWORLD_UNIQUE_SEEDS[args.alfworld_val_split][: args.val_size]
        assert len(val_seeds) == args.val_size, f"At most {len(val_seeds)} unique validation tasks"
    else:
        np.random.seed(1000)
        val_seeds = np.random.randint(2**16, 2**32 - 1, size=args.val_size).tolist()

    output_dir = os.path.join(os.path.expanduser(args.output_dir), args.env)
    os.makedirs(output_dir, exist_ok=True)
    datasets.Dataset.from_list(make_rows("train", train_seeds)).to_parquet(os.path.join(output_dir, "train.parquet"))
    datasets.Dataset.from_list(make_rows("test", val_seeds)).to_parquet(os.path.join(output_dir, "test.parquet"))
    print(f"Wrote {len(train_seeds)} training and {len(val_seeds)} validation rows to {output_dir}")


if __name__ == "__main__":
    main()
