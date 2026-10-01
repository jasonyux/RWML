#!/usr/bin/env bash
# Converts every verl FSDP actor checkpoint in a run directory to a Hugging Face model:
#   <ckpt_dir>/global_step_<N>/actor  ->  <ckpt_dir>/global_step_<N>/actor_hf
set -euo pipefail

CKPT_DIR=${1:?usage: $0 <verl checkpoint dir>}

for actor_dir in "$CKPT_DIR"/global_step_*/actor; do
    [ -d "$actor_dir" ] || continue
    target_dir="$(dirname "$actor_dir")/actor_hf"
    if [ -d "$target_dir" ]; then
        echo "Skipping $actor_dir: $target_dir already exists"
        continue
    fi
    python3 -m verl.model_merger merge --backend fsdp --local_dir "$actor_dir" --target_dir "$target_dir"
done
