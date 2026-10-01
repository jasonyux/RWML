#!/usr/bin/env bash
# Serves Qwen3-Embedding-8B with an OpenAI-compatible /v1/embeddings endpoint for the RWML rewards.
# Run it on GPUs not used for training, then set EMBED_API_BASE=http://<host>:<port>/v1 for the training scripts.
# Newer vLLM versions replace `--task embed` with `--runner pooling`.
set -xeuo pipefail

PORT=${PORT:-8000}

vllm serve Qwen/Qwen3-Embedding-8B \
    --task embed \
    --served-model-name qwen-embedding-8b \
    --port "$PORT" \
    "$@"
