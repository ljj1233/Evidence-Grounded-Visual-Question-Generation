#!/usr/bin/env python3
"""Launch the vLLM OpenAI-compatible server for the SFT baseline (base + LoRA).

Reads base model and image-resolution budget from the SFT config so the server
matches training. Run with the vLLM environment:

  .venv-vllm-cu128-v011/bin/python scripts/vvqg/serve_vllm_baseline.py

This is the optional vLLM serving path (kept for future GRPO rollouts). For
offline evaluation use the faster ``scripts/vvqg/generate.py`` (PIL image
processor + cross-image batching).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "new_vqg" / "sft.yaml"
DEFAULT_LORA = REPO_ROOT / "outputs" / "checkpoints" / "qwen3vl8b_lora_r16_vqav2_50k" / "checkpoint-final"


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the SFT baseline with vLLM.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--lora", type=Path, default=DEFAULT_LORA)
    parser.add_argument("--served-model-name", default="qwen3vl")
    parser.add_argument("--lora-name", default="sft_baseline")
    parser.add_argument("--max-lora-rank", type=int, default=16)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--async-scheduling", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--generation-config", default="vllm")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    base_model = args.base_model or os.path.expanduser(config["base_model"])
    image = config.get("image") or {}
    mm_processor_kwargs = {
        key: int(image[key]) for key in ("min_pixels", "max_pixels") if image.get(key) is not None
    }

    command = [
        "vllm", "serve", base_model,
        "--served-model-name", args.served_model_name,
        "--enable-lora",
        "--max-lora-rank", str(args.max_lora_rank),
        "--lora-modules", f"{args.lora_name}={args.lora}",
        "--dtype", "bfloat16",
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-num-batched-tokens", str(args.max_num_batched_tokens),
        "--max-num-seqs", str(args.max_num_seqs),
        "--generation-config", args.generation_config,
        "--limit-mm-per-prompt", '{"image": 1}',
        "--mm-processor-kwargs", json.dumps(mm_processor_kwargs),
        "--host", args.host,
        "--port", str(args.port),
    ]
    if args.async_scheduling:
        command.append("--async-scheduling")
    print("exec:", " ".join(command), flush=True)
    os.execvp("vllm", command)


if __name__ == "__main__":
    main()
