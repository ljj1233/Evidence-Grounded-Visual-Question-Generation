#!/usr/bin/env python3
"""CLI for Phase 4B Online UVFC + Violation GRPO (static / dynamic)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vvqg.training.online_uvfc_relative_grpo import run_online  # noqa: E402

PATH_KEYS = (
    "base_model",
    "sft_checkpoint",
    "uvfc_dir",
    "flat_model",
    "train_manifest",
    "output_dir",
    "dev_manifest",
    "greedy_out",
    "sampled_out",
    "guard_calibration_manifest",
    "guard_sft_checkpoint",
)


def validate_inputs(cfg: dict) -> None:
    """Fail fast before G2 if an input artifact is missing or invalid."""
    import torch

    adapter = Path(cfg["sft_checkpoint"])
    if not (adapter / "adapter_config.json").is_file():
        raise SystemExit(f"init adapter not found: {adapter}")
    weights = Path(cfg["uvfc_dir"]) / "weights.pt"
    if not weights.is_file():
        raise SystemExit(f"UVFC weights not found: {weights}")
    w = torch.load(weights, map_location="cpu")
    for key in ("v", "u"):
        if not torch.isfinite(w[key]).all() or int((w[key] > 0).sum()) == 0:
            raise SystemExit(f"UVFC '{key}' is invalid in {weights}")
    guard_sft = Path(cfg.get("guard_sft_checkpoint", cfg["sft_checkpoint"]))
    if not (guard_sft / "adapter_config.json").is_file():
        raise SystemExit(f"guard SFT adapter not found: {guard_sft}")


def resolve(cfg: dict) -> dict:
    for key in PATH_KEYS:
        value = cfg.get(key)
        if value and not str(value).startswith("/") and key != "base_model" and key != "flat_model":
            cfg[key] = str(ROOT / value)
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    import yaml

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg.setdefault("seed", 42)
    cfg.setdefault("instruction", "Generate a question about this image.")
    cfg.setdefault("attn_implementation", "sdpa")
    cfg.setdefault("inner_epochs", 2)
    cfg.setdefault("generation_batch_size", 8)
    cfg.setdefault("flat_batch_size", 32)
    cfg.setdefault("log_ratio_clamp", 20.0)
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.output_dir is not None:
        cfg["output_dir"] = str(args.output_dir)
    resolve(cfg)
    validate_inputs(cfg)
    summary = run_online(cfg)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
