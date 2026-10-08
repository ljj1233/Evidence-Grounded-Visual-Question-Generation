#!/usr/bin/env python3
"""Support-Delta GRPO trainer CLI."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vvqg.paths import local_model_path

from vvqg.support_delta.trainer import run_pilot  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--train-manifest", type=Path, required=True)
    ap.add_argument("--base-model", default=local_model_path('Qwen3-VL-8B-Instruct'))
    ap.add_argument("--base-checkpoint", type=Path, default=ROOT / "outputs/phase5c_qse_relativeguard/checkpoints/step_final")
    ap.add_argument("--flat-model", default=local_model_path('Qwen2.5-VL-7B-Instruct'))
    ap.add_argument("--uvfc-dir", type=Path, default=ROOT / "outputs/phase26/vfc")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--groups-per-step", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--inner-epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--beta-kl", type=float, default=0.02)
    ap.add_argument("--clip-epsilon", type=float, default=0.2)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--checkpoint-steps", type=int, nargs="*", default=[0, 50, 100, 150, 200, 250, 300])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = {
        "output_dir": str(args.output_dir), "train_manifest": str(args.train_manifest),
        "base_model": args.base_model, "base_checkpoint": str(args.base_checkpoint),
        "flat_model": args.flat_model, "uvfc_dir": str(args.uvfc_dir),
        "group_size": args.group_size, "groups_per_step": args.groups_per_step, "max_steps": args.max_steps,
        "inner_epochs": args.inner_epochs, "lr": args.lr, "beta_kl": args.beta_kl,
        "clip_epsilon": args.clip_epsilon, "max_grad_norm": args.max_grad_norm,
        "checkpoint_steps": args.checkpoint_steps, "seed": args.seed,
    }
    summary = run_pilot(cfg)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
