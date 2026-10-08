#!/usr/bin/env python3
"""Formal offline DPO-v1 trainer CLI — full 1 epoch over all eligible pairs."""

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

from vvqg.dpo_pilot.trainer import run_dpo  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", type=Path, required=True)
    ap.add_argument("--pairs-train", type=Path, required=True)
    ap.add_argument("--base-model", default=local_model_path('Qwen3-VL-8B-Instruct'))
    ap.add_argument("--base-checkpoint", type=Path, default=ROOT / "outputs/phase5c_qse_relativeguard/checkpoints/step_final")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint-every", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = one full epoch")
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args()
    out = args.report_dir.resolve() / "dpo"
    cfg = {"output_dir": str(out), "pairs_train": str(args.pairs_train),
           "base_model": args.base_model, "base_checkpoint": str(args.base_checkpoint),
           "max_steps": args.max_steps, "batch_size": args.batch_size, "lr": args.lr, "beta": args.beta,
           "seed": args.seed, "resume": str(args.resume) if args.resume else None,
           "checkpoint_steps": list(range(args.checkpoint_every, 1000000, args.checkpoint_every))}
    summary = run_dpo(cfg)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
