#!/usr/bin/env python3
"""Reference sanity for the controlled batch-8 DPO run.

Checks, on one effective batch, that the freshly initialized policy equals the
frozen reference: DPO margin ≈ 0 and loss ≈ ln 2.  Writes REFERENCE_SANITY.json.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vvqg.paths import local_model_path
from vvqg.dpo_pilot.trainer import _seq_logprobs  # noqa: E402
from vvqg.training.fixed_group_relative import build_trainable_model  # noqa: E402
from vvqg.training.online_uvfc_relative_grpo import build_frozen_model  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--base-model", default=local_model_path('Qwen3-VL-8B-Instruct'))
    ap.add_argument("--base-checkpoint", type=Path, default=ROOT / "outputs/phase5c_qse_relativeguard/checkpoints/step_final")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import torch

    device = torch.device("cuda")
    pairs = [json.loads(l) for l in args.pairs.read_text().splitlines() if l.strip()]
    random.Random(args.seed).shuffle(pairs)
    batch = pairs[: args.batch_size]
    instr = "Generate a question about this image."
    policy, processor, _ = build_trainable_model(args.base_model, str(args.base_checkpoint), "sdpa")
    policy.to(device)
    reference, rproc = build_frozen_model(args.base_model, str(args.base_checkpoint), "sdpa")
    reference.to(device)
    c, r = _seq_logprobs(policy, processor, batch, instr, device, False)
    rc, rr = _seq_logprobs(reference, rproc, batch, instr, device, False)
    margins = [args.beta * ((float(a) - float(b)) - (float(x) - float(y))) for a, x, b, y in zip(c, r, rc, rr)]
    loss = sum(math.log(1 + math.exp(-m)) for m in margins) / len(margins)
    report = {"batch_size": args.batch_size, "beta": args.beta, "n_pairs": len(batch),
              "max_abs_margin": max(abs(m) for m in margins), "loss": loss, "ln2": math.log(2),
              "policy_equals_reference": max(abs(m) for m in margins) < 1e-4 and abs(loss - math.log(2)) < 1e-4,
              "chosen_logratio_mean": sum(float(a) - float(b) for a, b in zip(c, rc)) / len(batch)}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["policy_equals_reference"]:
        raise SystemExit("REFERENCE SANITY FAILED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
