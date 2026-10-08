#!/usr/bin/env python3
"""Assert the online UVFC path reproduces the frozen Phase-2.6 scores.

Scores N candidates from `scores4000.jsonl` with the online `score_uvfc` (same
batch size as the frozen run) and reports UVFC error and active-feature equality.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vvqg.paths import local_model_path

import torch  # noqa: E402

from vvqg.rewards.visual_information import uvfc as uvfc_mod  # noqa: E402
from vvqg.training.online_uvfc_grpo import build_frozen_model  # noqa: E402

BASE = local_model_path('Qwen3-VL-8B-Instruct')
ADAPTER = ROOT / "outputs/checkpoints/qwen3vl8b_lora_r16_vqav2_50k/checkpoint-final"
VFC = ROOT / "outputs/phase26/vfc"
SCORES = VFC / "scores4000.jsonl"
CANDIDATES = ROOT / "outputs/reward_calibration/sft_candidates_4x.jsonl"
CANON = ROOT / "outputs/phase25_offline_reward_audit/candidates4000.jsonl"


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frozen = uvfc_mod.load_uvfc(VFC, device)
    model, processor = build_frozen_model(BASE, str(ADAPTER))
    model.to(device)
    reference = {row["candidate_id"]: row for row in (json.loads(line) for line in SCORES.read_text(encoding="utf-8").splitlines() if line.strip())}
    # Replicate the frozen score stage's iteration order (sft_candidates_4x order, batch 4).
    pool = [json.loads(line) for line in CANDIDATES.read_text(encoding="utf-8").splitlines() if line.strip()]
    canon = {
        (str(row["image_id"]), row["question"], int(row["generation_id"])): row["candidate_id"]
        for row in (json.loads(line) for line in CANON.read_text(encoding="utf-8").splitlines() if line.strip())
    }
    batch_size = 4
    max_uvfc_error = 0.0
    feature_mismatches = 0
    scored = 0
    for start in range(0, 32, batch_size):
        chunk = [
            {
                "image": row["image"],
                "question": row["question"],
                "candidate_id": canon[(str(row["image_id"]), row["question"], int(row["generation_id"]))],
            }
            for row in pool[start : start + batch_size]
        ]
        online = uvfc_mod.score_uvfc(model, processor, chunk, frozen, device)
        for row, result in zip(chunk, online):
            expected = reference[row["candidate_id"]]
            max_uvfc_error = max(max_uvfc_error, abs(result["UVFC"] - float(expected["UVFC"])))
            if set(result["active_features"]) != set(expected["active_features"]):
                feature_mismatches += 1
            scored += 1
    report = {
        "candidates": scored,
        "batch_size": batch_size,
        "max_abs_uvfc_error": max_uvfc_error,
        "active_feature_mismatches": feature_mismatches,
        "uvfc_tolerance_1e-4": max_uvfc_error <= 1e-4,
        "active_features_exact": feature_mismatches == 0,
        "pass": max_uvfc_error <= 1e-4 and feature_mismatches == 0,
    }
    out = VFC / "online_verification.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
