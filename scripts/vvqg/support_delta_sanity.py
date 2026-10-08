#!/usr/bin/env python3
"""Support-Delta GRPO — reward sanity check (local, frozen pi_0, no training)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vvqg.paths import local_model_path

from vvqg.generation.inference import generate_questions, model_prompt  # noqa: E402
from vvqg.rewards.support.structured_flat import StructuredFlatVerifier, support_summary  # noqa: E402
from vvqg.rewards.visual_information import uvfc as uvfc_mod  # noqa: E402
from vvqg.support_delta.reward import (  # noqa: E402
    VIOLATION,
    coverage_guard,
    duplicate_flags,
    group_advantages,
    support_delta_reward,
)
from vvqg.training.fixed_group_relative import norm_question  # noqa: E402
from vvqg.training.online_uvfc_relative_grpo import build_frozen_model  # noqa: E402

INSTRUCTION = "Generate a question about this image."
K = 8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--base-model", default=local_model_path('Qwen3-VL-8B-Instruct'))
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/phase5c_qse_relativeguard/checkpoints/step_final")
    ap.add_argument("--flat-model", default=local_model_path('Qwen2.5-VL-7B-Instruct'))
    ap.add_argument("--uvfc-dir", type=Path, default=ROOT / "outputs/phase26/vfc")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--n-images", type=int, default=None)
    ap.add_argument("--floor", type=float, default=0.5)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch

    anchors = {row["image_id"]: row for row in (json.loads(line) for line in args.anchors.read_text().splitlines() if line.strip())}
    image_ids = sorted(anchors)
    if args.n_images:
        image_ids = image_ids[: args.n_images]
    rows = [anchors[image_id] for image_id in image_ids]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = build_frozen_model(args.base_model, str(args.checkpoint), "sdpa")
    model.to(device).eval()
    prompt = model_prompt(processor, INSTRUCTION)
    flat = StructuredFlatVerifier(args.flat_model, device, batch_size=32)
    frozen = uvfc_mod.load_uvfc(args.uvfc_dir, device)
    started = time.perf_counter()

    generated = generate_questions(model, processor, [row["image"] for row in rows], prompt, num_samples=K,
                                   temperature=args.temperature, top_p=args.top_p, max_new_tokens=args.max_new_tokens,
                                   seed=42, batch_size=8)
    rollout_rows = []
    for row, questions in zip(rows, generated):
        for generation_id, question in enumerate(questions):
            rollout_rows.append({"image_id": row["image_id"], "image": row["image"], "question": question.strip(),
                                 "generation_id": generation_id})
    parsed = flat.verify_structured(rollout_rows)
    for row, parse in zip(rollout_rows, parsed):
        summary = support_summary(parse)
        row.update(summary)
        row["parser_status"] = parse.get("status")
    del flat
    torch.cuda.empty_cache()
    for start in range(0, len(rollout_rows), 8):
        chunk = rollout_rows[start : start + 8]
        for row, score in zip(chunk, uvfc_mod.score_uvfc(model, processor, chunk, frozen, device)):
            row["UVFC"] = float(score["UVFC"])

    by_image: dict[str, list[dict]] = {}
    for row in rollout_rows:
        by_image.setdefault(row["image_id"], []).append(row)

    component_rows, group_variance = [], []
    reward_std_by_group, zero_variance_groups, guard_triggers, dup_penalties, malformed_penalties = [], 0, 0, 0, 0
    total_rollouts = 0
    for image_id, members in by_image.items():
        q0 = anchors[image_id]
        questions = [member["question"] for member in members]
        duplicates = duplicate_flags(questions)
        uvfc_group = [float(q0["UVFC"])] + [member["UVFC"] for member in members]
        rewards = []
        for index, member in enumerate(members):
            malformed = len(member["question"].strip()) < 2
            parser_failed = member.get("parser_status") != "ok"
            result = support_delta_reward(
                support_floor=member.get("support_floor"), flat_state=member.get("flat_state"),
                malformed=malformed, q0_score=float(q0["support_score"]), uvfc_group=uvfc_group,
                q0_index=0, rollout_index=index + 1, duplicate=duplicates[index], parser_failed=parser_failed,
            )
            member.update(result)
            rewards.append(result["reward"])
            guard_triggers += 1 if result["coverage_guard"] > 0 else 0
            dup_penalties += 1 if result["duplicate_penalty"] > 0 else 0
            malformed_penalties += 1 if result["malformed_penalty"] > 0 else 0
            total_rollouts += 1
            component_rows.append({"image_id": image_id, **{key: member.get(key) for key in
                ("generation_id", "question", "flat_state", "support_floor", "support_score", "delta_support",
                 "coverage_guard", "duplicate_penalty", "malformed_penalty", "reward", "UVFC")}})
        advantages, std, zero = group_advantages(rewards)
        for member, advantage in zip(members, advantages):
            member["advantage"] = advantage
        reward_std_by_group.append(std)
        zero_variance_groups += 1 if zero else 0
        group_variance.append({"image_id": image_id, "reward_std": std, "zero_variance": zero,
                               "reward_mean": float(np.mean(rewards)), "q0_support_score": float(q0["support_score"])})

    support_scores = [row["support_score"] for row in rollout_rows]
    flat_states = [row.get("flat_state") for row in rollout_rows]
    non_violation = [row["support_score"] for row in rollout_rows if row.get("flat_state") != VIOLATION and row.get("flat_state") != "FAILED"]
    violation = [row["support_score"] for row in rollout_rows if row.get("flat_state") == VIOLATION]
    delta_support = np.asarray([row["delta_support"] for row in rollout_rows])
    lengths = [len(processor.tokenizer(row["question"], add_special_tokens=False)["input_ids"]) for row in rollout_rows]
    from vvqg.evidence_constrained.audit import spearman

    template_counts = Counter(norm_question(row["question"]) for row in rollout_rows)
    generic_pat = Counter()
    diagnostics = {
        "n_images": len(by_image),
        "n_rollouts": total_rollouts,
        "supported_score_mean": float(np.mean(non_violation)) if non_violation else None,
        "unsupported_score_mean": float(np.mean(violation)) if violation else None,
        "supported_gt_unsupported": (float(np.mean(non_violation)) > float(np.mean(violation))) if non_violation and violation else None,
        "delta_support_positive": int((delta_support > 0).sum()),
        "delta_support_negative": int((delta_support < 0).sum()),
        "delta_support_zero": int((delta_support == 0).sum()),
        "reward_mean": float(np.mean([row["reward"] for row in rollout_rows])),
        "reward_std": float(np.std([row["reward"] for row in rollout_rows])),
        "mean_group_reward_std": float(np.mean(reward_std_by_group)),
        "zero_variance_group_rate": zero_variance_groups / len(by_image),
        "coverage_guard_rate": guard_triggers / total_rollouts,
        "duplicate_penalty_rate": dup_penalties / total_rollouts,
        "malformed_rate": malformed_penalties / total_rollouts,
        "template_dominance": max(template_counts.values()) / total_rollouts,
        "reward_length_spearman": spearman([row["reward"] for row in rollout_rows], lengths),
        "q0_support_score_mean": float(np.mean([row["support_score"] for row in rows])),
        "q0_support_floor_mean": float(np.mean([float(row["support_floor"]) for row in rows if row.get("support_floor") is not None])),
        "q0_flat_state_counts": dict(Counter(row.get("flat_state") for row in rows)),
        "ai_flat_state_counts": dict(Counter(flat_states)),
        "wall_seconds": time.perf_counter() - started,
    }
    checks = {
        "supported_gt_unsupported": diagnostics["supported_gt_unsupported"],
        "delta_has_both_signs": diagnostics["delta_support_positive"] > 0 and diagnostics["delta_support_negative"] > 0,
        "not_all_reward_identical": diagnostics["reward_std"] > 1e-6,
        "zero_variance_below_0.30": diagnostics["zero_variance_group_rate"] < 0.30,
        "coverage_guard_below_0.50": diagnostics["coverage_guard_rate"] < 0.50,
        "malformed_below_0.05": diagnostics["malformed_rate"] < 0.05,
        "template_not_collapsed": diagnostics["template_dominance"] <= 0.20,
    }
    status = "PASS_REWARD_SANITY" if all(value for value in checks.values()) else "FAIL_REWARD_SANITY"

    with (args.outdir / "REWARD_COMPONENTS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(component_rows[0].keys()))
        writer.writeheader()
        writer.writerows(component_rows)
    with (args.outdir / "GROUP_VARIANCE.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_variance[0].keys()))
        writer.writeheader()
        writer.writerows(group_variance)
    (args.outdir / "REWARD_SANITY.json").write_text(json.dumps({"status": status, "checks": checks, "diagnostics": diagnostics}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "checks": checks, "diagnostics": diagnostics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
