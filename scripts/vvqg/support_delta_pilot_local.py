#!/usr/bin/env python3
"""Support-Delta GRPO — fixed-output generation + local Flat metrics on the 256 pilot set."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vvqg.paths import local_model_path

from vvqg.evidence_constrained.reporting import global_duplicate_rate, generic_rate, question_stats, template_dominance  # noqa: E402
from vvqg.generation.inference import generate_questions, model_prompt  # noqa: E402
from vvqg.rewards.support.structured_flat import StructuredFlatVerifier, support_summary  # noqa: E402
from vvqg.training.online_uvfc_relative_grpo import build_frozen_model  # noqa: E402

INSTRUCTION = "Generate a question about this image."
INTERVENTION = ROOT / "reports/published_baseline_intervention_20261004T032525Z/transformed_images"


def condition_path(pilot_row: dict, condition: str) -> str:
    if condition == "original":
        return str(pilot_row["image_path"])
    return str((INTERVENTION / condition / f"{pilot_row['image_id']}.jpg").relative_to(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-manifest", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--checkpoint", action="append", required=True, help="tag=path")
    ap.add_argument("--base-model", default=local_model_path('Qwen3-VL-8B-Instruct'))
    ap.add_argument("--flat-model", default=local_model_path('Qwen2.5-VL-7B-Instruct'))
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    import torch

    pilot = [json.loads(line) for line in args.pilot_manifest.read_text().splitlines() if line.strip()]
    for row in pilot:
        row["image_id"] = str(row["image_id"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flat = StructuredFlatVerifier(args.flat_model, device, batch_size=32)
    table = []
    for spec in args.checkpoint:
        tag, _, path = spec.partition("=")
        model, processor = build_frozen_model(args.base_model, path, "sdpa")
        model.to(device).eval()
        prompt = model_prompt(processor, INSTRUCTION)
        for condition in ("original", "blurred", "shuffled"):
            images = [condition_path(row, condition) for row in pilot]
            generated = generate_questions(model, processor, images, prompt, num_samples=1, temperature=0.0, top_p=1.0,
                                           max_new_tokens=32, seed=42, batch_size=args.batch_size, desc=f"{tag}-{condition}")
            predictions = [{"image_id": row["image_id"], "image": row["image_path"], "question": group[0].strip(),
                            "condition": condition, "system_id": tag, "checkpoint": path} for row, group in zip(pilot, generated)]
            flat_rows = [{"image_id": row["image_id"], "image": row["image"], "question": row["question"]} for row in predictions]
            parsed = flat.verify_structured(flat_rows)
            floors, states = [], []
            for row, parse in zip(flat_rows, parsed):
                summary = support_summary(parse)
                row.update(summary)
                floors.append(float(summary["support_floor"])); states.append(summary["flat_state"])
            n = len(flat_rows)
            questions = [row["question"] for row in flat_rows]
            image_ids = [row["image_id"] for row in flat_rows]
            stats = question_stats(processor, questions, image_ids)
            table.append({
                "tag": tag, "condition": condition, "n": n,
                "flat_supported_rate": sum(state == "NO_CONFIRMED_VIOLATION" for state in states) / n,
                "flat_unsupported_rate": sum(state == "CONFIRMED_VIOLATION" for state in states) / n,
                "flat_uncertain_rate": sum(state == "UNCERTAIN" for state in states) / n,
                "flat_parser_failure_rate": sum(state == "FAILED" for state in states) / n,
                "flat_support_floor_mean": sum(floors) / n,
                "duplicate_rate_global": global_duplicate_rate(questions),
                "generic_rate": generic_rate(questions),
                "template_dominance": template_dominance(questions),
                "mean_token_length": stats["mean_token_length"],
                "malformed_rate": sum(len(str(q).strip()) < 2 for q in questions) / n,
            })
            (args.outdir / f"predictions_{tag}_{condition}.jsonl").write_text(
                "".join(json.dumps({**pred, **{k: flat_row[k] for k in ("flat_state", "support_floor", "support_mean") if k in flat_row}}, ensure_ascii=False) + "\n"
                        for pred, flat_row in zip(predictions, flat_rows)), encoding="utf-8")
            print(json.dumps({"tag": tag, "condition": condition, **table[-1]}), flush=True)
        del model
        torch.cuda.empty_cache()

    with (args.outdir / "CHECKPOINT_METRICS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0].keys()))
        writer.writeheader(); writer.writerows(table)
    (args.outdir / "LOCAL_METRICS.json").write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
