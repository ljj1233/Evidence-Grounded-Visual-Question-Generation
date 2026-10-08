"""Fixed-output local evaluation for CER (frozen Flat verifier, no external API).

Questions are generated from a checkpoint under a condition image (original /
shuffled / blurred) and always Flat-verified against the *original* image, so the
original-shuffled gap measures visual dependence.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

from vvqg.evidence_constrained.reporting import question_stats
from vvqg.generation.inference import generate_questions, model_prompt
from vvqg.rewards.support.structured_flat import StructuredFlatVerifier, support_summary
from vvqg.training.online_uvfc_grpo import build_frozen_model

CONDITION_KEYS = {
    "original": ("original_path", "image"),
    "shuffled": ("shuffled_path", "shuffled_image_path"),
    "blurred": ("blurred_path", "blurred_image_path"),
}


def condition_image(row: dict[str, Any], condition: str) -> str:
    for key in CONDITION_KEYS[condition]:
        if row.get(key):
            return str(row[key])
    raise KeyError(f"no path for condition {condition} in {sorted(row)}")


def generate_condition(
    rows: Sequence[dict[str, Any]],
    condition: str,
    checkpoint: str,
    base_model: str,
    out_dir: Path,
    instruction: str,
    seed: int = 42,
    batch_size: int = 8,
    max_new_tokens: int = 32,
    attn_implementation: str = "sdpa",
) -> list[dict[str, Any]]:
    import torch

    images = [condition_image(row, condition) for row in rows]
    model, processor = build_frozen_model(base_model, checkpoint, attn_implementation)
    model.to("cuda").eval()
    processor.tokenizer.padding_side = "left"
    prompt = model_prompt(processor, instruction)
    started = time.perf_counter()
    generated = generate_questions(
        model, processor, images, prompt, num_samples=1, temperature=0.0, top_p=1.0,
        max_new_tokens=max_new_tokens, seed=seed, batch_size=batch_size, desc=f"gen-{condition}",
    )
    del model
    torch.cuda.empty_cache()
    out = [
        {
            "image_id": str(row["image_id"]),
            "condition": condition,
            "prediction_image": image,
            "evaluator_image": condition_image(row, "original"),
            "question": questions[0].strip(),
            "checkpoint": str(checkpoint),
            "seed": seed,
        }
        for row, image, questions in zip(rows, images, generated)
    ]
    (out_dir / f"generation_{condition}.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in out), encoding="utf-8"
    )
    (out_dir / f"runtime_{condition}.json").write_text(
        json.dumps({"condition": condition, "checkpoint": str(checkpoint), "n": len(rows), "wall_seconds": time.perf_counter() - started}, indent=2) + "\n",
        encoding="utf-8",
    )
    return out


def flat_score(
    rows: Sequence[dict[str, Any]],
    flat: StructuredFlatVerifier,
    out_dir: Path,
    condition: str,
) -> dict[str, float]:
    scored = [{"image_id": str(r["image_id"]), "image": r["evaluator_image"], "question": r["question"]} for r in rows]
    parsed = flat.verify_structured(scored)
    states = []
    for row, parse in zip(scored, parsed):
        summary = support_summary(parse)
        row.update(summary)
        row["parser_status"] = parse.get("status")
        states.append(summary.get("flat_state"))
    (out_dir / f"flat_{condition}.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scored), encoding="utf-8"
    )
    total = len(states)
    return {
        "condition": condition,
        "n": total,
        "flat_supported_rate": sum(state == "NO_CONFIRMED_VIOLATION" for state in states) / total if total else 0.0,
        "flat_unsupported_rate": sum(state == "CONFIRMED_VIOLATION" for state in states) / total if total else 0.0,
        "flat_uncertain_rate": sum(state == "UNCERTAIN" for state in states) / total if total else 0.0,
        "flat_parser_failure_rate": sum(state == "FAILED" for state in states) / total if total else 0.0,
        "flat_support_floor_mean": sum(float(row["support_floor"]) for row in scored) / total if total else 0.0,
        "questions": [row["question"] for row in scored],
        "image_ids": [row["image_id"] for row in scored],
    }


def checkpoint_report(
    rows: Sequence[dict[str, Any]],
    checkpoint: str,
    base_model: str,
    flat_model: str,
    out_dir: Path,
    instruction: str,
    seed: int = 42,
    batch_size: int = 8,
    processor: Any = None,
) -> dict[str, Any]:
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flat = StructuredFlatVerifier(flat_model, device, batch_size=max(batch_size, 8))
    report: dict[str, Any] = {"checkpoint": str(checkpoint), "n_images": len(rows), "conditions": {}}
    for condition in ("original", "shuffled", "blurred"):
        generated = generate_condition(rows, condition, checkpoint, base_model, out_dir, instruction, seed=seed, batch_size=batch_size)
        scored = flat_score(generated, flat, out_dir, condition)
        report["conditions"][condition] = {key: value for key, value in scored.items() if key not in ("questions", "image_ids")}
        if processor is not None:
            report["conditions"][condition].update(question_stats(processor, scored["questions"], scored["image_ids"]))
    report["original_shuffled_support_gap"] = (
        report["conditions"]["original"]["flat_supported_rate"] - report["conditions"]["shuffled"]["flat_supported_rate"]
    )
    report["original_blurred_support_gap"] = (
        report["conditions"]["original"]["flat_supported_rate"] - report["conditions"]["blurred"]["flat_supported_rate"]
    )
    (out_dir / "checkpoint_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
