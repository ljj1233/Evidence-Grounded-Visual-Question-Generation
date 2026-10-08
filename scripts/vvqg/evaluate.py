#!/usr/bin/env python3
"""Canonical active VQG evaluation entry point.

Computes only the supported active metrics. Historical specificity protocols are
archived under ``archive/evaluation_legacy`` and are intentionally NOT offered
here. BERTScore / PAC-S / VQAScore are excluded.

Examples
--------
  .venv-vllm-cu128-v011/bin/python scripts/vvqg/evaluate.py \
      --predictions outputs/evaluation/legacy_val_debug/sft_baseline_val_debug.jsonl \
      --references data/vqav2/splits/val_debug.jsonl \
      --metrics bleu4,qbleu4_variant,cider,distinct2,dup_rate \
      --out outputs/evaluation_sft
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from vvqg.evaluation.common.schemas import PredictionRecord, ReferenceRecord
from vvqg.evaluation.common.bootstrap import bootstrap_ci, paired_corpus_bootstrap  # noqa: E402
from vvqg.evaluation.common.io import read_jsonl, write_json, write_jsonl  # noqa: E402
from vvqg.evaluation.metrics import (  # noqa: E402
    cider_score,
    corpus_bleu4,
    distinct_2,
    duplicate_rate,
    most_frequent,
    qbleu4_variant,
)

from vvqg.evaluation.metrics.bleu import bleu_statistics, bleu_from_statistics
from vvqg.evaluation.metrics.qbleu import qbleu_statistics, qbleu_from_statistics
from vvqg.evaluation.metrics.cider import cider_scores
from vvqg.evaluation.metrics.diversity import diversity_summary
import numpy as np

SUPPORTED_METRICS = ("bleu4", "qbleu4_variant", "cider", "distinct2", "dup_rate", "diversity", "clipscore")


def load_references(path: Path) -> dict[str, list[str]]:
    refs: dict[str, list[str]] = {}
    rows: list[ReferenceRecord] = read_jsonl(path)
    for row in rows:
        refs.setdefault(str(row["image_id"]), []).append(row["question"])
    return refs


def resolve_image(path: str) -> str:
    candidate = Path(path)
    return str(candidate if candidate.is_absolute() else REPO_ROOT / candidate)


def evaluate_system(records: list[PredictionRecord], references: Optional[dict[str, list[str]]], metrics: list[str], device: str, batch_size: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = [record["question"] for record in records]
    summary: dict[str, Any] = {"n": len(records)}
    per_sample: list[dict[str, Any]] = [{"image_id": record["image_id"], "question": record["question"]} for record in records]

    refs = [references[str(record["image_id"])] for record in records] if references else None
    if set(metrics) & {"bleu4", "qbleu4_variant", "cider"} and refs is None:
        raise SystemExit("--references is required for bleu4 / qbleu4_variant / cider")
    if "bleu4" in metrics:
        if refs is None:
            raise SystemExit("--references is required for bleu4 / qbleu4_variant / cider")
        summary["corpus_bleu4"] = corpus_bleu4(candidates, refs)
        sentence = [corpus_bleu4([c], [r]) for c, r in zip(candidates, refs)]
        for row, value in zip(per_sample, sentence):
            row["bleu4"] = value
    if "qbleu4_variant" in metrics:
        summary["corpus_qbleu4_variant"] = qbleu4_variant(candidates, refs)
    if "cider" in metrics:
        summary["corpus_cider"] = cider_score(candidates, refs)
    if "distinct2" in metrics:
        summary["distinct_2"] = distinct_2(candidates)
    if "dup_rate" in metrics:
        summary["normalized_excess_duplicate_rate"] = duplicate_rate(candidates)
        summary["most_frequent"] = most_frequent(candidates, top=10)
    if "diversity" in metrics:
        summary["diversity"] = diversity_summary(records)
    if "clipscore" in metrics:
        images = [resolve_image(record["image_path"]) for record in records]
        from vvqg.evaluation.metrics import ClipScorer

        scorer = ClipScorer(device=device, batch_size=batch_size)
        scores = scorer.score_pairs(list(zip(images, candidates)))
        for row, value in zip(per_sample, scores):
            row["clipscore_raw_question"] = value
        summary["clipscore_raw_question"] = bootstrap_ci(scores)
    return summary, per_sample


def compare_reference_metrics(left: list[PredictionRecord], right: list[PredictionRecord], references: dict[str, list[str]], resamples: int = 5000, seed: int = 42) -> dict[str, Any]:
    """Image-paired corpus bootstrap; CIDEr intervals condition on full-corpus IDF."""
    aligned = [sorted(records, key=lambda r: str(r["image_id"])) for records in (left, right)]
    left_ids, right_ids = [[str(r["image_id"]) for r in records] for records in aligned]
    assert left_ids == right_ids and len(set(left_ids)) == len(left_ids)
    samples = []
    for records in aligned:
        questions = [r["question"] for r in records]
        refs = [references[str(r["image_id"])] for r in records]
        stats = bleu_statistics(questions, refs)
        cider = cider_scores(questions, refs)
        if cider is None:
            raise RuntimeError("CIDEr requires pycocoevalcap in the metrics environment")
        samples.append({"bleu4": stats, "qbleu4_variant": qbleu_statistics(questions, refs, stats),
                        "cider": np.asarray(cider)[:, None]})
    n = len(left)
    reducers = {"bleu4": bleu_from_statistics, "qbleu4_variant": qbleu_from_statistics,
                "cider": lambda totals: totals[..., 0] / n}
    reports = {name: paired_corpus_bootstrap(samples[0][name], samples[1][name], reducer, resamples, seed)
               for name, reducer in reducers.items()}
    result = {side: {"n": n} for side in ("left", "right")}
    result["delta"] = {}
    for name, report in reports.items():
        for side in ("left", "right"):
            result[side]["corpus_" + name] = report[side]["estimate"]
            result[side][name + "_bootstrap_ci"] = report[side]
        result["delta"]["corpus_" + name] = report["delta"]
    result["bootstrap_protocol"] = {
        "unit": "image_id; paired systems use identical sampled image indices",
        "bleu": "sum resampled clipped counts, denominators and lengths, then recompute corpus score",
        "qbleu": "same corpus bootstrap using question-type-weighted clipped counts; frozen local variant",
        "cider": "conditional on IDF estimated once from the full fixed reference corpus; resample per-image scores without recomputing IDF",
        "ci": "percentile 95%, numpy linear quantiles", "seed": seed, "resamples": resamples,
        "delta_direction": "left minus right",
    }
    return result


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonical active VQG evaluation (BLEU-4, Q-BLEU4-variant, CIDEr, Distinct-2, dup rate, CLIPScore adaptation).")
    parser.add_argument("--predictions", type=Path, required=True, help="prediction JSONL with image_id + question")
    parser.add_argument("--references", type=Path, default=None, help="reference JSONL with image_id + question (required for reference metrics)")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--metrics", default="bleu4,qbleu4_variant,cider,distinct2,dup_rate", help=f"comma-separated subset of {','.join(SUPPORTED_METRICS)}")
    parser.add_argument("--device", default="cpu", help="CLIPScore device (cpu/cuda)")
    parser.add_argument("--batch-size", type=int, default=64, help="CLIPScore batch size")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    unknown = [m for m in metrics if m not in SUPPORTED_METRICS]
    if unknown:
        raise SystemExit(f"unsupported metrics: {unknown}; supported: {SUPPORTED_METRICS}")
    references = load_references(args.references) if args.references else None
    records = read_jsonl(args.predictions)
    summary, per_sample = evaluate_system(records, references, metrics, args.device, args.batch_size)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "system_summary.json", summary)
    write_jsonl(args.out / "per_sample_metrics.jsonl", per_sample)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
