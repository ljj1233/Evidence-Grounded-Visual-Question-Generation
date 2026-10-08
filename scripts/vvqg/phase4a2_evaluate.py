#!/usr/bin/env python3
"""Phase 4A.2 evaluation runner: traditional, SeeQ visual-centric, GPT-4o support,
and semantic diversity for the 1:1 replay model on development1000."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vvqg.paths import local_model_path

ROOT = Path(__file__).resolve().parents[2]
METRICS_PY = ROOT / ".venv-vllm-cu128-v011/bin/python"
GEN_PY = ROOT / ".venv/bin/python"
REFERENCES = ROOT / "outputs/evaluation/sft_greedy/val2014_sample1000.jsonl"
ZERO_SHOT_SAMPLED = ROOT / "outputs/phase23/baselines/zero_shot/sampled_4x.jsonl"
DIMENSIONS = ("visual_search_difficulty", "visual_evidence_coverage", "visual_context_reasoning", "visual_spatial_reasoning")


def run(command: list[str]) -> None:
    print("$", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True)


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if Path(path).is_file() else []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--retry", action="store_true")
    parser.add_argument("--label", default="Replay")
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    evaluation = run_dir / "evaluation"
    generation = run_dir / "generation"
    greedy = generation / "greedy1000.jsonl"
    sampled = generation / "sampled4x1000.jsonl"

    seeq_cmd = [
        str(METRICS_PY), str(ROOT / "scripts/vvqg/evaluate_visual_centric.py"),
        "--predictions", str(greedy), "--output", str(evaluation / "seeq/scores.jsonl"),
        "--model", "gpt-5.6-luna", "--allow-variant", "--concurrency", str(args.concurrency),
        "--label", f"SeeQ {args.label}",
    ]
    if args.retry:
        seeq_cmd.append("--retry-failures")
    run(seeq_cmd)
    support_cmd = [
        str(METRICS_PY), str(ROOT / "scripts/vvqg/evaluate_visual_support.py"),
        "--predictions", f"{args.variant}={greedy}", "--out", str(evaluation / "support"),
        "--model", "gpt-4o", "--concurrency", str(args.concurrency),
        "--label", f"Support {args.label}",
    ]
    if args.retry:
        support_cmd.append("--retry-failures")
    run(support_cmd)
    run([
        str(METRICS_PY), str(ROOT / "scripts/vvqg/evaluate.py"),
        "--predictions", str(greedy), "--references", str(REFERENCES),
        "--metrics", "bleu4,qbleu4_variant,cider,distinct2,dup_rate",
        "--out", str(evaluation / "traditional"),
    ])
    run([
        str(METRICS_PY), str(ROOT / "scripts/vvqg/evaluate.py"),
        "--predictions", str(sampled), "--references", str(REFERENCES),
        "--metrics", "distinct2,dup_rate", "--out", str(evaluation / "diversity_sampled"),
    ])
    run([
        str(GEN_PY), str(ROOT / "scripts/vvqg/evaluate_seeq_diversity.py"),
        "--model", local_model_path('Qwen3-Embedding-4B'),
        "--zero-shot", str(ZERO_SHOT_SAMPLED), "--sft", str(sampled),
        "--output", str(evaluation / "diversity"), "--batch-size", "32",
        "--label", f"SemDiv {args.label}",
    ])

    seeq_rows = [r for r in read(evaluation / "seeq/scores.jsonl") if r.get("status") == "ok"]
    visual_centric = {key: sum(r["scores"][key] for r in seeq_rows) / len(seeq_rows) / 5.0 for key in DIMENSIONS} | {"valid": len(seeq_rows)}
    visual_centric["seeq_summary"] = {key: visual_centric[key] for key in DIMENSIONS}
    (evaluation / "visual_centric.json").write_text(json.dumps(visual_centric, indent=2) + "\n", encoding="utf-8")

    support_rows = [r for r in read(evaluation / f"support/{args.variant}.jsonl") if r.get("status") == "ok"]
    visual_support = {
        label: sum(r.get("overall_visual_support") == label for r in support_rows) / len(support_rows)
        for label in ("supported", "unsupported", "uncertain")
    } | {"valid": len(support_rows)}
    visual_support["failure_type_rates"] = {
        ftype: sum(r.get("failure_type") == ftype for r in support_rows) / len(support_rows)
        for ftype in ("missing_entity", "wrong_attribute", "wrong_relation", "ambiguous_reference", "unobservable_target", "external_knowledge_required")
    }
    (evaluation / "visual_support.json").write_text(json.dumps(visual_support, indent=2) + "\n", encoding="utf-8")

    trad = json.loads((evaluation / "traditional/system_summary.json").read_text())
    sampled_div = json.loads((evaluation / "diversity_sampled/system_summary.json").read_text())
    traditional = {key: trad.get(key) for key in ("n", "corpus_bleu4", "corpus_qbleu4_variant", "corpus_cider", "distinct_2", "normalized_excess_duplicate_rate")}
    traditional["sampled_distinct_2"] = sampled_div.get("distinct_2")
    traditional["sampled_normalized_excess_duplicate_rate"] = sampled_div.get("normalized_excess_duplicate_rate")
    (evaluation / "traditional.json").write_text(json.dumps(traditional, indent=2) + "\n", encoding="utf-8")

    diversity = json.loads((evaluation / "diversity/summary.json").read_text())
    (evaluation / "diversity.json").write_text(json.dumps(diversity, indent=2) + "\n", encoding="utf-8")

    summary = {"visual_centric": visual_centric, "visual_support": visual_support, "traditional": traditional, "semantic_diversity": diversity["systems"]["sft"]["mean"]}
    (evaluation / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
