#!/usr/bin/env python3
"""Phase 4B evaluation runner for one variant (development1000, greedy + sampled4x).

Drives the canonical evaluators with the correct environments and writes
normalized metric JSON under ``<run>/evaluation/``.  SeeQ and GPT-4o support are
external judges; BLEU/CIDEr/diversity are local.
"""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--retry", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    evaluation = run_dir / "evaluation"
    greedy = run_dir / "greedy1000.jsonl"
    sampled = run_dir / "sampled4x1000.jsonl"

    # 1) SeeQ visual-centric dims (Luna variant judge, documented substitution)
    run([
        str(METRICS_PY), str(ROOT / "scripts/vvqg/evaluate_visual_centric.py"),
        "--predictions", str(greedy), "--output", str(evaluation / "seeq/scores.jsonl"),
        "--model", "gpt-5.6-luna", "--allow-variant", "--concurrency", str(args.concurrency),
    ])

    # 2) Frozen independent GPT-4o visual support
    support_cmd = [
        str(METRICS_PY), str(ROOT / "scripts/vvqg/evaluate_visual_support.py"),
        "--predictions", f"{args.variant}={greedy}", "--out", str(evaluation / "support"),
        "--model", "gpt-4o", "--concurrency", str(args.concurrency),
    ]
    if args.retry:
        support_cmd.append("--retry-failures")
    run(support_cmd)

    # 3) Traditional metrics on greedy, surface diversity on sampled4x
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

    # 4) Semantic diversity (Qwen3-Embedding-4B, generation environment)
    if ZERO_SHOT_SAMPLED.is_file():
        run([
            str(GEN_PY), str(ROOT / "scripts/vvqg/evaluate_seeq_diversity.py"),
            "--model", local_model_path('Qwen3-Embedding-4B'),
            "--zero-shot", str(ZERO_SHOT_SAMPLED), "--sft", str(sampled),
            "--output", str(evaluation / "diversity"), "--batch-size", "32",
        ])

    summary = summarize(evaluation)
    (evaluation / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def summarize(evaluation: Path) -> dict:
    seeq_path = evaluation / "seeq/scores.jsonl"
    support_paths = sorted((evaluation / "support").glob("*.jsonl")) if (evaluation / "support").is_dir() else []
    traditional_path = evaluation / "traditional/system_summary.json"
    sampled_path = evaluation / "diversity_sampled/system_summary.json"
    diversity_path = evaluation / "diversity/summary.json"

    seeq = {}
    if seeq_path.is_file():
        rows = [json.loads(line) for line in seeq_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        valid = [row for row in rows if row.get("status") == "ok"]
        seeq = {key: sum(row["scores"][key] for row in valid) / len(valid) / 5.0 for key in DIMENSIONS} if valid else {}
        seeq["valid"] = len(valid)

    support = {}
    support_rows = []
    for path in support_paths:
        support_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    valid_support = [row for row in support_rows if row.get("status") == "ok"]
    if valid_support:
        support = {
            "supported": sum(row.get("overall_visual_support") == "supported" for row in valid_support) / len(valid_support),
            "unsupported": sum(row.get("overall_visual_support") == "unsupported" for row in valid_support) / len(valid_support),
            "uncertain": sum(row.get("overall_visual_support") == "uncertain" for row in valid_support) / len(valid_support),
            "valid": len(valid_support),
        }

    traditional = json.loads(traditional_path.read_text()) if traditional_path.is_file() else {}
    sampled = json.loads(sampled_path.read_text()) if sampled_path.is_file() else {}
    semantic = None
    if diversity_path.is_file():
        semantic = json.loads(diversity_path.read_text()).get("systems", {}).get("sft", {}).get("mean")
    return {
        "seeq": seeq,
        "support": support,
        "traditional": {key: traditional.get(key) for key in ("corpus_bleu4", "corpus_qbleu4_variant", "corpus_cider", "distinct_2", "normalized_excess_duplicate_rate")},
        "sampled_diversity": {key: sampled.get(key) for key in ("distinct_2", "normalized_excess_duplicate_rate")},
        "semantic_diversity": semantic,
    }


if __name__ == "__main__":
    raise SystemExit(main())
