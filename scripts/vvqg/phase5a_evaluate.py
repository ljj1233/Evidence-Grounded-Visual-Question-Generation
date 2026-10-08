#!/usr/bin/env python3
"""Phase 5A evaluation runner: same pipeline as SFT/RFT/GRPO (traditional,
SeeQ visual-centric, GPT-4o visual support) on development1000."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METRICS_PY = ROOT / ".venv-vllm-cu128-v011/bin/python"
REFERENCES = ROOT / "outputs/evaluation/sft_greedy/val2014_sample1000.jsonl"
DIMENSIONS = ("visual_search_difficulty", "visual_evidence_coverage", "visual_context_reasoning", "visual_spatial_reasoning")
FAILURE_TYPES = ("missing_entity", "wrong_attribute", "wrong_relation", "ambiguous_reference", "unobservable_target", "external_knowledge_required")


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []


def run(command: list[str]) -> None:
    print("$", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--seed-seeq", type=Path, default=None)
    parser.add_argument("--seed-support", type=Path, default=None)
    args = parser.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = read(args.predictions)

    seeq_input = out / "seeq_input.jsonl"
    with seeq_input.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"image_id": row["image_id"], "image": row.get("image") or row.get("image_path"), "question": row["question"]}, ensure_ascii=False) + "\n")
    seeq_scores = out / "seeq/scores.jsonl"
    if args.seed_seeq and args.seed_seeq.is_file() and not seeq_scores.is_file():
        seeq_scores.parent.mkdir(parents=True, exist_ok=True)
        seeq_scores.write_bytes(args.seed_seeq.read_bytes())
    support_cache = out / "support/cache.jsonl"
    if args.seed_support and args.seed_support.is_file() and not support_cache.is_file():
        support_cache.parent.mkdir(parents=True, exist_ok=True)
        support_cache.write_bytes(args.seed_support.read_bytes())

    seeq_cmd = [str(METRICS_PY), str(ROOT / "scripts/vvqg/evaluate_visual_centric.py"),
                "--predictions", str(seeq_input), "--output", str(seeq_scores),
                "--model", "gpt-5.6-luna", "--allow-variant", "--concurrency", str(args.concurrency),
                "--label", f"SeeQ {args.system}"]
    support_cmd = [str(METRICS_PY), str(ROOT / "scripts/vvqg/evaluate_visual_support.py"),
                   "--predictions", f"{args.system}={args.predictions}", "--out", str(out / "support"),
                   "--model", "gpt-4o", "--concurrency", str(args.concurrency), "--label", f"Support {args.system}"]
    if args.retry_failures:
        seeq_cmd.append("--retry-failures")
        support_cmd.append("--retry-failures")
    run(seeq_cmd)
    run(support_cmd)
    run([str(METRICS_PY), str(ROOT / "scripts/vvqg/evaluate.py"),
         "--predictions", str(args.predictions), "--references", str(REFERENCES),
         "--metrics", "bleu4,qbleu4_variant,cider,distinct2,dup_rate", "--out", str(out / "traditional")])

    seeq_rows = [r for r in read(seeq_scores) if r.get("status") == "ok"]
    visual_centric = {key: sum(r["scores"][key] for r in seeq_rows) / len(seeq_rows) / 5.0 for key in DIMENSIONS} | {"valid": len(seeq_rows)}
    (out / "visual_centric.json").write_text(json.dumps(visual_centric, indent=2) + "\n", encoding="utf-8")

    support_rows = [r for r in read(out / f"support/{args.system}.jsonl") if r.get("status") == "ok"]
    visual_support = {label: sum(r.get("overall_visual_support") == label for r in support_rows) / len(support_rows) for label in ("supported", "unsupported", "uncertain")}
    visual_support["valid"] = len(support_rows)
    visual_support["failure_type_rates"] = {ftype: sum(r.get("failure_type") == ftype for r in support_rows) / len(support_rows) for ftype in FAILURE_TYPES}
    (out / "visual_support.json").write_text(json.dumps(visual_support, indent=2) + "\n", encoding="utf-8")

    traditional = {key: value for key, value in json.loads((out / "traditional/system_summary.json").read_text()).items()}
    summary = {"system": args.system, "predictions": str(args.predictions),
               "traditional": traditional, "visual_centric": visual_centric, "visual_support": visual_support}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
