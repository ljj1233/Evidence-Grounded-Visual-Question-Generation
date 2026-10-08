#!/usr/bin/env python3
"""GQA evaluation runner: same evaluator implementation as VQAv2, GQA references.

Traditional (BLEU-4/CIDEr/Q-BLEU4-v/Distinct-2/Dup) uses the GQA reference file;
SeeQ visual-centric (gpt-5.6-luna variant) and GPT-4o visual support use the same
frozen APIs as the VQAv2 published baselines.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METRICS_PY = ROOT / ".venv-vllm-cu128-v011/bin/python"
DIMENSIONS = ("visual_search_difficulty", "visual_evidence_coverage", "visual_context_reasoning", "visual_spatial_reasoning")
FAILURE_TYPES = ("missing_entity", "wrong_attribute", "wrong_relation", "ambiguous_reference", "unobservable_target", "external_knowledge_required")


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []


def run(command: list[str]) -> None:
    print("$", " ".join(str(p) for p in command), flush=True)
    subprocess.run(command, check=True)


def gqa_references_to_jsonl(refs_path: Path, out: Path) -> int:
    refs = json.loads(refs_path.read_text())
    with out.open("w", encoding="utf-8") as handle:
        for image_id, payload in refs.items():
            for question in payload["references"]:
                handle.write(json.dumps({"image_id": str(image_id), "question": question}, ensure_ascii=False) + "\n")
    return len(refs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--refs", type=Path, required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--retry-failures", action="store_true")
    args = parser.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = read(args.predictions)

    refs_jsonl = out / "references.jsonl"
    gqa_references_to_jsonl(args.refs, refs_jsonl)

    seeq_input = out / "seeq_input.jsonl"
    with seeq_input.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"image_id": str(row["image_id"]), "image": row.get("image") or row.get("image_path"), "question": row["question"]}, ensure_ascii=False) + "\n")
    seeq_cmd = [str(METRICS_PY), str(ROOT / "scripts/vvqg/evaluate_visual_centric.py"),
                "--predictions", str(seeq_input), "--output", str(out / "seeq/scores.jsonl"),
                "--model", "gpt-5.6-luna", "--allow-variant", "--concurrency", str(args.concurrency),
                "--label", f"SeeQ {args.system}"]
    support_cmd = [str(METRICS_PY), str(ROOT / "scripts/vvqg/evaluate_visual_support.py"),
                   "--predictions", f"{args.system}={args.predictions}", "--out", str(out / "support"),
                   "--model", "gpt-4o", "--concurrency", str(min(args.concurrency, 16)), "--label", f"Support {args.system}"]
    if args.retry_failures:
        seeq_cmd.append("--retry-failures")
        support_cmd.append("--retry-failures")
    run(seeq_cmd)
    run(support_cmd)
    run([str(METRICS_PY), str(ROOT / "scripts/vvqg/evaluate.py"),
         "--predictions", str(args.predictions), "--references", str(refs_jsonl),
         "--metrics", "bleu4,qbleu4_variant,cider,distinct2,dup_rate", "--out", str(out / "traditional")])

    seeq_rows = [r for r in read(out / "seeq/scores.jsonl") if r.get("status") == "ok"]
    visual_centric = {key: sum(r["scores"][key] for r in seeq_rows) / len(seeq_rows) / 5.0 for key in DIMENSIONS} | {"valid": len(seeq_rows)}
    (out / "visual_centric.json").write_text(json.dumps(visual_centric, indent=2) + "\n")

    support_rows = [r for r in read(out / f"support/{args.system}.jsonl") if r.get("status") == "ok"]
    visual_support = {label: sum(r.get("overall_visual_support") == label for r in support_rows) / len(support_rows) for label in ("supported", "unsupported", "uncertain")}
    visual_support["valid"] = len(support_rows)
    visual_support["failure_type_rates"] = {ft: sum(r.get("failure_type") == ft for r in support_rows) / len(support_rows) for ft in FAILURE_TYPES}
    (out / "visual_support.json").write_text(json.dumps(visual_support, indent=2) + "\n")

    traditional = json.loads((out / "traditional/system_summary.json").read_text())
    summary = {"system": args.system, "predictions": str(args.predictions), "refs": str(args.refs),
               "traditional": traditional, "visual_centric": visual_centric, "visual_support": visual_support,
               "seeq_valid": len(seeq_rows), "support_valid": len(support_rows), "predictions_n": len(rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
