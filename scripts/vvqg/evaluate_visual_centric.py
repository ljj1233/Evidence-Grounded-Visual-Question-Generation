#!/usr/bin/env python3
"""Run the audited SeeQ single-question rubric with strict failure handling."""
from __future__ import annotations
import argparse, asyncio, json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vvqg.evaluation.visual_centric.seeq import evaluate

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="gpt-5.4")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--allow-variant", action="store_true")
    p.add_argument("--label", default="SeeQ evaluation")
    p.add_argument("--retry-failures", action="store_true")
    args = p.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = asyncio.run(evaluate(rows, args.output, model=args.model, concurrency=args.concurrency, allow_variant=args.allow_variant, label=args.label, retry_failures=args.retry_failures))
    print(json.dumps(result, indent=2))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
