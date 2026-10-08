#!/usr/bin/env python3
"""Evaluate greedy predictions with the frozen independent support rubric."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vvqg.evaluation.visual_support.api import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", action="append", required=True, help="system_id=predictions.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--concurrency", type=int, default=16, choices=range(1, 17))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--label", default="Visual support")
    args = parser.parse_args()
    env = dict(line.split("=", 1) for line in args.env_file.read_text().splitlines() if "=" in line and not line.lstrip().startswith("#"))
    env = {k.strip(): v.strip().strip('"').strip("'") for k, v in env.items()}
    endpoint = env.get("AUDIT_JUDGE_BASE_URL") or env["LLM_JUDGE_BASE_URL"]
    inputs = {k: Path(v) for k, v in (item.split("=", 1) for item in args.predictions)}
    report = asyncio.run(evaluate(inputs, args.out, endpoint, env["LLM_JUDGE_API_KEY"], args.model, args.concurrency, args.limit, args.retry_failures, label=args.label))
    print(json.dumps({"runtime": report["runtime"], "systems": report["systems"]}, indent=2))


if __name__ == "__main__":
    main()
