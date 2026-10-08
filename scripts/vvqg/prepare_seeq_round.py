#!/usr/bin/env python3
"""Freeze one SeeQ-style round after proposer/rewrite/filter generation.

The input must contain only model-generated proposal/answer/rewrite/filter
fields. This command intentionally has no VQAv2 annotation or question reader.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vvqg.baselines.common import BACKBONE, SEED, stable_hash, write_jsonl
from vvqg.baselines.seeq_style import SeeQConfig, accepted_triples, dual_format_records, check_stable_reference, SEEQ_COMMIT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--m0-hash", required=True)
    parser.add_argument("--current-m0-hash", required=False)
    parser.add_argument("--retained", type=int, default=10_000)
    parser.add_argument("--round", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    check_stable_reference(args.m0_hash, args.current_m0_hash or args.m0_hash)
    rows = []
    with args.generation.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("source") not in {"seeq_proposer", "seeq_self_evolution"}:
                raise AssertionError("generation input is not a SeeQ model artifact")
            if any(key in row for key in ("human_question", "human_answer", "vqa_answer", "annotation")):
                raise AssertionError("human VQA supervision entered SeeQ generation")
            rows.append(row)
    triples = accepted_triples(rows, SeeQConfig(retained_per_round=args.retained))
    formatted = dual_format_records(triples)
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "retained_10000.jsonl", triples)
    write_jsonl(args.output / "formatted_20000.jsonl", formatted)
    stats = {
        "schema": "seeq-style-round-v1",
        "seed": SEED,
        "backbone": BACKBONE,
        "round": args.round,
        "m0_hash": args.m0_hash,
        "official_seeq_commit": SEEQ_COMMIT,
        "retained": len(triples),
        "formatted": len(formatted),
        "generation_sha256": stable_hash(rows),
        "retained_sha256": stable_hash(triples),
        "rejected": sum(row.get("accepted") is not True for row in rows),
        "proposals": len(rows),
    }
    (args.output / "runtime.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(stats, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
