#!/usr/bin/env python3
"""Paired image-level bootstrap for a set of systems against one or more baselines,
plus broad question-type stratified GPT support.  Reads existing SeeQ/support caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "src"))
from vvqg.evaluation.question_types import TYPES, broad_type  # noqa: E402
from vvqg.evidence_constrained.audit import image_level_bootstrap  # noqa: E402

SEEQ_MODEL = "gpt-5.6-luna"


def rd(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations-dir", type=Path, required=True)
    ap.add_argument("--tags", required=True)
    ap.add_argument("--baselines", default="base")
    ap.add_argument("--seeq", type=Path, required=True)
    ap.add_argument("--support-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bootstrap-n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    tags = args.tags.split(",")
    baselines = args.baselines.split(",")

    from vvqg.evaluation.visual_centric.seeq import _audit, image_digest
    dim = _audit()["seeq_protocol"]["dimensions"][0]
    ph = hashlib.sha256((dim["prompt_system"] + dim["prompt_user"]).encode()).hexdigest()

    def key(rec, ih):
        return hashlib.sha256(json.dumps({"version": "seeq-single-question-v1", "model": SEEQ_MODEL, "prompt_hash": ph,
            "image_id": str(rec["image_id"]), "image_hash": ih, "question": rec["question"]}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    seeq = {r["cache_key"]: r for r in rd(args.seeq) if r.get("status") == "ok"}
    per, qtype = {}, {}
    for tag in set(tags) | set(baselines):
        preds = rd(args.generations_dir / f"{tag}.jsonl")
        sup = {str(r["image_id"]): r for r in rd(args.support_dir / f"{tag}.jsonl")}
        d = {}
        for row in preds:
            i = str(row["image_id"])
            s = seeq.get(key(row, image_digest(ROOT / row["image"])))
            lab = sup.get(i, {}).get("overall_visual_support") if sup.get(i, {}).get("status") == "ok" else None
            d[i] = {"gpt_supported": (1.0 if lab == "supported" else 0.0) if lab else None,
                    "gpt_unsupported": (1.0 if lab == "unsupported" else 0.0) if lab else None,
                    "gpt_uncertain": (1.0 if lab == "uncertain" else 0.0) if lab else None,
                    "label": lab,
                    "coverage": (s["scores"]["visual_evidence_coverage"] / 5.0) if s else None,
                    "search": (s["scores"]["visual_search_difficulty"] / 5.0) if s else None,
                    "context": (s["scores"]["visual_context_reasoning"] / 5.0) if s else None,
                    "spatial": (s["scores"]["visual_spatial_reasoning"] / 5.0) if s else None}
            qtype[i] = broad_type(row["question"])
        per[tag] = d
    ids = sorted(per[tags[0]], key=int)
    metrics = ("gpt_supported", "gpt_unsupported", "gpt_uncertain", "coverage", "search", "context", "spatial")

    def paired(base, tag, m):
        pairs = [(per[base][i][m], per[tag][i][m]) for i in ids if per[base][i][m] is not None and per[tag][i][m] is not None]
        deltas = [b - a for a, b in pairs]
        ci = image_level_bootstrap(deltas, n=args.bootstrap_n, seed=args.seed)
        return {"valid_n": len(deltas), "base": sum(a for a, _ in pairs) / len(pairs), "value": sum(b for _, b in pairs) / len(pairs),
                "delta": ci["mean"], "ci_lo": ci["lo"], "ci_hi": ci["hi"]}

    result = {"vs": {}}
    for base in baselines:
        for tag in tags:
            if tag == base:
                continue
            result["vs"][f"{tag}_vs_{base}"] = {m: paired(base, tag, m) for m in metrics}

    strat = {}
    for tag in tags:
        by = defaultdict(lambda: [0, 0])
        for i in ids:
            lab = per[tag][i]["label"]
            if lab is None:
                continue
            by[qtype[i]][1] += 1
            by[qtype[i]][0] += int(lab == "supported")
        strat[tag] = {t: {"n": v[1], "supported": v[0], "rate": (v[0] / v[1] if v[1] else None)} for t, v in by.items()}
    result["type_stratified_gpt_support"] = strat
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
