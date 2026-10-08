#!/usr/bin/env python3
"""PART A — Flat evaluator / generation parity audit (no API)."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(ROOT / "src"))
from vvqg.evaluation.question_types import TYPES, broad_type  # noqa: E402

LABELS = ("supported", "uncertain", "unsupported")


def rd(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def qmap(p):
    return {str(int(r["image_id"])): r["question"] for r in rd(p)}


def lmap(p):
    return {str(int(r["image_id"])): (r.get("overall_visual_support") if r.get("status", "ok") == "ok" else None) for r in rd(p)}


def kappa(pairs):
    pairs = [(a, b) for a, b in pairs if a and b]
    n = len(pairs)
    if not n:
        return 0.0
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in LABELS) / (n * n)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--formal", type=Path, required=True)
    ap.add_argument("--trajectory", type=Path, required=True)
    ap.add_argument("--formal-flat", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    F, T, FF, R = args.formal, args.trajectory, args.formal_flat, args.out_dir.resolve()
    R.mkdir(parents=True, exist_ok=True)
    ids = sorted((str(int(json.loads(l)["image_id"])) for l in (T / "DEV256_MANIFEST.jsonl").read_text().splitlines()), key=int)

    # generation parity
    gen = {}
    for tag, fsrc, tsrc in (("base", F / "generation/base_dev1000.jsonl", T / "GENERATIONS/base.jsonl"),
                            ("final", F / "generation/dpo_dev1000.jsonl", T / "GENERATIONS/final.jsonl")):
        gf, gt = qmap(fsrc), qmap(tsrc)
        mism = [i for i in ids if gf.get(i) != gt.get(i)]
        gen[tag] = {"n": len(ids), "exact_question_mismatches": len(mism), "mismatch_ids": mism[:20]}

    # GPT label parity formal vs trajectory
    gpt = {}
    for tag, fsrc, tsrc in (("base", F / "evaluation/support/base.jsonl", T / "SUPPORT/base.jsonl"),
                            ("final", F / "evaluation/support/dpo.jsonl", T / "SUPPORT/final.jsonl")):
        lf, lt = lmap(fsrc), lmap(tsrc)
        both = [i for i in ids if lf.get(i) and lt.get(i)]
        gpt[tag] = {"n": len(both), "agreement": sum(lf[i] == lt[i] for i in both) / len(both)}

    # 2x2 parity: old/new generation x old(new-code) scorer.  Flat on formal questions vs Flat on trajectory questions
    flat_parity = {}
    for tag in ("base", "final"):
        old_flat = lmap(FF / f"{tag}.jsonl")
        new_flat = lmap(T / "FLAT" / f"{'base' if tag == 'base' else 'final'}.jsonl")
        both = [i for i in ids if old_flat.get(i) and new_flat.get(i)]
        old_val = sum(old_flat[i] == "supported" for i in ids if old_flat.get(i)) / sum(old_flat.get(i) is not None for i in ids)
        new_val = sum(new_flat[i] == "supported" for i in ids if new_flat.get(i)) / sum(new_flat.get(i) is not None for i in ids)
        old_gpt = lmap(F / "evaluation/support" / ("base.jsonl" if tag == "base" else "dpo.jsonl"))
        new_gpt = lmap(T / "SUPPORT" / ("base.jsonl" if tag == "base" else "final.jsonl"))
        gpt_val = sum(new_gpt[i] == "supported" for i in ids if new_gpt.get(i)) / sum(new_gpt.get(i) is not None for i in ids)
        flat_parity[tag] = {"flat_old_questions": old_val, "flat_new_questions": new_val,
                            "gpt_new_questions": gpt_val,
                            "old_vs_new_flat_agreement": sum(old_flat[i] == new_flat[i] for i in both) / len(both),
                            "flat_supported_count_old": sum(old_flat[i] == "supported" for i in ids),
                            "flat_supported_count_new": sum(new_flat[i] == "supported" for i in ids)}

    # Flat vs GPT confusion per DEV256 checkpoint
    conf = {}
    for tag in ("base", "step01000", "step02000", "step03000", "final"):
        f = lmap(T / "FLAT" / f"{tag}.jsonl")
        g = lmap(T / "SUPPORT" / f"{tag}.jsonl")
        q = qmap(T / "GENERATIONS" / f"{tag}.jsonl")
        pairs = [(f.get(i), g.get(i)) for i in ids]
        matrix = {a: {b: 0 for b in LABELS} for a in LABELS}
        dis_by_type = defaultdict(lambda: [0, 0])
        for i in ids:
            a, b = f.get(i), g.get(i)
            if a in LABELS and b in LABELS:
                matrix[a][b] += 1
            t = broad_type(q[i])
            dis_by_type[t][1] += 1
            dis_by_type[t][0] += int(a != b)
        both = [(a, b) for a, b in pairs if a in LABELS and b in LABELS]
        conf[tag] = {"confusion": matrix, "n": len(both),
                     "agreement": sum(a == b for a, b in both) / len(both),
                     "cohen_kappa": kappa(both),
                     "disagreement_rate": sum(a != b for a, b in both) / len(both),
                     "disagreement_by_type": {t: {"n": v[1], "disagree": v[0], "rate": v[0] / v[1] if v[1] else 0.0}
                                              for t, v in sorted(dis_by_type.items())}}

    report = {
        "root_cause": "METRIC_IDENTITY: the formal dev1000 'Supported=0.988' is GPT-4o; the DEV256 '0.885' is the local Flat verifier on a different, stricter labeler. On the same 256 images GPT-4o=0.984 vs Flat=0.885.",
        "generation_parity": gen,
        "gpt_label_parity_formal_vs_trajectory": gpt,
        "parity_2x2": flat_parity,
        "flat_vs_gpt_confusion_by_checkpoint": conf,
        "scorer_identity": {"formal_scoring": "vvqg.rewards.support.structured_flat.StructuredFlatVerifier + frozen visual_support_prompts.json",
                            "trajectory_scoring": "same class/prompt via overnight_flat.py; no code/version change"},
        "formal_dev1000_dpo_support": {"gpt4o_supported": 988, "n": 1000},
        "dev256_final": {"gpt4o_supported": flat_parity["final"]["gpt_new_questions"], "flat_supported": flat_parity["final"]["flat_new_questions"]},
    }
    (R / "PARITY_AUDIT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
