#!/usr/bin/env python3
"""Task 2 — K=4 within-image diversity diagnostics (no API)."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(ROOT / "src"))
from vvqg.evaluation.question_types import TYPES, broad_type, distinct_n, tokens, type_counts, type_entropy  # noqa: E402
from vvqg.evidence_constrained.audit import image_level_bootstrap  # noqa: E402
from vvqg.training.fixed_group_relative import norm_question  # noqa: E402


def read(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", type=Path, required=True)
    ap.add_argument("--tags", required=True)
    ap.add_argument("--base-tag", default="base")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--bootstrap-n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    R = args.out_dir.resolve()
    R.mkdir(parents=True, exist_ok=True)
    tags = args.tags.split(",")

    per = {}
    rows_out = []
    for tag in tags:
        rows = read(args.gen_dir / f"{tag}.jsonl")
        rows_out += rows
        by = defaultdict(list)
        for r in rows:
            by[str(r["image_id"])].append(r["question"])
        per[tag] = by
    (R / "GENERATIONS.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows_out), encoding="utf-8")

    ids = sorted(per[tags[0]])
    csv = "tag,image_id,n,normalized_unique_rate,unique_type_rate,exact_within_dup_rate,all_same_type\n"
    per_image = {tag: {} for tag in tags}
    for tag in tags:
        for i in ids:
            qs = per[tag][i]
            norm_unique = len({norm_question(q) for q in qs}) / len(qs)
            types = {broad_type(q) for q in qs}
            exact_dup = (len(qs) - len({q.strip().lower() for q in qs})) / len(qs)
            per_image[tag][i] = {"norm_unique": norm_unique, "type_unique": len(types) / len(qs),
                                 "exact_dup": exact_dup, "all_same_type": 1.0 if len(types) == 1 else 0.0}
            csv += f"{tag},{i},{len(qs)},{norm_unique:.4f},{len(types)/len(qs):.4f},{exact_dup:.4f},{1 if len(types)==1 else 0}\n"
    (R / "WITHIN_IMAGE_DIVERSITY.csv").write_text(csv)

    def paired(metric):
        b = [per_image[args.base_tag][i][metric] for i in ids]
        out = {}
        for tag in tags:
            t = [per_image[tag][i][metric] for i in ids]
            ci = image_level_bootstrap([y - x for x, y in zip(b, t)], n=args.bootstrap_n, seed=args.seed)
            out[tag] = {"base": sum(b) / len(b), "value": sum(t) / len(t), "delta": ci["mean"], "ci_lo": ci["lo"], "ci_hi": ci["hi"]}
        return out

    type_div = {}
    for tag in tags:
        rows = read(args.gen_dir / f"{tag}.jsonl")
        qs = [r["question"] for r in rows]
        counts = type_counts(qs)
        per_img_all_same = sum(per_image[tag][i]["all_same_type"] for i in ids) / len(ids)
        type_div[tag] = {"overall_type_counts": counts, "overall_type_rates": {k: round(v / len(qs), 4) for k, v in counts.items()},
                         "type_entropy_bits": round(type_entropy(qs), 4), "all_four_same_type_rate": round(per_img_all_same, 4),
                         "distinct1": round(distinct_n(qs, 1), 4), "distinct2": round(distinct_n(qs, 2), 4),
                         "cross_image_exact_duplicate_rate": round((len(qs) - len({q.strip().lower() for q in qs})) / len(qs), 4)}
    result = {"base_tag": args.base_tag, "n_images": len(ids), "k": 4,
              "within_image": {"normalized_unique_rate": paired("norm_unique"), "unique_type_rate": paired("type_unique"),
                               "exact_within_duplicate_rate": paired("exact_dup"), "all_same_type_rate": paired("all_same_type")},
              "type_diversity": type_div,
              "note": "cross-image exact duplicate is reported separately from within-image diversity"}
    (R / "TYPE_DIVERSITY.json").write_text(json.dumps(result, indent=2) + "\n")

    lines = ["# K=4 within-image diversity", "", f"n_images={len(ids)} k=4 (base={args.base_tag})", "",
             "| tag | norm_unique/4 | unique_types/4 | exact_dup/4 | all_same_type | distinct1 | distinct2 | what_rate | avg_type_entropy |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for tag in tags:
        w = result["within_image"]
        lines.append(f"| {tag} | {w['normalized_unique_rate'][tag]['value']:.3f} | {w['unique_type_rate'][tag]['value']:.3f} | "
                     f"{w['exact_within_duplicate_rate'][tag]['value']:.3f} | {w['all_same_type_rate'][tag]['value']:.3f} | "
                     f"{type_div[tag]['distinct1']:.3f} | {type_div[tag]['distinct2']:.3f} | "
                     f"{type_div[tag]['overall_type_rates']['what']:.3f} | {type_div[tag]['type_entropy_bits']:.3f} |")
    (R / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"within_image": result["within_image"], "type_diversity": type_div}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
