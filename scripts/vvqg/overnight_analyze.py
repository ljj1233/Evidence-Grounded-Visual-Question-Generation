#!/usr/bin/env python3
"""Assemble trajectory / controlled-run diagnostics (types, paired bootstrap, tables)."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "src"))
from vvqg.evaluation.question_types import TYPES, broad_type, distinct_n, tokens, type_counts, type_entropy  # noqa: E402
from vvqg.evidence_constrained.audit import image_level_bootstrap  # noqa: E402

SEEQ_MODEL = "gpt-5.6-luna"


def read(path: Path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def seeq_keys():
    from vvqg.evaluation.visual_centric.seeq import _audit, image_digest
    dim = _audit()["seeq_protocol"]["dimensions"][0]
    prompt_hash = hashlib.sha256((dim["prompt_system"] + dim["prompt_user"]).encode()).hexdigest()

    def key(record, ih):
        payload = {"version": "seeq-single-question-v1", "model": SEEQ_MODEL, "prompt_hash": prompt_hash,
                   "image_id": str(record["image_id"]), "image_hash": ih, "question": record["question"]}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def digest(image):
        return image_digest(ROOT / image)

    return key, digest


def paired(base_vals, tag_vals, n, seed):
    pairs = [(b, t) for b, t in zip(base_vals, tag_vals) if b is not None and t is not None]
    if not pairs:
        return {"valid_n": 0}
    deltas = [t - b for b, t in pairs]
    ci = image_level_bootstrap(deltas, n=n, seed=seed)
    return {"valid_n": len(deltas), "base": sum(b for b, _ in pairs) / len(pairs), "value": sum(t for _, t in pairs) / len(pairs),
            "delta": ci["mean"], "ci_lo": ci["lo"], "ci_hi": ci["hi"],
            "up": sum(d > 1e-12 for d in deltas), "down": sum(d < -1e-12 for d in deltas), "tied": sum(abs(d) <= 1e-12 for d in deltas)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations-dir", type=Path, required=True)
    ap.add_argument("--tags", required=True)
    ap.add_argument("--base-tag", default="base")
    ap.add_argument("--flat-dir", type=Path, default=None)
    ap.add_argument("--seeq", type=Path, default=None)
    ap.add_argument("--support-dir", type=Path, default=None)
    ap.add_argument("--trad-dir", type=Path, default=None)
    ap.add_argument("--train-log", type=Path, default=None)
    ap.add_argument("--step-map", default="", help="tag:step pairs e.g. step01000:1000,final:6633")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--bootstrap-n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    R = args.out_dir.resolve()
    R.mkdir(parents=True, exist_ok=True)
    tags = args.tags.split(",")

    gens = {tag: read(args.generations_dir / f"{tag}.jsonl") for tag in tags}
    ids = sorted({str(r["image_id"]) for r in gens[tags[0]]})

    flat = {}
    if args.flat_dir:
        flat = {tag: {str(r["image_id"]): r for r in read(args.flat_dir / f"{tag}.jsonl")} for tag in tags}
    support = {}
    if args.support_dir:
        support = {tag: {str(r["image_id"]): r for r in read(args.support_dir / f"{tag}.jsonl")} for tag in tags}
    seeq = {}
    key = digest = None
    if args.seeq:
        key, digest = seeq_keys()
        seeq = {r["cache_key"]: r for r in read(args.seeq) if r.get("status") == "ok"}

    def per_image(tag):
        out = {}
        for row in gens[tag]:
            i = str(row["image_id"])
            entry = {"question": row["question"], "type": broad_type(row["question"]), "ntok": len(tokens(row["question"]))}
            if flat:
                f = flat[tag].get(i, {})
                label = f.get("overall_visual_support") if f else None
                entry.update({"flat_supported": 1.0 if label == "supported" else 0.0 if label else None,
                              "flat_unsupported": 1.0 if label == "unsupported" else 0.0 if label else None,
                              "flat_uncertain": 1.0 if label == "uncertain" else 0.0 if label else None,
                              "flat_label": label})
            if support:
                s = support[tag].get(i, {})
                label = s.get("overall_visual_support") if s.get("status") == "ok" else None
                entry.update({"gpt_supported": 1.0 if label == "supported" else 0.0 if label else None,
                              "gpt_unsupported": 1.0 if label == "unsupported" else 0.0 if label else None,
                              "gpt_uncertain": 1.0 if label == "uncertain" else 0.0 if label else None,
                              "gpt_label": label})
            if seeq:
                score = seeq.get(key(row, digest(row["image"])))
                if score:
                    entry.update({"coverage": score["scores"]["visual_evidence_coverage"] / 5.0,
                                  "search": score["scores"]["visual_search_difficulty"] / 5.0,
                                  "context": score["scores"]["visual_context_reasoning"] / 5.0,
                                  "spatial": score["scores"]["visual_spatial_reasoning"] / 5.0})
                else:
                    entry.update({"coverage": None, "search": None, "context": None, "spatial": None})
            out[i] = entry
        return out

    per = {tag: per_image(tag) for tag in tags}
    metric_names = [k for k in ("flat_supported", "flat_unsupported", "flat_uncertain", "gpt_supported",
                                "gpt_unsupported", "gpt_uncertain", "coverage", "search", "context", "spatial")]
    boot = {tag: {m: paired([per[args.base_tag][i].get(m) for i in ids], [per[tag][i].get(m) for i in ids],
                            args.bootstrap_n, args.seed) for m in metric_names} for tag in tags}

    type_dist, rows = {}, {}
    for tag in tags:
        qs = [per[tag][i]["question"] for i in ids]
        counts = type_counts(qs)
        row = {"supported_flat": boot[tag]["flat_supported"].get("value"), "supported_gpt4o": boot[tag]["gpt_supported"].get("value"),
               "unsupported_gpt4o": boot[tag]["gpt_unsupported"].get("value"), "coverage": boot[tag]["coverage"].get("value"),
               "context": boot[tag]["context"].get("value"), "search": boot[tag]["search"].get("value"),
               "spatial": boot[tag]["spatial"].get("value"), "type_entropy": round(type_entropy(qs), 4),
               "what_rate": round(counts["what"] / len(qs), 4), "count_yesno_rate": round((counts["count"] + counts["yes/no"]) / len(qs), 4),
               "mean_length_tokens": round(sum(len(tokens(q)) for q in qs) / len(qs), 3),
               "exact_duplicate_rate": round((len(qs) - len({q.strip().lower() for q in qs})) / len(qs), 4),
               "unique_rate": round(len({q.strip().lower() for q in qs}) / len(qs), 4),
               "distinct1": round(distinct_n(qs, 1), 4), "distinct2": round(distinct_n(qs, 2), 4)}
        if args.trad_dir:
            p = args.trad_dir / tag / "system_summary.json"
            if p.is_file():
                s = json.loads(p.read_text())
                row.update({"bleu4": s["corpus_bleu4"], "cider": s["corpus_cider"], "qbleu4_variant": s["corpus_qbleu4_variant"]})
        row["question_changed_rate_vs_base"] = round(sum(per[tag][i]["question"] != per[args.base_tag][i]["question"] for i in ids) / len(ids), 4)
        rows[tag] = row
        type_dist[tag] = {"counts": counts, "rates": {k: round(v / len(qs), 4) for k, v in counts.items()}, "entropy_bits": round(type_entropy(qs), 4)}

    stepmap = {k: int(v) for k, v in (kv.split(":") for kv in args.step_map.split(",") if kv)}
    logratio = {}
    if args.train_log:
        log = {json.loads(l)["step"]: json.loads(l) for l in args.train_log.read_text().splitlines() if l.strip()}
        for tag, s in stepmap.items():
            if s in log:
                logratio[tag] = log[s]["kl_chosen_logratio"]

    metric_list = ["supported_flat", "supported_gpt4o", "unsupported_gpt4o", "coverage", "context", "search", "spatial",
                   "bleu4", "cider", "qbleu4_variant", "what_rate", "count_yesno_rate", "type_entropy", "mean_length_tokens",
                   "exact_duplicate_rate", "unique_rate", "distinct1", "distinct2", "question_changed_rate_vs_base"]
    csv = "tag,step,logratio_diag," + ",".join(metric_list) + "\n"
    order = sorted(tags, key=lambda t: stepmap.get(t, -1))
    for tag in order:
        vals = ",".join("" if rows[tag].get(m) is None else f"{rows[tag][m]:.6f}" for m in metric_list)
        csv += f"{tag},{stepmap.get(tag,'')},{'' if tag not in logratio else round(logratio[tag],4)},{vals}\n"
    (R / "TRAJECTORY_TABLE.csv").write_text(csv)
    (R / "TYPE_DISTRIBUTION.json").write_text(json.dumps(type_dist, indent=2) + "\n")
    (R / "PAIRED_BOOTSTRAP.json").write_text(json.dumps({"base_tag": args.base_tag, "bootstrap": {"n": args.bootstrap_n, "seed": args.seed},
        "results": boot}, indent=2) + "\n")
    if flat:
        (R / "FLAT_RESULTS.json").write_text(json.dumps({tag: {i: {k: per[tag][i][k] for k in per[tag][i] if k.startswith("flat_") or k == "question"} for i in ids} for tag in tags}, indent=2) + "\n")
    if support:
        (R / "GPT4O_RESULTS.json").write_text(json.dumps({tag: {i: {k: per[tag][i][k] for k in per[tag][i] if k.startswith("gpt_") or k == "question"} for i in ids} for tag in tags}, indent=2) + "\n")
    if seeq:
        (R / "SEEQ_RESULTS.json").write_text(json.dumps({tag: {"coverage": boot[tag]["coverage"], "search": boot[tag]["search"],
            "context": boot[tag]["context"], "spatial": boot[tag]["spatial"]} for tag in tags}, indent=2) + "\n")
    if args.trad_dir:
        (R / "REFERENCE_METRICS.json").write_text(json.dumps({tag: {k: rows[tag].get(k) for k in ("bleu4", "cider", "qbleu4_variant")} for tag in tags}, indent=2) + "\n")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
