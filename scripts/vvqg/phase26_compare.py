#!/usr/bin/env python3
"""Shared offline ranking audit for Phase 2.6 positive signals.

The evaluator is deliberately signal-agnostic: Track A/B only provide a JSONL
file containing candidate_id (or image_id/question/generation_id) and a scalar.
No signal file is modified and no evaluator API is called.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / "outputs/reward_calibration/sft_candidates_4x.jsonl"
CANON = ROOT / "outputs/phase25_offline_reward_audit/candidates4000.jsonl"
IN = ROOT / "outputs/phase25_offline_reward_audit"
OUT = ROOT / "outputs/phase26/comparison"
DIMS = ("visual_search_difficulty", "visual_evidence_coverage",
        "visual_context_reasoning", "visual_spatial_reasoning")


def read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def key(row: dict) -> tuple[str, str, int | None]:
    return str(row["image_id"]), row["question"], int(row["generation_id"]) if row.get("generation_id") is not None else None


def tie(row: dict, seed: int = 42) -> str:
    return hashlib.sha256(f"{seed}|{row['candidate_id']}".encode()).hexdigest()


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    out = np.empty(len(values), float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        out[order[i:j]] = (i + j - 1) / 2 + 1
        i = j
    return out


def rho(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)]
    if len(pairs) < 3:
        return None
    a, b = map(np.asarray, zip(*pairs)); ra, rb = rank(a), rank(b)
    den = np.std(ra) * np.std(rb)
    return float(np.mean((ra - ra.mean()) * (rb - rb.mean())) / den) if den else None


def bootstrap_corr(xs: list[float], ys: list[float], draws: int = 5000) -> dict:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)]
    if len(pairs) < 3:
        return {"estimate": None, "ci95": None, "n": len(pairs)}
    a, b = map(np.asarray, zip(*pairs))
    return {"estimate": rho(a.tolist(), b.tolist()), "ci95": None, "n": len(a), "resamples": 0, "seed": 42}


def bootstrap_mean(values: list[float], draws: int = 5000) -> dict:
    a = np.asarray(values, float)
    if not len(a):
        return {"estimate": None, "ci95": None, "n": 0}
    rng = np.random.default_rng(42); means = a[rng.integers(0, len(a), size=(draws, len(a)))].mean(1)
    return {"estimate": float(a.mean()), "ci95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))], "n": len(a), "resamples": draws, "seed": 42}


def load_rows() -> list[dict]:
    pool, canon = read(POOL), read(CANON)
    pool_keys = {key(r) for r in pool}; canon_keys = {key(r) for r in canon}
    if len(pool) != 4000 or len({key(r) for r in pool}) != 4000:
        raise RuntimeError(f"expected 4000 frozen reward-calibration candidates, got {len(pool)}")
    if pool_keys != canon_keys:
        raise RuntimeError("reward-calibration pool differs from Phase-2.5 canonical candidate keys")
    return canon


def optional(path: Path) -> list[dict]:
    return read(path) if path.exists() else []


def merge() -> list[dict]:
    rows = load_rows()
    vic = {r["candidate_id"]: r for r in read(IN / "vic/scores.jsonl")}
    luna = {key(r): r for r in read(IN / "external_eval/seeq_luna_scores.jsonl")}
    support_path = IN / "external_eval/visual_support_scores.jsonl"
    support = {r["candidate_id"]: r for r in read(support_path)}
    flat = {r["candidate_id"]: r for r in read(IN / "violation/states.jsonl")}
    types = {r["candidate_id"]: r for r in optional(ROOT / "outputs/phase25b_vic_failure_attribution/gpt4o/question_structure.jsonl")}
    out = []
    for c in rows:
        v, s, u, f, t = vic[c["candidate_id"]], luna[key(c)], support[c["candidate_id"]], flat[c["candidate_id"]], types.get(c["candidate_id"], {})
        scores = s.get("seeq_scores") or s.get("scores")
        labels = t.get("labels", {})
        out.append({**c, "G_text": v.get("G_text"), "search": scores.get(DIMS[0]) if scores else None, "coverage": scores.get(DIMS[1]) if scores else None,
                    "context": scores.get(DIMS[2]) if scores else None, "spatial": scores.get(DIMS[3]) if scores else None,
                    "support": u.get("overall_visual_support"), "flat_state": f.get("state"),
                    "primary_type": labels.get("primary_type", "other"), "question_tokens": len(c["question"].split())})
    return out


def scalar(row: dict, name: str) -> float | None:
    aliases = {"qdep-blank": "D_blank", "qdep-mismatch": "D_mismatch", "vfc": "VFC", "uvfc": "UVFC", "g_text": "G_text"}
    mapped = aliases.get(name.casefold(), name)
    names = {name, mapped, name.lower(), name.replace("-", "_")}
    for k in names:
        if k in row and isinstance(row[k], (int, float)) and not isinstance(row[k], bool): return float(row[k])
    for parent in ("signals", "scores", "values", "metrics"):
        x = row.get(parent)
        if isinstance(x, dict):
            for k in names:
                if k in x and isinstance(x[k], (int, float)): return float(x[k])
    for k in ("value", "score"):
        if isinstance(row.get(k), (int, float)): return float(row[k])
    return None


def signal_rows(path: Path, name: str, base: list[dict]) -> dict[str, float]:
    raw = read(path)
    by_id = {r["candidate_id"]: r for r in base}; by_key = {key(r): r["candidate_id"] for r in base}
    out = {}
    for r in raw:
        cid = r.get("candidate_id")
        if cid not in by_id:
            cid = by_key.get(key(r))
        if cid is None: continue
        value = scalar(r, name)
        if value is None and name.casefold() in ("g_text", "gtext"):
            value = scalar(r, "G_text")
        if value is not None: out[cid] = value
    if not out: raise RuntimeError(f"no scalar '{name}' found in {path}")
    return out


def distinct2(texts: list[str]) -> float:
    pairs = [tuple(x.casefold().split()[i:i + 2]) for x in texts for i in range(max(0, len(x.split()) - 1))]
    return len(set(pairs)) / len(pairs) if pairs else 0.0


def selector_metrics(rows: list[dict]) -> dict:
    qs = [r["question"] for r in rows]; n = len(rows)
    return {"n": n, "search": float(np.mean([r["search"] for r in rows])), "coverage": float(np.mean([r["coverage"] for r in rows])),
            "context": float(np.mean([r["context"] for r in rows])), "spatial": float(np.mean([r["spatial"] for r in rows])),
            "supported": sum(r["support"] == "supported" for r in rows) / n, "unsupported": sum(r["support"] == "unsupported" for r in rows) / n,
            "distinct_2": distinct2(qs), "duplicate_rate": 1 - len(set(qs)) / n}


def evaluate(name: str, values: dict[str, float], base: list[dict]) -> dict:
    rows = [{**r, "signal": values.get(r["candidate_id"])} for r in base if r["candidate_id"] in values and r["search"] is not None]
    groups = defaultdict(list)
    for r in rows: groups[str(r["image_id"])].append(r)
    def top_bottom(sub, require_four=False):
        gs = defaultdict(list)
        for r in sub: gs[str(r["image_id"])].append(r)
        minimum = 4 if require_four else 2
        pairs = [(max(v, key=lambda r: (r["signal"], tie(r))), min(v, key=lambda r: (r["signal"], tuple(reversed(tie(r)))))) for v in gs.values() if len(v) >= minimum]
        return {"n_images": len(pairs), "search": bootstrap_mean([a["search"] - b["search"] for a, b in pairs]), "coverage": bootstrap_mean([a["coverage"] - b["coverage"] for a, b in pairs])}
    def concord(dim):
        good = hit = 0
        for v in groups.values():
            for a, b in itertools.combinations(v, 2):
                ds, de = a["signal"] - b["signal"], a[dim] - b[dim]
                if ds and de: good += 1; hit += int(np.sign(ds) == np.sign(de))
        return {"concordant": hit, "denominator": good, "rate": hit / good if good else None}
    complete = [v for v in groups.values() if len(v) == 4]
    top = [max(v, key=lambda r: (r["signal"], tie(r))) for v in complete]
    constrained = []
    for v in complete:
        admissible = [r for r in v if r["flat_state"] != "CONFIRMED_VIOLATION"]
        constrained.append(max(admissible or v, key=lambda r: (r["signal"], tie(r))))
    randoms = []
    for seed in range(42, 142):
        randoms.append(selector_metrics([sorted(v, key=lambda r: tie(r, seed))[0] for v in groups.values() if len(v) == 4]))
    top_metrics = selector_metrics(top)
    top1_vs_random = {}
    for metric in ("search", "coverage", "context", "spatial", "supported", "unsupported"):
        deltas = np.asarray([top_metrics[metric] - sample[metric] for sample in randoms], dtype=float)
        top1_vs_random[metric] = {"estimate": float(deltas.mean()), "ci95": [float(np.quantile(deltas, .025)), float(np.quantile(deltas, .975))], "n": len(deltas), "seed": 42}
    supported = [r for r in rows if r["support"] == "supported"]
    anti = {"token_length": bootstrap_corr([r["signal"] for r in rows], [r["question_tokens"] for r in rows]),
            "question_type": {k: {"n": sum(r["primary_type"] == k for r in rows), "mean": float(np.mean([r["signal"] for r in rows if r["primary_type"] == k]))} for k in sorted({r["primary_type"] for r in rows})},
            "support_state": {k: {"n": sum(r["support"] == k for r in rows), "mean": float(np.mean([r["signal"] for r in rows if r["support"] == k]))} for k in ("supported", "unsupported", "uncertain")}}
    return {"signal": name, "n": len(rows), "global_correlations": {d: bootstrap_corr([r["signal"] for r in rows], [r[d] for r in rows]) for d in ("search", "coverage")},
            "top_bottom": top_bottom(rows, require_four=True), "pairwise_concordance": {"search": concord("search"), "coverage": concord("coverage")},
            "top1": selector_metrics(top), "top1_vs_random": top1_vs_random, "constrained_top1": selector_metrics(constrained), "constrained_fallback_count": sum(not any(r["flat_state"] != "CONFIRMED_VIOLATION" for r in v) for v in complete),
            "random_from_4": {"draws": 100, "mean": {k: float(np.mean([x[k] for x in randoms])) for k in randoms[0]}, "p025_p975": {k: [float(np.quantile([x[k] for x in randoms], .025)), float(np.quantile([x[k] for x in randoms], .975))] for k in randoms[0]}},
            "supported_only": {"n": len(supported), "global_correlations": {d: bootstrap_corr([r["signal"] for r in supported], [r[d] for r in supported]) for d in ("search", "coverage")}, "top_bottom": top_bottom(supported)},
            "anti_confound": anti}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--signal", action="append", default=[], metavar="NAME=JSONL"); ap.add_argument("--out", type=Path, default=OUT); args = ap.parse_args()
    base = merge(); specs = {"G_text": IN / "vic/scores.jsonl"}
    for item in args.signal:
        name, path = item.split("=", 1); specs[name] = Path(path)
    # Auto-consume conventional Track-A/B score files when present.
    qdep = ROOT / "outputs/phase26/qdep/scores4000.jsonl"
    vfc = ROOT / "outputs/phase26/vfc/scores4000.jsonl"
    if qdep.exists(): specs.update({"QDep-Blank": qdep, "QDep-Mismatch": qdep})
    if vfc.exists(): specs.update({"VFC": vfc, "UVFC": vfc})
    reports = {}
    for name, path in specs.items(): reports[name] = evaluate(name, signal_rows(path, name, base), base)
    if {"QDep-Blank", "QDep-Mismatch"} <= reports.keys():
        dump(ROOT / "outputs/phase26/qdep/metrics.json", {k: reports[k] for k in ("QDep-Blank", "QDep-Mismatch")})
        dump(ROOT / "outputs/phase26/qdep/supported_only.json", {k: reports[k]["supported_only"] for k in ("QDep-Blank", "QDep-Mismatch")})
        dump(ROOT / "outputs/phase26/qdep/decision.json", {"decision": "QDEP_GO" if any(reports[k]["top1_vs_random"][d]["ci95"][0] > 0 and reports[k]["top1_vs_random"]["unsupported"]["ci95"][1] <= 0.05 for k in ("QDep-Blank", "QDep-Mismatch") for d in ("search", "coverage")) else "QDEP_NO_GO", "signals": {k: {"top1": reports[k]["top1"], "random": reports[k]["random_from_4"]} for k in ("QDep-Blank", "QDep-Mismatch")}})
    if {"VFC", "UVFC"} <= reports.keys():
        dump(ROOT / "outputs/phase26/vfc/metrics.json", {k: reports[k] for k in ("VFC", "UVFC")})
        dump(ROOT / "outputs/phase26/vfc/supported_only.json", {k: reports[k]["supported_only"] for k in ("VFC", "UVFC")})
        dump(ROOT / "outputs/phase26/vfc/decision.json", {"decision": "VFC_GO" if any(reports[k]["top1_vs_random"][d]["ci95"][0] > 0 and reports[k]["top1_vs_random"]["unsupported"]["ci95"][1] <= 0.05 for k in ("VFC", "UVFC") for d in ("search", "coverage")) else "VFC_NO_GO", "signals": {k: {"top1": reports[k]["top1"], "random": reports[k]["random_from_4"]} for k in ("VFC", "UVFC")}})
    args.out.mkdir(parents=True, exist_ok=True)
    for name, report in reports.items(): dump(args.out / f"{name}.json", report)
    summary = {k: {"n": v["n"], "top1": v["top1"], "global_correlations": v["global_correlations"]} for k, v in reports.items()}
    dump(args.out / "comparison.json", {"candidate_count": len(base), "image_count": len({r['image_id'] for r in base}), "seed": 42, "bootstrap_resamples": 5000, "signals": summary})
    dump(args.out / "signal_bakeoff.json", {"protocol": "phase26_shared_validation", "signals": reports})
    dump(args.out / "paired_bootstrap.json", {k: {"top_bottom": v["top_bottom"], "global_correlations": v["global_correlations"]} for k, v in reports.items()})
    # JSONL artifacts retain the actual selected candidate for auditability.
    for filename, constrained_flag in (("positive_selector.jsonl", False), ("constrained_selector.jsonl", True)):
        lines = []
        for name, path in specs.items():
            values = signal_rows(path, name, base); by_image = defaultdict(list)
            for r in base:
                if r["candidate_id"] in values and r["search"] is not None: by_image[str(r["image_id"])].append({**r, "signal": values[r["candidate_id"]]})
            for image_id, group in sorted(by_image.items()):
                admissible = [r for r in group if r["flat_state"] != "CONFIRMED_VIOLATION"]
                chosen = max((admissible or group) if constrained_flag else group, key=lambda r: (r["signal"], tie(r)))
                lines.append(json.dumps({"signal": name, "selector": "constrained" if constrained_flag else "positive", "image_id": image_id, "candidate_id": chosen["candidate_id"], "question": chosen["question"], "signal_value": chosen["signal"], "flat_state": chosen["flat_state"], "fallback": constrained_flag and not admissible}, ensure_ascii=False))
        (args.out / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")
    dump(args.out / "final_selector_metrics.json", {k: {"positive": v["top1"], "constrained": v["constrained_top1"], "random_from_4": v["random_from_4"], "constrained_fallback_count": v["constrained_fallback_count"]} for k, v in reports.items()})
    dump(args.out / "supported_only.json", {k: v["supported_only"] for k, v in reports.items()})
    dump(args.out / "anti_confounds.json", {k: v["anti_confound"] for k, v in reports.items()})
    decisions = {}
    for k, v in reports.items():
        delta = v["top1_vs_random"]
        gain = delta["search"]["ci95"][0] > 0 or delta["coverage"]["ci95"][0] > 0
        unsupported_ok = delta["unsupported"]["ci95"][1] <= 0.05
        decisions[k] = {"decision": "GO" if gain and unsupported_ok else "NO_GO", "positive_gain": gain, "unsupported_ok": unsupported_ok, "unsupported_delta": delta["unsupported"]}
    candidates = [k for k in decisions if k not in {"G_text"} and decisions[k]["decision"] == "GO"]
    winner = max(candidates, key=lambda k: (reports[k]["top_bottom"]["search"]["estimate"] > 0,
        reports[k]["top_bottom"]["coverage"]["estimate"] > 0,
        reports[k]["top1"]["search"] - reports[k]["random_from_4"]["mean"]["search"],
        reports[k]["top1"]["coverage"] - reports[k]["random_from_4"]["mean"]["coverage"])) if candidates else None
    final = "POSITIVE_SIGNAL_REFORMULATION_FAILED"
    if winner and winner.startswith("QDep"): final = "QDEP_ONLY_PROMISING"
    if winner and winner in {"VFC", "UVFC"}: final = "VFC_ONLY_PROMISING"
    dump(args.out / "final_decision.json", {"decision": final, "winner": winner, "decisions": decisions,
        "grpo_started": False, "final_holdout_touched": False})
    dump(args.out / "manifest.json", {"pool": str(POOL.relative_to(ROOT)), "signals": {k: str(v) for k, v in specs.items()}, "api_calls": 0, "policy_training": False, "final_holdout_touched": False})


if __name__ == "__main__": main()
