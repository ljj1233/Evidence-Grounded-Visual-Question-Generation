#!/usr/bin/env python3
"""Task 4 — same-type pair feasibility / text-shortcut audit (CPU only).

Uses the existing formal candidate pool + Flat labels (no relabeling, no training,
no API).  Builds a diagnostic-only same-image, same-broad-type Supported>Unsupported
pair set and reruns the image-grouped 5-fold text-only NB ranking diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "src"))
from vvqg.evaluation.question_types import TYPES, broad_type, tokens  # noqa: E402
from vvqg.training.fixed_group_relative import norm_question  # noqa: E402


def read(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def nb_diagnostic(pairs: list[dict], seed: int = 42) -> dict:
    import numpy as np

    imgs = sorted({str(p["image_id"]) for p in pairs})
    rng = np.random.default_rng(seed)
    rng.shuffle(imgs)
    folds = np.array_split(imgs, 5)
    per_fold, feature_llr_total = [], Counter()

    def counts(texts, labels):
        cnt = {0: Counter(), 1: Counter()}
        tot = {0: 0, 1: 0}
        for text, label in zip(texts, labels):
            for word in tokens(text):
                cnt[label][word] += 1
                tot[label] += 1
        return cnt, tot

    all_logratio = None
    for k in range(5):
        test = set(folds[k])
        train = [p for p in pairs if str(p["image_id"]) not in test]
        texts, labels = [], []
        for p in train:
            texts += [p["chosen"], p["rejected"]]
            labels += [1, 0]
        cnt, tot = counts(texts, labels)
        vocab = len(set().union(*(set(c) for c in cnt.values()))) or 1

        def score(q):
            return sum(math.log((cnt[1][w] + 1) / (tot[1] + vocab)) - math.log((cnt[0][w] + 1) / (tot[0] + vocab)) for w in tokens(q))

        te = [p for p in pairs if str(p["image_id"]) in test]
        diffs = np.array([score(p["chosen"]) - score(p["rejected"]) for p in te])
        per_fold.append(float((diffs > 0).mean()) if len(diffs) else 0.5)
        if all_logratio is None:
            all_logratio = (cnt, tot, vocab)
        same_type = float(np.mean([broad_type(p["chosen"]) == broad_type(p["rejected"]) for p in te]))
    cnt, tot, vocab = all_logratio
    for word in set(cnt[1]) | set(cnt[0]):
        llr = math.log((cnt[1][word] + 1) / (tot[1] + vocab)) - math.log((cnt[0][word] + 1) / (tot[0] + vocab))
        feature_llr_total[word] = llr * (cnt[1][word] + cnt[0][word])
    top = sorted(feature_llr_total.items(), key=lambda kv: abs(kv[1]), reverse=True)[:15]
    return {"mean_pairwise_ranking_accuracy": float(np.mean(per_fold)), "per_fold": [round(x, 4) for x in per_fold],
            "same_type_rate": same_type, "n_pairs": len(pairs),
            "top_lexical_logratio_features": [{"token": w, "weighted_llr": round(v, 3)} for w, v in top]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--original-pairs", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    R = args.out_dir.resolve()
    R.mkdir(parents=True, exist_ok=True)

    rows = read(args.pool)
    by_image = defaultdict(list)
    for row in rows:
        by_image[str(row["image_id"])].append(row)

    pairs = []
    for image_id, members in by_image.items():
        counts = Counter(norm_question(m["question"]) for m in members)
        valid = [m for m in members if m.get("overall_visual_support") in ("supported", "unsupported")
                 and m.get("flat_state") not in (None, "FAILED") and len(str(m["question"]).strip()) >= 2
                 and counts[norm_question(m["question"])] == 1]
        supported = [m for m in valid if m.get("flat_state") == "NO_CONFIRMED_VIOLATION"]
        unsupported = [m for m in valid if m.get("flat_state") == "CONFIRMED_VIOLATION"]
        chosen = rejected = None
        for t in TYPES:  # deterministic: first type (fixed order) with both states
            s = [m for m in supported if broad_type(m["question"]) == t]
            u = [m for m in unsupported if broad_type(m["question"]) == t]
            if s and u:
                chosen, rejected = max(s, key=lambda m: float(m["UVFC"])), min(u, key=lambda m: float(m["UVFC"]))
                break
        if chosen is None or norm_question(chosen["question"]) == norm_question(rejected["question"]):
            continue
        pairs.append({"pair_id": hashlib.sha256(f"{image_id}\x00{chosen['question']}\x00{rejected['question']}".encode()).hexdigest()[:16],
                      "image_id": image_id, "image": chosen["image"], "type": broad_type(chosen["question"]),
                      "chosen": str(chosen["question"]).strip(), "rejected": str(rejected["question"]).strip(),
                      "chosen_UVFC": chosen.get("UVFC"), "rejected_UVFC": rejected.get("UVFC")})
    pairs.sort(key=lambda p: str(p["image_id"]))
    (R / "PAIR_MANIFEST.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8")

    original = read(args.original_pairs)
    orig_meta = json.loads((ROOT / "reports/offline_dpo_v1_formal_20261007T102440Z/PAIR_FILTER_STATS.json").read_text())
    pair_types = Counter(p["type"] for p in pairs)
    len_ch = sum(len(tokens(p["chosen"])) for p in pairs) / len(pairs)
    len_re = sum(len(tokens(p["rejected"])) for p in pairs) / len(pairs)
    stats = {
        "original_pairs": len(original), "original_unique_images": len({p["image_id"] for p in original}),
        "same_type_pairs": len(pairs), "same_type_unique_images": len(pairs),
        "fraction_of_original": round(len(pairs) / len(original), 4),
        "pair_type_counts": dict(pair_types), "pair_type_balance": {k: round(pair_types[k] / len(pairs), 4) for k in TYPES},
        "mean_token_length_chosen": round(len_ch, 3), "mean_token_length_rejected": round(len_re, 3),
        "mean_length_diff": round(len_ch - len_re, 3),
        "selection_rule": "per image: chosen=max UVFC supported, rejected=min UVFC unsupported, within first type in fixed order (yes/no,count,color,what,other) having both; one pair/image",
    }
    (R / "PAIR_STATS.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    diag = {
        "taxonomy": list(TYPES),
        "original_pairs": nb_diagnostic(original),
        "same_type_candidate_pairs": nb_diagnostic(pairs),
        "note": "diagnostic only; same-type dataset is NOT trained",
    }
    (R / "TEXT_SHORTCUT_DIAGNOSTIC.json").write_text(json.dumps(diag, indent=2) + "\n", encoding="utf-8")
    (R / "TOP_LEXICAL_FEATURES.json").write_text(json.dumps({
        "original_pairs": diag["original_pairs"]["top_lexical_logratio_features"],
        "same_type_candidate_pairs": diag["same_type_candidate_pairs"]["top_lexical_logratio_features"],
    }, indent=2) + "\n", encoding="utf-8")
    (R / "PROTOCOL.json").write_text(json.dumps({
        "task": "same_type_pair_feasibility_shortcut_audit", "cpu_only": True, "training": False, "api": False,
        "pool": str(args.pool), "original_pairs": str(args.original_pairs),
        "taxonomy": list(TYPES), "diagnostic": "image-grouped 5-fold naive-Bayes bag-of-words pairwise ranking",
        "does_not_modify_pairs": True}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stats": stats, "original_acc": diag["original_pairs"]["mean_pairwise_ranking_accuracy"],
                      "same_type_acc": diag["same_type_candidate_pairs"]["mean_pairwise_ranking_accuracy"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
