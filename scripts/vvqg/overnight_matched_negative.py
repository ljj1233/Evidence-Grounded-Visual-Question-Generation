#!/usr/bin/env python3
"""PART C — matched-negative pair construction + text-only shortcut probes.

Keeps the original chosen; replaces only the rejected with a Flat-unsupported
candidate of the SAME broad type, chosen deterministically by (same interrogative
head, smallest token-length gap, highest TF-IDF cosine, question string).

Produces CONTROL (original pairs restricted to M) and TREATMENT (matched pairs on
exactly the same M), then two grouped 5-fold text-only pairwise probes.
No training, no API, no dev-result usage.
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

HEADS = ("how many", "what color", "which color")


def head(q: str) -> str:
    q = q.strip().lower()
    for p in HEADS:
        if q.startswith(p):
            return p
    return q.split()[0] if q.split() else ""


def rd(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def folds_for(ids, seed=42, k=5):
    import numpy as np
    arr = sorted(set(ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(arr)
    return [set(f) for f in np.array_split(arr, k)]


def nb_probe(pairs, seed=42):
    import numpy as np
    ids = [p["image_id"] for p in pairs]
    folds = folds_for(ids, seed)
    per = []
    for test in folds:
        tr = [p for p in pairs if p["image_id"] not in test]
        cnt = {0: Counter(), 1: Counter()}
        tot = {0: 0, 1: 0}
        for p in tr:
            for label, text in ((1, p["chosen"]), (0, p["rejected"])):
                for w in tokens(text):
                    cnt[label][w] += 1
                    tot[label] += 1
        vocab = len(set().union(*(set(c) for c in cnt.values()))) or 1

        def score(q):
            return sum(math.log((cnt[1][w] + 1) / (tot[1] + vocab)) - math.log((cnt[0][w] + 1) / (tot[0] + vocab)) for w in tokens(q))

        te = [p for p in pairs if p["image_id"] in test]
        d = np.array([score(p["chosen"]) - score(p["rejected"]) for p in te])
        per.append(float((d > 0).mean()) if len(d) else 0.5)
    return {"mean": float(np.mean(per)), "per_fold": [round(x, 4) for x in per]}


def tfidf_probe(pairs, seed=42):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    ids = [p["image_id"] for p in pairs]
    folds = folds_for(ids, seed)
    per = []
    for test in folds:
        tr = [p for p in pairs if p["image_id"] not in test]
        te = [p for p in pairs if p["image_id"] in test]
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        x = vec.fit_transform([t for p in tr for t in (p["chosen"], p["rejected"])])
        y = [1, 0] * len(tr)
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(x, y)
        cs = clf.decision_function(vec.transform([p["chosen"] for p in te]))
        rs = clf.decision_function(vec.transform([p["rejected"] for p in te]))
        per.append(float((cs > rs).mean()) if len(te) else 0.5)
    return {"mean": float(np.mean(per)), "per_fold": [round(x, 4) for x in per]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--original-pairs", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    R = args.out_dir.resolve()
    R.mkdir(parents=True, exist_ok=True)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    pool = rd(args.pool)
    by_image = defaultdict(list)
    for row in pool:
        by_image[str(row["image_id"])].append(row)
    orig = rd(args.original_pairs)

    all_q = [str(r["question"]).strip() for r in pool]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    vectorizer.fit(all_q)
    cs_cache = {}

    def cos(texts_a, text_b):
        va = vectorizer.transform(texts_a)
        vb = vectorizer.transform([text_b])
        return cosine_similarity(va, vb).ravel()

    matched, control = [], []
    for p in orig:
        image_id = str(p["image_id"])
        chosen = str(p["chosen"]).strip()
        ctype = broad_type(chosen)
        members = by_image[image_id]
        counts = Counter(norm_question(m["question"]) for m in members)
        cands = [m for m in members if m.get("flat_state") == "CONFIRMED_VIOLATION"
                 and m.get("overall_visual_support") == "unsupported"
                 and broad_type(m["question"]) == ctype and len(str(m["question"]).strip()) >= 2
                 and counts[norm_question(m["question"])] == 1]
        if not cands:
            continue
        sims = cos([str(m["question"]).strip() for m in cands], chosen)
        ch = head(chosen)
        keyed = sorted(range(len(cands)), key=lambda j: (
            0 if head(str(cands[j]["question"])) == ch else 1,
            abs(len(tokens(str(cands[j]["question"]))) - len(tokens(chosen))),
            round(-float(sims[j]), 6),
            str(cands[j]["question"])))
        rej = cands[keyed[0]]
        matched.append({"pair_id": hashlib.sha256(f"{image_id}\x00{chosen}\x00{rej['question']}".encode()).hexdigest()[:16],
                        "image_id": image_id, "image": p["image"], "type": ctype,
                        "chosen": chosen, "rejected": str(rej["question"]).strip(),
                        "original_rejected": str(p["rejected"]).strip(),
                        "chosen_UVFC": rej.get("UVFC"), "cos_sim": round(float(sims[keyed[0]]), 4)})
        control.append({"pair_id": p.get("pair_id", ""), "image_id": image_id, "image": p["image"], "type": ctype,
                        "chosen": chosen, "rejected": str(p["rejected"]).strip()})

    matched.sort(key=lambda x: x["image_id"]); control.sort(key=lambda x: x["image_id"])
    (R / "PAIR_MANIFEST.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in matched), encoding="utf-8")
    (R / "CONTROL_PAIRS.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in control), encoding="utf-8")
    (R / "TREATMENT_PAIRS.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in matched), encoding="utf-8")

    def gap(pairs):
        return sum(abs(len(tokens(p["chosen"])) - len(tokens(p["rejected"]))) for p in pairs) / len(pairs)

    def sim(pairs):
        return sum(float(cos([p["chosen"]], p["rejected"])[0]) for p in pairs) / len(pairs)

    stats = {
        "M": len(matched), "fraction_of_original": round(len(matched) / len(orig), 4),
        "type_histogram": dict(Counter(p["type"] for p in matched)),
        "control_type_match_rate": round(sum(broad_type(p["chosen"]) == broad_type(p["rejected"]) for p in control) / len(control), 4),
        "treatment_type_match_rate": round(sum(broad_type(p["chosen"]) == broad_type(p["rejected"]) for p in matched) / len(matched), 4),
        "control_mean_abs_length_gap": round(gap(control), 4), "treatment_mean_abs_length_gap": round(gap(matched), 4),
        "control_mean_tfidf_cos": round(sim(control), 4), "treatment_mean_tfidf_cos": round(sim(matched), 4),
        "matching_rule": "mandatory same broad type; then (same interrogative head, smallest token-length gap, highest pool-TFIDF cosine, question string); deterministic; no dev info",
    }
    (R / "PAIR_STATS.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    probes = {"nb": {"control": nb_probe(control), "treatment": nb_probe(matched)},
              "tfidf_logreg": {"control": tfidf_probe(control), "treatment": tfidf_probe(matched)}}
    (R / "TEXT_SHORTCUT_DIAGNOSTIC.json").write_text(json.dumps(probes, indent=2) + "\n", encoding="utf-8")
    (R / "PROTOCOL.json").write_text(json.dumps({"task": "matched_negative_construction", "no_training": True, "no_api": True,
        "original_pairs": str(args.original_pairs), "pool": str(args.pool), "taxonomy": list(TYPES),
        "frozen_matching_rule": stats["matching_rule"], "probes": ["grouped 5-fold NB bag-of-words", "grouped 5-fold TF-IDF + logistic regression"],
        "no_dev_result_in_matching": True}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stats": stats, "probes": probes}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
