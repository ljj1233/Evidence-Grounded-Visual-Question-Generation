#!/usr/bin/env python3
"""Stage-wise diversity funnel and intent distribution for the faithful SeeQ run."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vvqg.baselines.seeq_faithful.diversity import (  # noqa: E402
    lexical_metrics, load_embedder, per_image_questions, semantic_diversity,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def subset(rows: list[dict], n_images: int) -> list[dict]:
    keep = sorted({str(r["image_id"]) for r in rows})[:n_images]
    keep = set(keep)
    return [r for r in rows if str(r["image_id"]) in keep]


def stage_metrics(rows: list[dict], key: str, embed, n_images: int) -> dict:
    m = lexical_metrics(rows, key)
    if embed is not None:
        m["semantic_diversity"] = semantic_diversity(per_image_questions(subset(rows, n_images), key), embed)
    return m


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/same_backbone_baselines/seeq_faithful")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--embed-images", type=int, default=1000)
    parser.add_argument("--skip-semdiv", action="store_true")
    args = parser.parse_args()
    out = args.out or (args.root / "evaluation")
    out.mkdir(parents=True, exist_ok=True)

    embed = None if args.skip_semdiv else load_embedder()
    funnel, intents = {}, {}
    for round_id in (1, 2):
        rd = args.root / f"round{round_id}"
        proposals = read_jsonl(rd / "proposals/proposals.jsonl")
        rewrites = read_jsonl(rd / "rewrites/rewrites.jsonl")
        filters = read_jsonl(rd / "filters/filters.jsonl")
        accepted = [r for r in filters if r.get("accepted") is True]
        retained = read_jsonl(rd / "retained/retained.jsonl")
        funnel[f"R{round_id}"] = {
            "proposal": stage_metrics(proposals, "question", embed, args.embed_images),
            "rewrite": stage_metrics(rewrites, "rewritten_question", embed, args.embed_images),
            "accepted": stage_metrics(accepted, "rewritten_question", embed, args.embed_images),
            "retained": stage_metrics(retained, "question", embed, args.embed_images),
        }
        prop = Counter(r["intent"] for r in filters)
        acc = Counter(r["intent"] for r in accepted)
        ret = Counter(r["intent"] for r in retained)
        intents[f"R{round_id}"] = {
            intent: {"proposed": prop[intent], "accepted": acc[intent], "retained": ret[intent],
                     "acceptance_rate": round(acc[intent] / prop[intent], 4) if prop[intent] else 0.0}
            for intent in sorted(prop)
        }
        # matched M_t sampled output if present
        for tag in (f"m{round_id}_sampled4", f"m{round_id}_sampled5"):
            path = out / f"{tag}.jsonl"
            if path.exists():
                funnel[f"R{round_id}"][tag] = stage_metrics(read_jsonl(path), "question", embed, args.embed_images)

    (out / "diversity_funnel.json").write_text(json.dumps(funnel, indent=2, sort_keys=True) + "\n")
    (out / "intent_distribution.json").write_text(json.dumps(intents, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"funnel_stages": {k: list(v) for k, v in funnel.items()}, "intents": intents}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
