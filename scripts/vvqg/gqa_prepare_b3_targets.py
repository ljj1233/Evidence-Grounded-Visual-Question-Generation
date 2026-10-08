#!/usr/bin/env python3
"""GQA-native B3 (Guiding VQG) category/object targets.

Same construction as the VQAv2 builder (IQ answer categories + question/answer
concept alignment + frozen detector candidates), but the answer source is the
GQA train_balanced questions file. Answers are used only as *training-time*
guidance targets; they never enter an inference manifest.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vvqg.paths import local_model_path
from vvqg.baselines.common import SEED, read_jsonl, stable_hash, write_jsonl

CATEGORY_FILE = ROOT / "src/vvqg/data/baselines/iq_answer_categories.json"
GQA_QUESTIONS = ROOT / "data/raw/gqa/questions1.2/train_balanced_questions.json"


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def normalized_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--detector-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--embedding-model", type=Path, default=Path(local_model_path('Qwen3-Embedding-4B')).expanduser())
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    categories = json.loads(CATEGORY_FILE.read_text(encoding="utf-8"))
    category_to_id = {name: index for index, name in enumerate(categories)}
    answer_to_category = {normalized_text(a): category_to_id[name] for name, answers in categories.items() for a in answers}
    questions = json.loads(GQA_QUESTIONS.read_text())
    by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    for rec in questions.values():
        by_key[(str(rec["imageId"]), " ".join(sorted(tokens(rec["question"]))))].append(str(rec["answer"]).lower().strip())

    detector = {}
    with args.detector_cache.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                detector[str(row["image_id"])] = row["candidates"]

    from vvqg.baselines.implicit_guided.semantic_alignment import make_embedder, semantic_object_targets_batch
    embed = make_embedder(args.embedding_model, args.device)

    rows, usage = [], defaultdict(int)
    unmapped = 0
    histogram = defaultdict(int)
    pending = []
    for record in read_jsonl(args.records):
        image_id = str(record["image_id"])
        if image_id not in detector:
            continue
        key = (image_id, " ".join(sorted(tokens(record["question"]))))
        answers = by_key.get(key)
        if not answers:
            continue
        answer = answers[usage[key] % len(answers)]
        usage[key] += 1
        category = answer_to_category.get(normalized_text(answer))
        mapped = category is not None
        category = category_to_id["other"] if category is None else category
        unmapped += int(not mapped)
        histogram[str(category)] += 1
        candidates = detector[image_id]
        rows.append({"image_id": image_id, "image": record["image"], "question": record["question"],
                     "category_id": category, "category_mapped": mapped, "unmapped_answer": None if mapped else answer,
                     "answer": answer, "candidate_labels": candidates})
        pending.append((candidates, record["question"], answer))
        if args.limit and len(rows) >= args.limit:
            break

    for row, target in zip(rows, semantic_object_targets_batch(pending, embed)):
        row["object_target"] = target

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows)
    manifest = {"seed": SEED, "records": len(rows), "mapped_count": len(rows) - unmapped, "unmapped_count": unmapped,
                "category_histogram": dict(sorted(histogram.items())), "categories": categories,
                "category_source": str(CATEGORY_FILE), "answer_source": str(GQA_QUESTIONS),
                "detector_cache": str(args.detector_cache), "target_sha256": stable_hash(rows),
                "extra_train_only_supervision": "gqa_answer_category_and_question_answer_concept_alignment",
                "detector_substitution": "torchvision_COCO_fasterrcnn_resnet50_fpn_v2",
                "semantic_alignment": "Qwen3-Embedding-4B frozen cosine alignment"}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: manifest[k] for k in ("records", "mapped_count", "unmapped_count")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
