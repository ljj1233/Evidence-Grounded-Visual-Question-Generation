#!/usr/bin/env python3
"""Build B3 train-only category/object targets from VQAv2 train annotations.

The detector cache is the only non-SFT input. No target fields are emitted to
evaluation manifests or inference calls.
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
ANNOTATIONS = ROOT / "data/vqav2/raw/annotations/v2_mscoco_train2014_annotations.json"
QUESTIONS = ROOT / "data/vqav2/raw/questions/v2_OpenEnded_mscoco_train2014_questions.json"


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def normalized_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def answer_category(answer: str, answer_to_category: dict[str, int], other_id: int) -> tuple[int, bool]:
    key = normalized_text(answer)
    category = answer_to_category.get(key)
    return (other_id, False) if category is None else (category, True)


def object_alignment(candidates: list[str], question: str, answer: str) -> list[int]:
    target = tokens(question) | tokens(answer)
    return [int(label != "<empty>" and bool(tokens(label) & target)) for label in candidates]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=ROOT / "src/vvqg/data/sft/train.jsonl")
    parser.add_argument("--detector-cache", type=Path, required=True, help="JSONL keyed by image_id with candidate labels")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/same_backbone_baselines/b3_implicit/cache/train_targets.jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--embedding-model", type=Path, default=Path(local_model_path('Qwen3-Embedding-4B')).expanduser())
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    categories = json.loads(CATEGORY_FILE.read_text(encoding="utf-8"))
    category_to_id = {name: index for index, name in enumerate(categories)}
    answer_to_category = {normalized_text(answer): category_to_id[name] for name, answers in categories.items() for answer in answers}
    annotations = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))["annotations"]
    question_text = {str(row["question_id"]): row["question"] for row in json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]}
    by_key = defaultdict(list)
    for ann in annotations:
        question = question_text[str(ann["question_id"])]
        by_key[(str(ann["image_id"]), " ".join(sorted(tokens(question))))].append(ann)
    detector = {}
    with args.detector_cache.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                detector[str(row["image_id"])] = row["candidates"]
    from vvqg.baselines.implicit_guided.semantic_alignment import make_embedder, semantic_object_targets_batch
    embed = make_embedder(args.embedding_model, args.device)
    rows = []
    usage = defaultdict(int)
    unmapped = 0
    category_histogram = defaultdict(int)
    pending_alignment = []
    for record in read_jsonl(args.records):
        image_id = Path(record["image"]).stem.rsplit("_", 1)[-1].lstrip("0") or "0"
        if image_id not in detector:
            continue
        key = (image_id, " ".join(sorted(tokens(record["question"]))))
        candidates = detector[image_id]
        matches = by_key[key]
        if usage[key] >= len(matches):
            raise KeyError(f"no VQAv2 annotation match for image/question: {key}")
        ann = matches[usage[key]]
        usage[key] += 1
        answer = str(ann["multiple_choice_answer"]).lower().strip()
        category, mapped = answer_category(answer, answer_to_category, category_to_id["other"])
        unmapped += int(not mapped)
        category_histogram[str(category)] += 1
        rows.append({"image_id": image_id, "image": record["image"], "question": record["question"], "category_id": category, "category_mapped": mapped, "unmapped_answer": None if mapped else answer, "answer": answer, "candidate_labels": candidates})
        pending_alignment.append((candidates, record["question"], answer))
        if args.limit and len(rows) >= args.limit:
            break
    targets = semantic_object_targets_batch(pending_alignment, embed)
    for row, target in zip(rows, targets):
        row["object_target"] = target
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows)
    manifest = {"seed": SEED, "records": len(rows), "mapped_count": len(rows) - unmapped, "unmapped_count": unmapped, "category_histogram": dict(sorted(category_histogram.items())), "categories": categories, "category_source": str(CATEGORY_FILE), "annotation_source": str(ANNOTATIONS), "question_source": str(QUESTIONS), "detector_cache": str(args.detector_cache), "target_sha256": stable_hash(rows), "extra_train_only_supervision": "answer_category_and_question_answer_concept_alignment", "detector_substitution": "torchvision_COCO_fasterrcnn_resnet50_fpn_v2", "semantic_alignment": "Qwen3-Embedding-4B frozen cosine alignment replacing released GloVe"}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
