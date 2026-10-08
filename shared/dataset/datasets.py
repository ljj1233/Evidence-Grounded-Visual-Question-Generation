"""Source-specific loaders that retain source annotations in the common VQG schema."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .data import canonical_record


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_vqav2(questions_path: Path, annotations_path: Path, image_dir: Path, source_split: str) -> Iterator[dict[str, Any]]:
    questions = _load_json(questions_path).get("questions", [])
    annotations = _load_json(annotations_path).get("annotations", [])
    annotation_by_id = {str(item.get("question_id")): item for item in annotations if isinstance(item, dict)}
    for item in sorted((entry for entry in questions if isinstance(entry, dict)), key=lambda entry: str(entry.get("question_id", ""))):
        question_id = str(item.get("question_id", ""))
        annotation = annotation_by_id.get(question_id, {})
        image_id = int(item.get("image_id", 0))
        answers = []
        if annotation.get("multiple_choice_answer"):
            answers.append(str(annotation["multiple_choice_answer"]))
        answers.extend(str(answer["answer"]) for answer in annotation.get("answers", []) if isinstance(answer, dict) and str(answer.get("answer", "")).strip())
        answers = list(dict.fromkeys(answers))
        yield canonical_record(
            dataset="vqav2", split=source_split, image_id=str(image_id),
            image_path=image_dir / f"COCO_{source_split}2014_{image_id:012d}.jpg", question_id=question_id,
            question=str(item.get("question", "")), answers=answers,
            answer_type=str(annotation.get("answer_type", "unknown")),
            metadata={
                "source_split": source_split,
                "source_question_type": annotation.get("question_type"),
                "multiple_choice_answer": annotation.get("multiple_choice_answer"),
            },
        )


def _gqa_depth(program: Any) -> int | None:
    return len(program) if isinstance(program, list) else None


def load_gqa(questions_path: Path, image_dir: Path, source_split: str) -> Iterator[dict[str, Any]]:
    questions = _load_json(questions_path)
    if not isinstance(questions, dict):
        raise ValueError(f"GQA questions must be a mapping: {questions_path}")
    for question_id in sorted(questions):
        item = questions[question_id]
        if not isinstance(item, dict):
            continue
        image_id = str(item.get("imageId", ""))
        semantic = item.get("semantic")
        answer = str(item.get("answer", "")).strip()
        yield canonical_record(
            dataset="gqa", split=source_split, image_id=image_id, image_path=image_dir / f"{image_id}.jpg",
            question_id=str(question_id), question=str(item.get("question", "")), answers=[answer] if answer else [],
            answer_type=str(item.get("types", {}).get("detailed", "unknown")) if isinstance(item.get("types"), dict) else "unknown",
            metadata={
                "source_split": source_split, "semantic": semantic, "structural": item.get("types", {}).get("structural") if isinstance(item.get("types"), dict) else None,
                "detailed": item.get("types", {}).get("detailed") if isinstance(item.get("types"), dict) else None,
                "reasoning_depth": _gqa_depth(semantic), "annotations": item.get("annotations", {}),
            },
        )


def _visual7w_images(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("images", "data"):
            if isinstance(payload.get(key), list):
                return [item for item in payload[key] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise ValueError("Visual7W annotation must contain an images/data list")


def load_visual7w(annotation_path: Path, image_dir: Path, source_split: str) -> Iterator[dict[str, Any]]:
    """Load the official telling annotation while retaining pointing metadata."""
    for image_item in _visual7w_images(_load_json(annotation_path)):
        image_id = str(image_item.get("image_id", image_item.get("imageId", "")))
        qa_pairs = image_item.get("qa_pairs", image_item.get("qaPairs", []))
        if not isinstance(qa_pairs, list):
            continue
        for index, pair in enumerate(qa_pairs):
            if not isinstance(pair, dict):
                continue
            question_id = str(pair.get("qa_id", pair.get("qaId", pair.get("question_id", f"{image_id}:{index}"))))
            answer = str(pair.get("answer", "")).strip()
            yield canonical_record(
                dataset="visual7w", split=source_split, image_id=image_id,
                image_path=image_dir / str(image_item.get("filename", f"{image_id}.jpg")), question_id=question_id,
                question=str(pair.get("question", "")), answers=[answer] if answer else [], answer_type="unknown",
                metadata={"source_split": source_split, "image_metadata": image_item, "qa_metadata": pair},
            )
