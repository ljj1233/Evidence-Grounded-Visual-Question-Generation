"""Current VQG loader/schema contracts.

Salvaged from the mixed `tests/archive/test_vqg_contracts.py` (deleted in the
legacy cleanup). Only assertions that exercise the retained loader
(`vqg.data` / `vqg.datasets`) are kept; the Stage-1/Stage-2 imports that made
the original file uncollectable are dropped.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vqg.data import (
    append_jsonl,
    assert_no_image_overlap,
    build_image_level_subsets,
    filter_records,
    read_jsonl,
    validate_record,
)
from vqg.datasets import load_vqav2


def _record(sample_id: str, image_id: str, question: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "dataset": "vqav2",
        "split": "train",
        "image_id": image_id,
        "image_path": f"/images/{image_id}.jpg",
        "question_id": sample_id,
        "question": question,
        "answers": ["yes"],
        "answer_type": "yes/no",
        "question_type": "yes_no",
        "metadata": {"source_split": "train"},
    }


def test_schema_preserves_required_vqg_fields() -> None:
    record = _record("vqav2:1", "1", "Is this valid?")
    validate_record(record)

    with pytest.raises(ValueError, match="image_id"):
        validate_record({key: value for key, value in record.items() if key != "image_id"})


def test_custom_subsets_are_disjoint_at_image_level() -> None:
    records = [
        _record("vqav2:1", "image-a", "What is visible?"),
        _record("vqav2:2", "image-a", "Is it outside?"),
        _record("vqav2:3", "image-b", "Who is here?"),
        _record("vqav2:4", "image-c", "What color is it?"),
    ]

    subsets = build_image_level_subsets(records, {"debug": 1, "development": 1}, seed=7)

    debug_images = {row["image_id"] for row in subsets["debug"]}
    development_images = {row["image_id"] for row in subsets["development"]}
    assert debug_images.isdisjoint(development_images)
    assert len(subsets["debug"]) in {1, 2}


def test_no_image_overlap_between_custom_splits() -> None:
    with pytest.raises(ValueError, match="image-level leakage"):
        assert_no_image_overlap({
            "development": [_record("vqav2:1", "shared-image", "What is visible?")],
            "test": [_record("vqav2:2", "shared-image", "Who is visible?")],
        })


def test_filtering_is_conservative_and_auditable(tmp_path: Path) -> None:
    valid_image = tmp_path / "valid.jpg"
    from PIL import Image

    Image.new("RGB", (2, 2)).save(valid_image)
    valid = _record("vqav2:1", "1", "Is this valid?")
    valid["image_path"] = str(valid_image)
    duplicate = _record("vqav2:2", "1", "Is this valid?")
    duplicate["image_path"] = str(valid_image)
    missing = _record("vqav2:3", "2", "What is missing?")
    missing["image_path"] = str(tmp_path / "missing.jpg")

    kept, filtered = filter_records([valid, duplicate, missing])

    assert [row["sample_id"] for row in kept] == ["vqav2:1"]
    assert filtered == [
        {"sample_id": "vqav2:2", "dataset": "vqav2", "original_split": "train", "reason": "exact_duplicate_image_question"},
        {"sample_id": "vqav2:3", "dataset": "vqav2", "original_split": "train", "reason": "missing_or_invalid_image"},
    ]


def test_empty_question_has_specific_filter_reason(tmp_path: Path) -> None:
    image = tmp_path / "valid.jpg"
    from PIL import Image

    Image.new("RGB", (2, 2)).save(image)
    empty = _record("vqav2:empty", "1", "")
    empty["image_path"] = str(image)

    kept, filtered = filter_records([empty])

    assert kept == []
    assert filtered[0]["reason"] == "empty_question"


def test_vqav2_parser_preserves_source_ids_and_answers(tmp_path: Path) -> None:
    questions = tmp_path / "questions.json"
    annotations = tmp_path / "annotations.json"
    questions.write_text(json.dumps({"questions": [{"question_id": 9, "image_id": 42, "question": "What is here?"}]}), encoding="utf-8")
    annotations.write_text(json.dumps({"annotations": [{"question_id": 9, "answer_type": "other", "answers": [{"answer": "car"}]}]}), encoding="utf-8")

    record = next(load_vqav2(questions, annotations, tmp_path, "val"))

    assert record["sample_id"] == "vqav2:9"
    assert record["image_id"] == "42"
    assert record["answers"] == ["car"]


def test_append_jsonl_preserves_completed_rows_for_resume(tmp_path: Path) -> None:
    partial = tmp_path / "candidates.partial.jsonl"
    append_jsonl(partial, [{"image_id": "one"}])
    append_jsonl(partial, [{"image_id": "two"}])

    assert read_jsonl(partial) == [{"image_id": "one"}, {"image_id": "two"}]
