import json
from pathlib import Path

import pytest

from vvqg.baselines.seeq_style import SeeQConfig, SeeQPipeline
from vvqg.baselines.seeq_style.pipeline import _accepted, _json_output, _rewrite_output, accepted_triples


class MockBackend:
    def __init__(self):
        self.calls = []

    def propose(self, images, tasks):
        self.calls.append(("propose", len(images)))
        return [{"Question": f"What is task {task}?"} for task in tasks]

    def answer(self, images, questions):
        self.calls.append(("answer", len(images)))
        return ["visible" for _ in questions]

    def rewrite(self, images, questions, answers):
        self.calls.append(("rewrite", len(images)))
        return [{"Rewritten question": question + " with evidence?"} for question in questions]

    def filter(self, images, questions, answers):
        self.calls.append(("filter", len(images)))
        return [{"accepted": True} for _ in questions]


def test_seeq_stage_resume_only_processes_missing_suffix(tmp_path: Path):
    images = [{"image_id": str(i), "image": f"{i}.jpg"} for i in range(8)]
    config = SeeQConfig(retained_per_round=4, turns=1, candidates_per_image=1)
    backend = MockBackend()
    pipeline = SeeQPipeline(backend, config)
    first = pipeline.run(images, tmp_path, limit=8)
    assert first["images_visited"] == 8
    for name in ("stage_b_answers.jsonl", "stage_c_rewrites.jsonl", "stage_d_filters.jsonl"):
        path = tmp_path / name
        path.write_text("\n".join(path.read_text().splitlines()[:3]) + "\n", encoding="utf-8")
    backend.calls.clear()
    resumed = pipeline.run(images, tmp_path, limit=8)
    assert resumed["accepted"] == 8
    assert backend.calls == [("propose", 5), ("answer", 5), ("rewrite", 5), ("filter", 5)]


def test_seeq_schedule_is_seeded_and_unique():
    images = [{"image_id": "a", "image": "a.jpg"}]
    config = SeeQConfig(seed=42, turns=2, candidates_per_image=2)
    left = SeeQPipeline.schedule(images, config, round_id=1)
    right = SeeQPipeline.schedule(images, config, round_id=1)
    assert left == right
    assert [row["proposal_id"] for row in left] == ["r1-000000-0-0", "r1-000000-0-1", "r1-000000-1-0", "r1-000000-1-1"]


def test_seeq_filter_rejects_malformed_judges():
    assert _accepted({"text": "accepted"})[0] is False
    assert _accepted({"hard_constraints": {}})[0] is False
    assert _accepted({"hard_constraints": {"Factuality": {"count": 0}, "Ambiguity": {"count": 0}, "Validity": {"answerable": 1}}})[0] is True
    assert _accepted({"hard_constraints": {"Factuality": 0, "Ambiguity": 0, "Validity": 1}})[0] is True


def test_seeq_accepts_python_dict_output_from_local_qwen():
    assert _json_output("{'Question': 'What is visible?'}")["Question"] == "What is visible?"


def test_rewrite_normalizes_bare_question_to_rewritten_key():
    assert _rewrite_output("What is the man doing?") == {"Rewritten question": "What is the man doing?"}
    assert _rewrite_output({"text": "Is it red?"}) == {"Rewritten question": "Is it red?"}
    assert _rewrite_output("{'Rewritten question': 'What color is it?'}") == {"Rewritten question": "What color is it?"}
    assert _rewrite_output("{'Question': 'What is visible?'}") == {"Question": "What is visible?"}


class BareRewriteBackend(MockBackend):
    def rewrite(self, images, questions, answers):
        return [{"text": question + " under what condition?"} for question in questions]


def test_seeq_stage_d_uses_bare_rewrite_text(tmp_path: Path):
    images = [{"image_id": "0", "image": "0.jpg"}]
    config = SeeQConfig(retained_per_round=1, turns=1, candidates_per_image=1)
    SeeQPipeline(BareRewriteBackend(), config).run(images, tmp_path, limit=1)
    row = json.loads((tmp_path / "stage_e_retained.jsonl").read_text().splitlines()[0])
    assert row["question"].startswith("What is task ") and row["question"].endswith("under what condition?")


def _accepted_rows(per_image: int, images: int) -> list[dict]:
    return [
        {"image_id": str(i), "image": f"{i}.jpg", "question": f"q{i}-{j}", "answer": "a", "accepted": True}
        for i in range(images) for j in range(per_image)
    ]


def test_accepted_triples_round_robins_over_images():
    rows = _accepted_rows(per_image=2, images=3)
    out = accepted_triples(rows, SeeQConfig(retained_per_round=3))
    assert [r["image_id"] for r in out] == ["0", "1", "2"]
    out = accepted_triples(rows, SeeQConfig(retained_per_round=6))
    assert [r["image_id"] for r in out] == ["0", "1", "2", "0", "1", "2"]


def test_accepted_triples_requires_exact_count():
    rows = _accepted_rows(per_image=1, images=1)
    with pytest.raises(AssertionError):
        accepted_triples(rows, SeeQConfig(retained_per_round=2))
