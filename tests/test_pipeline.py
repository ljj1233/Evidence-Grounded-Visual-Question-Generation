"""Stage A integration: closed loop, condition isolation, answer binding and resume."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from egvqg.pipeline import RunConfigMismatch, execute_run, summarize_run
from egvqg.providers import Completion, CompletionRequest, MockProvider, Provider
from egvqg.records import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/stage_a.json"
MOCK_SCRIPT = ROOT / "tests" / "fixtures" / "stage_a" / "mock_script.json"
RECORD_FILES = (
    "generation_attempts.jsonl",
    "questions.jsonl",
    "validity.jsonl",
    "answers.jsonl",
    "responses.jsonl",
    "scores.jsonl",
    "calls.jsonl",
)


class SpyProvider:
    def __init__(self, inner: Provider):
        self.inner = inner
        self.requests: list[CompletionRequest] = []

    def complete(self, request: CompletionRequest) -> Completion:
        self.requests.append(request)
        return self.inner.complete(request)


class InterruptingProvider:
    """Simulates a process interruption after a fixed number of successful calls."""

    def __init__(self, inner: Provider, *, crash_after: int):
        self.inner = inner
        self.remaining = crash_after

    def complete(self, request: CompletionRequest) -> Completion:
        if self.remaining <= 0:
            raise RuntimeError("simulated process interruption")
        self.remaining -= 1
        return self.inner.complete(request)


def _run(tmp_path: Path, run_id: str, *, spy: SpyProvider | None = None, limit: int | None = None):
    provider = spy or SpyProvider(MockProvider(MOCK_SCRIPT))
    result = execute_run(
        config_path=CONFIG,
        project_root=ROOT,
        run_id=run_id,
        limit=limit,
        output_root=tmp_path,
        provider=provider,
    )
    return result, provider


def test_closed_loop_isolation_answer_binding_and_failure_types(tmp_path: Path) -> None:
    result, spy = _run(tmp_path, "integration")
    metrics = result.metrics
    assert metrics["synthetic"] is True
    assert metrics["n_planned"] == 18
    assert metrics["validity"]["n_valid"] == 11
    assert metrics["validity"]["n_invalid"] == 5
    assert metrics["validity"]["n_unresolved"] == 2
    assert metrics["validity"]["n_invalid_confirmed"] == 3
    assert metrics["validity"]["n_generation_failure"] == 1
    assert metrics["validity"]["n_technical_failure"] == 1
    assert metrics["bdr"]["n_pass"] == 2
    assert metrics["bdr"]["n_unresolved"] == 3
    assert metrics["no_material"]["n_correct"] == 5
    assert metrics["no_material"]["n_eligible"] == 10

    run_dir = result.run_dir
    questions = read_jsonl(run_dir / "questions.jsonl")
    answers = read_jsonl(run_dir / "answers.jsonl")
    attempts = read_jsonl(run_dir / "generation_attempts.jsonl")
    responses = read_jsonl(run_dir / "responses.jsonl")
    scores = read_jsonl(run_dir / "scores.jsonl")
    materials = read_jsonl(run_dir / "materials.jsonl")

    assert len(questions) == 18
    assert len(attempts) == 19
    assert len(answers) == 16
    assert len(responses) == 16 * 4
    assert len(scores) == 16 * 4

    question_by_material_method = {
        (row["material_id"], row["method"]): row for row in questions
    }
    assert question_by_material_method[("m_device", "strong")]["status"] == "technical_failure"
    assert question_by_material_method[("m_device", "evidence_anchored")]["status"] == "generation_failure"

    attempts_by_material_method: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in attempts:
        attempts_by_material_method.setdefault((row["material_id"], row["method"]), []).append(row)
    assert {
        row["status"] for row in attempts_by_material_method[("m_device", "strong")]
    } == {"technical_failure"}
    assert len(attempts_by_material_method[("m_device", "strong")]) == 2
    assert [
        row["status"] for row in attempts_by_material_method[("m_device", "evidence_anchored")]
    ] == ["generation_failure"]
    assert attempts_by_material_method[("m_device", "evidence_anchored")][0]["failure_reason"] == "model_refusal"

    answer_by_material_method = {
        (row["material_id"], row["method"]): row for row in answers
    }
    assert answer_by_material_method[("m_circuit", "plain")]["answer"] == "2A"
    assert answer_by_material_method[("m_circuit", "strong")]["answer"] == "6V"
    unresolved_answer = answer_by_material_method[("m_thermo", "evidence_anchored")]
    assert unresolved_answer["status"] == "unresolved"
    assert unresolved_answer["answer"] is None
    assert len({row["answer_id"] for row in answers}) == len(answers)
    for key, answer in answer_by_material_method.items():
        assert answer["question_id"] == question_by_material_method[key]["question_id"]

    # Four isolated conditions per generated question.
    answer_requests = [request for request in spy.requests if request.task == "answer_question"]
    assert len(answer_requests) == 16 * 4
    assert set(Counter(request.question for request in answer_requests).values()) == {4}

    material_text = {row["material_id"]: row["text"] for row in materials}
    question_to_material = {
        row["question"]: row["material_id"] for row in questions if row["question"]
    }
    forbidden_evidence = (
        "阈值45.75℃",
        "20、30、40、50℃对应活性22、47、61、29",
        "电流表读数2A",
        "铜的导热系数401",
        "答案应为",
        "内部证据",
    )
    for request in answer_requests:
        assert request.metadata == {}
        assert request.condition in {"TI", "T", "I", "0"}
        if request.condition == "TI":
            assert request.text is not None
            assert request.image_path is not None
        elif request.condition == "T":
            assert request.text is not None
            assert request.image_path is None
        elif request.condition == "I":
            assert request.text is None
            assert request.image_path is not None
        else:
            assert request.text is None
            assert request.image_path is None
        if request.condition in {"I", "0"}:
            assert material_text[question_to_material[request.question]] not in request.prompt
        for forbidden in forbidden_evidence:
            assert forbidden not in request.prompt

    for row in responses:
        manifest = row["input_manifest"]
        assert manifest["evidence_included"] is False
        assert manifest["answer_included"] is False
        assert manifest["material_id_included"] is False
        assert manifest["text_included"] == (row["condition"] in {"TI", "T"})
        assert manifest["image_included"] == (row["condition"] in {"TI", "I"})


def test_resume_does_not_duplicate_and_config_change_is_rejected(tmp_path: Path) -> None:
    first, _ = _run(tmp_path, "resume")
    counts_before = {
        filename: len(read_jsonl(first.run_dir / filename)) for filename in RECORD_FILES
    }

    second_spy = SpyProvider(MockProvider(MOCK_SCRIPT))
    second, _ = _run(tmp_path, "resume", spy=second_spy)
    assert second.new_calls == 0
    assert second.new_records == 0
    assert second_spy.requests == []
    assert second.metrics == first.metrics
    counts_after = {
        filename: len(read_jsonl(second.run_dir / filename)) for filename in RECORD_FILES
    }
    assert counts_after == counts_before

    summarized = summarize_run(run_id="resume", project_root=ROOT, output_root=tmp_path)
    assert summarized.metrics == first.metrics

    with pytest.raises(RunConfigMismatch):
        execute_run(
            config_path=CONFIG,
            project_root=ROOT,
            run_id="resume",
            limit=2,
            output_root=tmp_path,
            provider=MockProvider(MOCK_SCRIPT),
        )


def test_partial_run_resumes_without_duplicate_records(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="simulated process interruption"):
        execute_run(
            config_path=CONFIG,
            project_root=ROOT,
            run_id="partial",
            output_root=tmp_path,
            provider=InterruptingProvider(MockProvider(MOCK_SCRIPT), crash_after=20),
        )

    partial_questions = read_jsonl(tmp_path / "partial" / "questions.jsonl")
    assert 0 < len(partial_questions) < 18

    resumed = execute_run(
        config_path=CONFIG,
        project_root=ROOT,
        run_id="partial",
        output_root=tmp_path,
        provider=MockProvider(MOCK_SCRIPT),
    )
    assert resumed.new_calls > 0
    assert resumed.metrics["validity"]["n_valid"] == 11
    assert resumed.metrics["bdr"]["n_pass"] == 2

    assert len(read_jsonl(tmp_path / "partial" / "questions.jsonl")) == 18
    assert len(read_jsonl(tmp_path / "partial" / "generation_attempts.jsonl")) == 19
    assert len(read_jsonl(tmp_path / "partial" / "answers.jsonl")) == 16
    assert len(read_jsonl(tmp_path / "partial" / "responses.jsonl")) == 64
    assert len(read_jsonl(tmp_path / "partial" / "scores.jsonl")) == 64
    assert len(read_jsonl(tmp_path / "partial" / "calls.jsonl")) == 145

    key_fields = {
        "questions.jsonl": "question_id",
        "generation_attempts.jsonl": "attempt_id",
        "answers.jsonl": "answer_id",
        "responses.jsonl": "response_id",
        "scores.jsonl": "score_id",
        "calls.jsonl": "call_id",
    }
    for filename, key in key_fields.items():
        rows = read_jsonl(tmp_path / "partial" / filename)
        assert len({row[key] for row in rows}) == len(rows)
