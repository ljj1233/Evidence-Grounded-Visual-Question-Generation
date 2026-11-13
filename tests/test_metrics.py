"""Metric denominators, tri-state BDR, unresolved propagation and no-material coverage."""

from __future__ import annotations

from typing import Any

from egvqg.metrics import compute_metrics


def _response(question_id: str, condition: str) -> dict[str, Any]:
    return {
        "response_id": f"r_{question_id}_{condition}",
        "question_id": question_id,
        "condition": condition,
        "status": "answered",
    }


def _score(question_id: str, condition: str, correct: bool | None, *, resolved: bool = True) -> dict[str, Any]:
    return {
        "score_id": f"s_{question_id}_{condition}",
        "response_id": f"r_{question_id}_{condition}",
        "question_id": question_id,
        "condition": condition,
        "status": "scored" if resolved else "unresolved",
        "correct": correct if resolved else None,
    }


def _case(
    question_id: str,
    *,
    question_status: str,
    validity: str | None,
    answer: str | None,
    scores: dict[str, tuple[bool | None, bool]],
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    question = {"question_id": question_id, "status": question_status}
    validity_row = {"question_id": question_id, "status": validity} if validity else None
    answer_row = {"question_id": question_id, "status": answer} if answer else None
    responses: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for condition, (correct, resolved) in scores.items():
        responses.append(_response(question_id, condition))
        score_rows.append(_score(question_id, condition, correct, resolved=resolved))
    return question, validity_row, answer_row, responses, score_rows


def test_fixed_denominators_unresolved_and_no_material_coverage() -> None:
    plan = [{"question_id": f"q{index}", "method": "plain"} for index in range(6)]
    cases = [
        # BDR pass; no-material answer is wrong.
        _case(
            "q0",
            question_status="ok",
            validity="valid",
            answer="verified",
            scores={"TI": (True, True), "T": (False, True), "I": (False, True), "0": (False, True)},
        ),
        # Validity unresolved, but T is correct: B=0 is already determined.
        _case(
            "q1",
            question_status="ok",
            validity="unresolved",
            answer="verified",
            scores={"TI": (True, True), "T": (True, True), "I": (False, True), "0": (True, True)},
        ),
        # Confirmed invalid question remains in N and B=0.
        _case(
            "q2",
            question_status="ok",
            validity="invalid",
            answer="verified",
            scores={"TI": (True, True), "T": (False, True), "I": (False, True), "0": (True, True)},
        ),
        # No valid output is a generation failure, not a deleted sample.
        _case("q3", question_status="generation_failure", validity=None, answer=None, scores={}),
        # Valid and verified, but the I score is unresolved: B stays unresolved.
        _case(
            "q4",
            question_status="ok",
            validity="valid",
            answer="verified",
            scores={"TI": (True, True), "T": (False, True), "I": (None, False), "0": (True, True)},
        ),
        # B passes, but no-material score is unresolved and excluded from that denominator.
        _case(
            "q5",
            question_status="ok",
            validity="valid",
            answer="verified",
            scores={"TI": (True, True), "T": (False, True), "I": (False, True), "0": (None, False)},
        ),
    ]
    questions, validity, answers, responses, scores = [], [], [], [], []
    for question, validity_row, answer_row, response_rows, score_rows in cases:
        questions.append(question)
        if validity_row:
            validity.append(validity_row)
        if answer_row:
            answers.append(answer_row)
        responses.extend(response_rows)
        scores.extend(score_rows)

    metrics = compute_metrics(
        plan,
        questions,
        validity,
        answers,
        responses,
        scores,
        run_id="unit",
        protocol_version="test",
        evidence_class="synthetic_unit_test",
    )

    assert metrics["n_planned"] == 6
    assert metrics["validity"]["n_valid"] == 3
    assert metrics["validity"]["n_invalid"] == 2  # q2 confirmed invalid + q3 generation failure
    assert metrics["validity"]["n_unresolved"] == 1
    assert metrics["validity"]["n_invalid_confirmed"] == 1
    assert metrics["validity"]["n_generation_failure"] == 1
    assert metrics["validity"]["valid_rate"] == 3 / 6
    assert metrics["validity"]["valid_rate_upper_with_unresolved"] == 4 / 6

    assert metrics["bdr"]["n_pass"] == 2  # q0 and q5
    assert metrics["bdr"]["n_fail"] == 3  # q1 known T correct, q2 invalid, q3 failure
    assert metrics["bdr"]["n_unresolved"] == 1  # q4 unresolved I
    assert metrics["bdr"]["bdr"] == 2 / 6
    assert metrics["bdr"]["bdr_upper_with_unresolved"] == 3 / 6

    assert metrics["no_material"]["n_valid_verified"] == 3
    assert metrics["no_material"]["n_eligible"] == 2  # q0, q4; q5 C0 is unresolved
    assert metrics["no_material"]["n_correct"] == 1
    assert metrics["no_material"]["n_unresolved"] == 1
    assert metrics["no_material"]["accuracy"] == 1 / 2
    assert metrics["no_material"]["denominator_share_of_planned"] == 2 / 6

    assert metrics["conditions"]["T"]["n_resolved"] == 5
    assert metrics["conditions"]["I"]["n_resolved"] == 4
    assert metrics["conditions"]["0"]["n_resolved"] == 4
