"""Anchor questions from the frozen Base policy and the Flat evidence gate."""

from __future__ import annotations

from typing import Any, Sequence

from vvqg.rewards.support.structured_flat import support_summary

VIOLATION = "CONFIRMED_VIOLATION"
SUPPORT_FLOOR_MIN = 0.5
MIN_CHARS = 2
MAX_TOKENS = 32


def anchor_eligibility(question: str, support: dict[str, Any], token_count: int) -> tuple[bool, str]:
    """Evidence-loss gate from the task spec; returns (eligible, reason)."""
    if not str(question).strip():
        return False, "empty_question"
    if len(str(question).strip()) < MIN_CHARS:
        return False, "question_too_short"
    if support.get("flat_state") == "FAILED":
        return False, "parser_failed"
    if token_count < 1 or token_count > MAX_TOKENS:
        return False, "question_length_invalid"
    if support.get("flat_state") == VIOLATION:
        return False, "confirmed_violation"
    if float(support.get("support_floor", -1.0)) < SUPPORT_FLOOR_MIN:
        return False, "support_floor_below_threshold"
    return True, "eligible"


def build_anchor_records(
    base_rows: Sequence[dict[str, Any]],
    questions: Sequence[str],
    token_counts: Sequence[int],
    parsed: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge Base questions, Flat parses and counterfactual paths; never drop rows."""
    records = []
    for row, question, token_count, parse in zip(base_rows, questions, token_counts, parsed):
        support = support_summary(parse)
        eligible, reason = anchor_eligibility(question, support, token_count)
        records.append({
            "image_id": str(row["image_id"]),
            "image": row["original_path"],
            "image_path": row["original_path"],
            "shuffled_image_path": row["shuffled_path"],
            "blurred_image_path": row["blurred_path"],
            "question": str(question).strip(),
            "question_token_count": int(token_count),
            "parser_status": parse.get("status"),
            "requirement_labels": support.get("requirement_labels"),
            "n_requirements": support.get("n_requirements"),
            "overall_visual_support": parse.get("overall_visual_support"),
            "flat_state": support.get("flat_state"),
            "support_floor": support.get("support_floor"),
            "support_mean": support.get("support_mean"),
            "eligible_for_evidence_loss": bool(eligible),
            "skip_reason": reason,
            "skipped_for_evidence_loss": not eligible,
        })
    return records
