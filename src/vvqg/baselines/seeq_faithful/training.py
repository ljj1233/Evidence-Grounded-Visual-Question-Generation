"""Faithful SeeQ dual-format (QG + QA) construction and messages.

PAPER 3.4.1 defines D_QG = {x_i -> (q_i, a_i)} and D_QA = {(x_i, q_i) -> a_i}.
The controlled same-backbone adaptation follows the task contract: QG is
image -> question and QA is image + question -> answer, so both formats use the
project's canonical question-generation instruction and stay directly
comparable with M0.  Recorded as LOCAL_CONTROLLED_ADAPTATION.
"""
from __future__ import annotations

from typing import Iterable

from ..common import assert_exact_count
from ..seeq_style.training import SeeQDataset, make_collator  # noqa: F401 (re-exported)

QG_INSTRUCTION = "Generate a question about this image."
QA_INSTRUCTION = "Answer this question about the image: "


def dual_format_records(triples: Iterable[dict]) -> list[dict]:
    triples = list(triples)
    output = []
    for row in triples:
        base = {k: row[k] for k in ("image_id", "image", "question", "answer") if k in row}
        if "intent" in row:
            base["intent"] = row["intent"]
        output.extend((
            {**base, "format": "qg", "input": "image", "target": row["question"]},
            {**base, "format": "qa", "input": "image+question", "target": row["answer"]},
        ))
    assert_exact_count(output, 2 * len(triples), "SeeQ faithful dual-format records")
    return output


def faithful_messages(row: dict) -> list[dict]:
    if row["format"] == "qg":
        user = [{"type": "image"}, {"type": "text", "text": QG_INSTRUCTION}]
    elif row["format"] == "qa":
        user = [{"type": "image"}, {"type": "text", "text": QA_INSTRUCTION + row["question"]}]
    else:
        raise ValueError(f"unknown SeeQ format: {row['format']}")
    return [{"role": "user", "content": user}, {"role": "assistant", "content": row["target"]}]


__all__ = ["dual_format_records", "faithful_messages", "SeeQDataset", "make_collator", "QG_INSTRUCTION", "QA_INSTRUCTION"]
