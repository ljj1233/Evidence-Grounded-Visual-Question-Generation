"""Q-BLEU4-variant (canonical active metric; NOT standard Q-BLEU4).

This is a local re-implementation of the question-type-weighted BLEU of Nema &
Khapra (EMNLP 2018). The official package
(https://github.com/PrekshaNema25/Answerability-Metric) is not installed and its
tuned component weight matrix is not reproduced. Matched n-grams are weighted
1.0 when the candidate question type appears among the reference types and 0.5
otherwise; the remaining BLEU machinery (closest reference length, zero-count
smoothing (1e-9 / (total + 1)), brevity penalty, lowercase regex tokenization) is shared with the
corpus BLEU-4 implementation.

Because equivalence to the published Q-BLEU4 is not established, the canonical
name is ``qbleu4_variant`` and active reports must not label it "Q-BLEU4".
"""

from __future__ import annotations

import numpy as np
from typing import Sequence
from vvqg.evaluation.metrics.bleu import bleu_statistics, bleu_from_statistics, tokenize

TYPE_RULES = {
    "what": "what", "which": "which", "where": "where", "when": "when", "why": "why", "how": "how",
    "who": "who", "whose": "who", "whom": "who",
    "is": "be", "are": "be", "was": "be", "were": "be", "am": "be", "be": "be",
    "do": "do", "does": "do", "did": "do",
    "can": "modal", "could": "modal", "will": "modal", "would": "modal", "should": "modal",
    "shall": "modal", "may": "modal", "might": "modal", "must": "modal",
    "has": "have", "have": "have", "had": "have",
}
TYPE_MATCH_WEIGHT = 1.0
TYPE_MISMATCH_WEIGHT = 0.5
MAX_ORDER = 4


def question_type(question: str) -> str:
    words = tokenize(question)
    return TYPE_RULES.get(words[0], "other") if words else "empty"


def qbleu_statistics(candidates: Sequence[str], references: Sequence[Sequence[str]], stats: np.ndarray | None = None) -> np.ndarray:
    result = bleu_statistics(candidates, references) if stats is None else stats.copy()
    weights = np.array([TYPE_MATCH_WEIGHT if question_type(c) in {question_type(r) for r in refs}
                        else TYPE_MISMATCH_WEIGHT for c, refs in zip(candidates, references)])
    result[:, :4] *= weights[:, None]
    return result


def qbleu_from_statistics(stats: np.ndarray) -> np.ndarray:
    return bleu_from_statistics(stats, zero_numerator=1e-9)


def qbleu4_variant(candidates: Sequence[str], references: Sequence[Sequence[str]]) -> float:
    return float(qbleu_from_statistics(qbleu_statistics(candidates, references).sum(axis=0)))
