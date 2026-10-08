"""Independent reference implementations used by the evaluation audit.

These are deliberately written separately from the active metric modules so the
audit can compare "declared definition" against an independent computation.
They are used by ``tests/test_evaluation_metrics.py`` and by the audit reports.

Nothing here calls an API or a GPU.
"""

from __future__ import annotations

from typing import Optional, Sequence

# Hand-computed expected values for the K=4 golden fixture (Section 10 of the
# audit spec). These are hard-coded; they are NOT produced by the module under
# test.
K4_FIXTURE_CASES = [
    {"source": 1, "applicable_negatives": 0},
    {"source": 1, "applicable_negatives": 1},
    {"source": 1, "applicable_negatives": 4},
    {"source": 0, "applicable_negatives": 2},
    {"source": 0, "applicable_negatives": 0},
]
K4_FIXTURE_EXPECTED = {
    "source_applicable_rate": 3 / 5,
    "any_negative_applicable_rate": 3 / 5,
    "mean_negative_fraction": 7 / 20,
    "conditional_any_negative_rate": 2 / 3,
    "conditional_mean_negative_fraction": 5 / 12,
    "source_unique_rate": 1 / 5,
}


def k4_rates(cases: Sequence[dict]) -> dict[str, Optional[float]]:
    """Independent computation of the 1+4 aggregate rates for golden fixtures."""
    n = len(cases)
    k = 4
    if n == 0:
        return {key: None for key in K4_FIXTURE_EXPECTED}
    source = [int(c["source"]) for c in cases]
    negatives = [int(c["applicable_negatives"]) for c in cases]
    source_applicable = sum(source)
    return {
        "source_applicable_rate": sum(source) / n,
        "any_negative_applicable_rate": sum(1 for m in negatives if m > 0) / n,
        "mean_negative_fraction": sum(negatives) / (k * n),
        "conditional_any_negative_rate": (
            sum(1 for i in range(n) if source[i] and negatives[i] > 0) / source_applicable if source_applicable else None
        ),
        "conditional_mean_negative_fraction": (
            sum(source[i] * negatives[i] for i in range(n)) / (k * source_applicable) if source_applicable else None
        ),
        "source_unique_rate": sum(1 for i in range(n) if source[i] and negatives[i] == 0) / n,
    }


def coco_caption_bleu4(candidates: Sequence[str], references: Sequence[Sequence[str]]) -> Optional[float]:
    """Corpus BLEU-4 from the original coco-caption scorer (pycocoevalcap wrapper)."""
    try:
        from pycocoevalcap.bleu.bleu import Bleu
    except ModuleNotFoundError:
        return None
    gts = {i: list(refs) for i, refs in enumerate(references)}
    res = {i: [candidate] for i, candidate in enumerate(candidates)}
    score, _ = Bleu(4).compute_score(gts, res, verbose=0)
    return float(score[3])
