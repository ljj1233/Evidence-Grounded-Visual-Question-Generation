"""CIDEr via the official ``pycocoevalcap`` implementation (canonical active metric).

``Cider().compute_score(gts, res)`` computes the document-frequency (IDF)
statistics over the whole corpus passed in one call and returns one score per
sample. The system-level value is the mean of those per-sample scores.

Policy note: IDF is corpus-scoped. The historical bake-off ran CIDEr once per
system, so each system's IDF was computed over that system's own 1000 samples.
Any future comparison must fix one reference corpus protocol across systems.
Do NOT shard the corpus and average shard scores -- the code path never does
this (it is a single call), and multiprocessing is not used here.
"""

from __future__ import annotations

from typing import Optional, Sequence


def cider_scores(candidates: Sequence[str], references: Sequence[Sequence[str]]) -> Optional[list[float]]:
    try:
        from pycocoevalcap.cider.cider import Cider
    except ModuleNotFoundError:
        return None
    gts = {i: list(refs) for i, refs in enumerate(references)}
    res = {i: [candidate] for i, candidate in enumerate(candidates)}
    _, scores = Cider().compute_score(gts, res)
    return [float(score) for score in scores]


def cider_score(candidates: Sequence[str], references: Sequence[Sequence[str]]) -> Optional[float]:
    scores = cider_scores(candidates, references)
    return sum(scores) / len(scores) if scores else None
