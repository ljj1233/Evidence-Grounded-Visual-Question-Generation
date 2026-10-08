"""Corpus BLEU-4 using lowercase regex tokens and frozen zero-count smoothing.

Counts are summed before scoring; closest reference lengths break ties shorter.
A zero clipped count becomes 1 / (total + 1), preserving the v1 local variant.
"""
from __future__ import annotations
import re
from collections import Counter
from typing import Sequence
import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9']+")
MAX_ORDER = 4


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu_statistics(candidates: Sequence[str], references: Sequence[Sequence[str]]) -> np.ndarray:
    """Per-record sufficient statistics: clipped[4], totals[4], cand_len, ref_len."""
    stats = np.zeros((len(candidates), 10), dtype=np.float64)
    for i, (candidate, refs) in enumerate(zip(candidates, references)):
        tokens = tokenize(candidate)
        ref_tokens = [tokenize(ref) for ref in refs]
        stats[i, 8] = len(tokens)
        if ref_tokens:
            stats[i, 9] = min((len(r) for r in ref_tokens), key=lambda n: (abs(n - len(tokens)), n))
        for order in range(1, 5):
            counts = _ngrams(tokens, order)
            maximum: Counter = Counter()
            for ref in ref_tokens:
                maximum |= _ngrams(ref, order)
            stats[i, order - 1] = sum((counts & maximum).values())
            stats[i, order + 3] = max(len(tokens) - order + 1, 0)
    return stats


def bleu_from_statistics(stats: np.ndarray, zero_numerator: float = 1.0) -> np.ndarray:
    """Score corpus totals, vectorized over leading dimensions (not records)."""
    stats = np.asarray(stats, dtype=np.float64)
    clipped, totals = stats[..., :4], stats[..., 4:8]
    zero = clipped <= 0
    precision = np.where(zero, zero_numerator, clipped) / np.maximum(totals + zero, 1)
    brevity = np.exp(np.minimum(0, 1 - stats[..., 9] / np.maximum(stats[..., 8], 1)))
    value = brevity * np.exp(np.log(precision).mean(axis=-1))
    return np.where((totals > 0).all(axis=-1), value, 0.0)


def corpus_bleu4(candidates: Sequence[str], references: Sequence[Sequence[str]]) -> float:
    return float(bleu_from_statistics(bleu_statistics(candidates, references).sum(axis=0)))
