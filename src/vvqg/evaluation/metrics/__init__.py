"""Canonical active evaluation metrics (one implementation per metric).

Active: BLEU-4 (corpus), CIDEr (official pycocoevalcap), Distinct-2,
normalized excess duplicate rate, Q-BLEU4-variant.
Diagnostic: raw-question CLIPScore adaptation, CLIP retrieval diagnostics
(see ``audit`` and the archived 2.2B/2.2C protocols).

Excluded/deprecated metrics (BERTScore, PAC-S, VQAScore, absolute GPT
visual-dependence) are NOT implemented here.
"""

from vvqg.evaluation.metrics.bleu import corpus_bleu4, tokenize as bleu_tokenize
from vvqg.evaluation.metrics.cider import cider_score, cider_scores
from vvqg.evaluation.metrics.clip import ClipScorer, clipscore_from_embeddings
from vvqg.evaluation.metrics.diversity import (
    distinct_2,
    duplicate_rate,
    most_frequent,
    normalized_excess_duplicate_rate,
    repeated_group_membership_rate,
)
from vvqg.evaluation.metrics.qbleu import qbleu4_variant, question_type

__all__ = [
    "corpus_bleu4",
    "bleu_tokenize",
    "cider_score",
    "cider_scores",
    "distinct_2",
    "duplicate_rate",
    "normalized_excess_duplicate_rate",
    "repeated_group_membership_rate",
    "most_frequent",
    "qbleu4_variant",
    "question_type",
    "ClipScorer",
    "clipscore_from_embeddings",
]
