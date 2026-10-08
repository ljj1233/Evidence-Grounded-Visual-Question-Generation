"""Unit tests for v2 candidate-assigned soft slots (pure functions, no 8B model)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vvqg.evidence.assignment import (  # noqa: E402
    FrozenQuestionEncoder, assignment_loss, hungarian_match, slot_diversity_loss,
)


def test_hungarian_maximises():
    sim = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    rows, cols = hungarian_match(sim)
    assert sorted(cols) == [0, 1]
    assert sim[rows, cols].sum() > 1.6


def test_assignment_shape_and_weighting():
    evidence = torch.randn(2, 4, 16)
    cand = torch.randn(2, 4, 16)
    weight = torch.tensor([[1.0, 0.5, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]])
    mask = torch.ones(2, 4)
    loss, stats = assignment_loss(evidence, cand, weight, mask)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert stats["slot_usage"].shape[0] == 4
    assert stats["assign_gap"] == stats["matched_sim"] - stats["shuffled_sim"]


def test_unsupported_zero_weight_skipped():
    evidence = torch.randn(1, 4, 8)
    cand = torch.randn(1, 4, 8)
    weight = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # only one positive-weight candidate
    mask = torch.ones(1, 4)
    loss, stats = assignment_loss(evidence, cand, weight, mask, min_candidates=2)
    assert stats["assign_skipped"] == 1
    assert float(loss) == 0.0


def test_duplicate_mask_skip():
    evidence = torch.randn(1, 4, 8)
    cand = torch.randn(1, 4, 8)
    weight = torch.ones(1, 4)
    mask = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # de-duplicated to <2
    loss, stats = assignment_loss(evidence, cand, weight, mask, min_candidates=2)
    assert stats["assign_skipped"] == 1


def test_assignment_is_deterministic():
    torch.manual_seed(0)
    evidence = torch.randn(2, 4, 16)
    cand = torch.randn(2, 4, 16)
    weight = torch.ones(2, 4)
    mask = torch.ones(2, 4)
    l1, s1 = assignment_loss(evidence, cand, weight, mask)
    l2, s2 = assignment_loss(evidence, cand, weight, mask)
    assert torch.equal(l1, l2) and s1["slot_usage"].equal(s2["slot_usage"])


def test_slot_permutation_invariance():
    evidence = torch.randn(1, 4, 16)
    cand = torch.randn(1, 4, 16)
    weight = torch.ones(1, 4)
    mask = torch.ones(1, 4)
    _, s1 = assignment_loss(evidence, cand, weight, mask)
    _, s2 = assignment_loss(evidence[[0]][:, [2, 0, 3, 1]], cand, weight, mask)
    assert abs(s1["matched_sim"] - s2["matched_sim"]) < 1e-6


def test_no_mean_pooling_supervision():
    """Changing one slot's evidence must change the assignment loss."""
    base = torch.randn(1, 4, 16)
    cand = torch.randn(1, 4, 16)
    weight = torch.ones(1, 4)
    mask = torch.ones(1, 4)
    l1, _ = assignment_loss(base, cand, weight, mask)
    changed = base.clone(); changed[0, 2] += 5.0
    l2, _ = assignment_loss(changed, cand, weight, mask)
    assert not torch.isclose(l1, l2)


def test_slot_diversity_direction():
    identical = torch.ones(1, 4, 8)
    distinct = torch.eye(4).unsqueeze(0).repeat(1, 1, 1)
    assert slot_diversity_loss(identical) > slot_diversity_loss(distinct)


class _StubLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(64, 8)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return SimpleNamespace(last_hidden_state=self.embed(input_ids))


class _StubTokenizer:
    def __call__(self, texts, padding=True, truncation=True, max_length=64, return_tensors="pt"):
        import torch
        ids = [[1 + (ord(ch) % 60) for ch in text] or [1] for text in texts]
        width = max(len(row) for row in ids)
        input_ids = torch.zeros(len(ids), width, dtype=torch.long)
        mask = torch.zeros(len(ids), width, dtype=torch.long)
        for i, row in enumerate(ids):
            input_ids[i, : len(row)] = torch.tensor(row)
            mask[i, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": mask}


def _encoder():
    base = SimpleNamespace(model=SimpleNamespace(language_model=_StubLM()),
                           config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=8)))
    return FrozenQuestionEncoder(base, _StubTokenizer(), slot_dim=4, seed=42)


def test_question_encoder_reproducible_and_distinct():
    enc = _encoder()
    a = enc.encode(["is the sky clear?", "is the sky clear?"], "cpu")
    assert torch.allclose(a[0], a[1])          # reproducible
    b = enc.encode(["is the sky clear?", "how many dogs?"], "cpu")
    assert not torch.allclose(b[0], b[1])      # not all identical


def test_question_encoder_padding_invariant():
    enc = _encoder()
    one = enc.encode(["cat"], "cpu")
    two = enc.encode(["cat", "cat in a very long padded sentence here"], "cpu")
    assert torch.allclose(one[0], two[0], atol=1e-5)  # padding does not change mean pool
