import torch
import pytest

from vvqg.baselines.common import masked_mean
from vvqg.baselines.contrastive_grounding import ContrastiveGroundingHead
from vvqg.baselines.implicit_guided import CATEGORIES, ImplicitGuidanceHead, hard_select
from vvqg.baselines.seeq_style import dual_format_records


def test_masked_mean_uses_target_tokens_only():
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    assert torch.equal(masked_mean(hidden, torch.tensor([[1, 1, 0]], dtype=torch.bool)), torch.tensor([[2.0, 3.0]]))


def test_contrastive_loss_is_finite_and_reaches_projection():
    head = ContrastiveGroundingHead(4, 3)
    hidden = torch.randn(2, 5, 4)
    mask = torch.ones(2, 5, dtype=torch.bool)
    z = torch.nn.functional.normalize(torch.randn(2, 3), dim=-1)
    out = head.loss(hidden, mask, z, z.roll(1, 0), z.roll(1, 0), torch.tensor(2.0), 0)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    assert head.projection.weight.grad is not None


def test_hard_select_has_k_active_candidates():
    selected = hard_select(torch.tensor([[1.0, 2.0, 3.0, 4.0]]), k=2, training=False)
    assert selected.sum().item() == pytest.approx(2.0)
    assert torch.equal(selected.nonzero(as_tuple=False)[:, 1], torch.tensor([2, 3]))


def test_guidance_uses_official_sixteen_categories_and_two_objects():
    assert len(CATEGORIES) == 16
    head = ImplicitGuidanceHead(8, 6)
    out = head(torch.randn(3, 5, 8), training=False)
    assert out["category_logits"].shape == (3, 16)
    assert torch.allclose(out["selection"].sum(-1), torch.full((3,), 2.0))
    assert out["guides"].shape == (3, 4, 6)


def test_seeq_dual_format_count():
    triples = [{"image_id": "1", "image": "x.jpg", "question": "What?", "answer": "yes"}]
    rows = dual_format_records(triples)
    assert len(rows) == 2
    assert {row["format"] for row in rows} == {"qg", "qa"}
