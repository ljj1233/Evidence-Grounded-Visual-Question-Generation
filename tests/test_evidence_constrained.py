from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from vvqg.evidence_constrained.anchors import anchor_eligibility
from vvqg.evidence_constrained.counterfactual import (
    FINAL_HOLDOUT,
    assert_not_final_holdout,
    dedup_train,
    derangement,
)
from vvqg.evidence_constrained.objective import (
    delta_counterfactual,
    kl_to_reference,
    margin_loss,
    split_pairs,
    token_normalized_logprob,
)
from vvqg.evidence_constrained.reporting import within_image_duplicate_rate
from vvqg.training.policy_logprobs import gather_token_logprobs

ROOT = Path(__file__).resolve().parents[1]


def _support(state="NO_CONFIRMED_VIOLATION", floor=1.0, status="ok"):
    return {"flat_state": state, "support_floor": floor, "status": status}


def test_dedup_train_image_ids_unique():
    rows = [{"image_id": "a", "image": "a.jpg"}, {"image_id": "a", "image": "a.jpg"}, {"image_id": "b", "image": "b.jpg"}]
    deduped = dedup_train(rows)
    assert [row["image_id"] for row in deduped] == ["a", "b"]


def test_derangement_has_no_fixed_points_and_is_deterministic():
    first = derangement(200, seed=42)
    assert (first == torch.as_tensor(first)).all()
    assert not (first == torch.arange(200).numpy()).any()
    second = derangement(200, seed=42)
    assert (first == second).all()


def test_token_normalization_is_mean():
    values = torch.tensor([-1.0, -2.0, -3.0])
    assert torch.isclose(token_normalized_logprob(values), torch.tensor(-2.0))


def test_gather_token_logprobs_excludes_padding():
    torch.manual_seed(0)
    logits = torch.randn(1, 6, 5)
    input_ids = torch.tensor([[1, 2, 3, 4, 0, 0]])
    out = gather_token_logprobs(logits, input_ids, [3], [2])
    assert len(out) == 1 and out[0].numel() == 2
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    expected = torch.stack([log_probs[0, 2, 4], log_probs[0, 3, 0]])
    assert torch.allclose(out[0], expected)


def test_split_pairs_layout_keeps_original_and_shuffled_aligned():
    per_token = [torch.tensor([-1.0]), torch.tensor([-2.0]), torch.tensor([-10.0]), torch.tensor([-20.0])]
    plus, minus = split_pairs(per_token, 2)
    assert [float(p) for p in plus] == [-1.0, -2.0]
    assert [float(m) for m in minus] == [-10.0, -20.0]
    delta = delta_counterfactual(token_normalized_logprob(plus[0]), token_normalized_logprob(minus[0]))
    assert float(delta) == 9.0


def test_delta_cf_sign():
    assert float(delta_counterfactual(torch.tensor(-1.0), torch.tensor(-2.0))) > 0
    assert float(delta_counterfactual(torch.tensor(-2.0), torch.tensor(-1.0))) < 0


def test_margin_loss_zero_when_delta_at_or_above_margin():
    assert float(margin_loss(torch.tensor([0.5, 1.0, 2.0]), margin=0.5)) == 0.0
    assert float(margin_loss(torch.tensor([0.0]), margin=0.5)) == pytest.approx(0.5)


def test_kl_to_reference_zero_when_equal():
    values = torch.tensor([-1.0, -2.0])
    assert float(kl_to_reference(values, values.clone())) == pytest.approx(0.0)
    assert float(kl_to_reference(torch.tensor([-2.0]), torch.tensor([-1.0]))) > 0.0


def test_within_image_duplicate_rate_only_counts_same_image():
    same_image = within_image_duplicate_rate(["What is this?", "What is this?"], ["a", "a"])
    cross_image = within_image_duplicate_rate(["What is this?", "What is this?"], ["a", "b"])
    assert same_image == pytest.approx(0.5)
    assert cross_image == 0.0


def test_anchor_gate_conditions():
    assert anchor_eligibility("What color is the car?", _support(), 5)[0]
    assert not anchor_eligibility("", _support(), 0)[0]
    assert not anchor_eligibility("What color is the car?", _support(state="CONFIRMED_VIOLATION", floor=0.0), 5)[0]
    assert not anchor_eligibility("What color is the car?", _support(floor=0.0), 5)[0]
    assert not anchor_eligibility("What color is the car?", _support(state="FAILED", floor=-1.0), 5)[0]


def test_final_holdout_is_refused():
    assert_not_final_holdout(FINAL_HOLDOUT.parent / "other.jsonl")
    with pytest.raises(ValueError):
        assert_not_final_holdout(FINAL_HOLDOUT)


@pytest.mark.integration
@pytest.mark.external_artifact(
    'outputs/phase4b_online_uvfc/data/train_10000.jsonl',
    'outputs/same_backbone_baselines/manifests/development1000_images.jsonl',
)
def test_train_and_development_are_disjoint():
    train = [json.loads(line) for line in (ROOT / "outputs/phase4b_online_uvfc/data/train_10000.jsonl").read_text().splitlines() if line.strip()]
    dev = [json.loads(line) for line in (ROOT / "outputs/same_backbone_baselines/manifests/development1000_images.jsonl").read_text().splitlines() if line.strip()]
    train_stems = {Path(row["image_id"]).stem for row in train}
    dev_stems = {Path(row["image_path"]).stem for row in dev}
    assert train_stems.isdisjoint(dev_stems)
