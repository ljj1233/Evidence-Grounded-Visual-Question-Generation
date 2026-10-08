"""Tests for the Phase 3A fixed-rollout group-relative pilot (checks A-K)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vvqg.training import fixed_group_relative as fgr  # noqa: E402
from vvqg.training.policy_logprobs import gather_token_logprobs  # noqa: E402

RANK_REWARDS = (1.5, 0.5, -0.5, -1.5)
IMAGE = "data/vqav2/raw/images/train2014/COCO_train2014_000000000263.jpg"


def member(question, uvfc, state="NO_CONFIRMED_VIOLATION", generation_id=0):
    return {
        "image_id": IMAGE,
        "image": IMAGE,
        "question": question,
        "generation_id": generation_id,
        "UVFC": uvfc,
        "flat_state": state,
    }


def rewards_by_question(rows):
    return {row["question"]: row["canonical_reward"] for row in rows}


# A. Tie mid-rank assignment.
def test_tie_midrank_assignment():
    rows = [
        member("a", 9.0, generation_id=0),
        member("b", 9.0, generation_id=1),
        member("c", 5.0, generation_id=2),
        member("d", 1.0, generation_id=3),
    ]
    out = fgr.canonicalize_group(rows, "p1")
    reward = rewards_by_question(out)
    assert reward == {"a": 1.0, "b": 1.0, "c": -0.5, "d": -1.5}
    assert all(row["tie_block_size"] == 2 for row in out if row["question"] in ("a", "b"))


def test_full_tie_zero_reward_and_zero_variance():
    rows = [member(letter, 7.0, generation_id=index) for index, letter in enumerate("abcd")]
    out = fgr.canonicalize_group(rows, "p1")
    assert all(row["canonical_reward"] == 0.0 for row in out)
    assert all(row["zero_variance_group"] for row in out)


# B. Exact duplicate identical questions cannot get different canonical reward.
def test_exact_duplicate_same_reward():
    rows = [
        member("What color is the frisbee?", 0.4, generation_id=0),
        member("What color is the frisbee?", 0.4, generation_id=1),
        member("other question", 0.9, generation_id=2),
        member("third question", 0.1, generation_id=3),
    ]
    out = fgr.canonicalize_group(rows, "p1")
    reward = rewards_by_question(out)
    assert reward["What color is the frisbee?"] == reward["What color is the frisbee?"]
    duplicates = [row for row in out if row["is_duplicate"]]
    assert len(duplicates) == 2
    assert duplicates[0]["canonical_reward"] == duplicates[1]["canonical_reward"]


def test_exact_duplicate_uvfc_inconsistency_raises():
    rows = [
        member("same?", 0.4, generation_id=0),
        member("same?", 0.5, generation_id=1),
        member("x", 0.2, generation_id=2),
        member("y", 0.1, generation_id=3),
    ]
    with pytest.raises(ValueError, match="UVFC_DUPLICATE_INCONSISTENCY"):
        fgr.canonicalize_group(rows, "p1")


# C. P2 lexicographic ordering: admissible before confirmed violation.
def test_p2_violation_last():
    rows = [
        member("violating", 9.0, "CONFIRMED_VIOLATION", generation_id=0),
        member("clean", 1.0, "NO_CONFIRMED_VIOLATION", generation_id=1),
        member("uncertain", 2.0, "UNCERTAIN", generation_id=2),
        member("failed", 0.5, "FAILED", generation_id=3),
    ]
    out = fgr.canonicalize_group(rows, "p2")
    reward = rewards_by_question(out)
    assert reward["violating"] == -1.5
    assert reward["uncertain"] == 1.5
    assert reward["clean"] == 0.5
    assert reward["failed"] == -0.5
    assert all(row["training_flat_state"] != "FAILED" or row["question"] == "failed" for row in out)


def test_p2_flat_duplicate_repair():
    rows = [
        member("dup", 3.0, "FAILED", generation_id=0),
        member("dup", 3.0, "NO_CONFIRMED_VIOLATION", generation_id=1),
        member("x", 1.0, "CONFIRMED_VIOLATION", generation_id=2),
        member("y", 0.5, "CONFIRMED_VIOLATION", generation_id=3),
    ]
    out = fgr.canonicalize_group(rows, "p2")
    states = {row["training_flat_state"] for row in out if row["question"] == "dup"}
    assert states == {"NO_CONFIRMED_VIOLATION"}


# D. Zero-variance group skip.
def test_zero_variance_group_skipped(tmp_path):
    path = tmp_path / "p1.jsonl"
    rows = []
    for index, letter in enumerate("abcd"):
        row = member(letter, 7.0, generation_id=index)
        row["group_id"] = IMAGE
        row["candidate_id"] = f"{IMAGE}#{index}"
        row["zero_variance_group"] = True
        row["advantage"] = 0.0
        rows.append(row)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert fgr.load_training_groups(path) == {}


# E. Completion token mask excludes prompt/image tokens.
def test_completion_mask_positions():
    torch = pytest.importorskip("torch")
    input_ids = torch.tensor([[10, 11, 12, 13, 14, 15]])
    starts, counts = [3], [3]
    positions = torch.arange(starts[0] - 1, starts[0] + counts[0] - 1)
    assert positions.tolist() == [2, 3, 4]
    logits = torch.zeros(1, 6, 20)
    logits[0, positions, input_ids[0, starts[0] : starts[0] + counts[0]]] = 5.0
    tokens = gather_token_logprobs(logits, input_ids, starts, counts)
    assert tokens[0].shape == (3,)
    assert all(value > -1.0 for value in tokens[0].tolist())


# F. Old-logprob parity: gathering is deterministic.
def test_gather_deterministic():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    logits = torch.randn(2, 8, 17)
    input_ids = torch.randint(0, 17, (2, 8))
    starts, counts = [3, 4], [4, 3]
    first = gather_token_logprobs(logits, input_ids, starts, counts)
    second = gather_token_logprobs(logits, input_ids, starts, counts)
    assert all(torch.equal(a, b) for a, b in zip(first, second))


# G. PPO clipped-loss sign.
def test_clipped_surrogate_clipping():
    torch = pytest.importorskip("torch")
    one = torch.tensor(1.0)
    assert torch.isclose(fgr.clipped_surrogate(torch.tensor(1.5), one), torch.tensor(1.2))
    assert torch.isclose(fgr.clipped_surrogate(torch.tensor(0.5), one), torch.tensor(0.5))
    assert torch.isclose(fgr.clipped_surrogate(torch.tensor(1.5), -one), torch.tensor(-1.5))


# H. Positive advantages increase preferred response likelihood.
def test_positive_advantage_gradient_direction():
    torch = pytest.importorskip("torch")
    old = torch.tensor([-4.0, -4.0])
    for advantage in (1.0, -1.0):
        logp = torch.tensor([-4.0, -4.0], requires_grad=True)
        ratio = fgr.importance_ratio(logp, old)
        loss = -fgr.clipped_surrogate(ratio, torch.tensor(advantage)).mean()
        loss.backward()
        assert (logp.grad > 0).all() if advantage < 0 else (logp.grad < 0).all()


# I. Base model stays frozen; only adapter parameters are trainable.
def test_count_parameters_freezes_base():
    torch = pytest.importorskip("torch")

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Linear(4, 4)
            self.adapter = torch.nn.Linear(4, 4)
            for parameter in self.base.parameters():
                parameter.requires_grad_(False)

    report = fgr.count_parameters(Wrapper())
    assert report["trainable"] == 16 + 4
    assert report["total"] == 2 * (16 + 4)


# J. P1/P2 both initialize from identical SFT hash.
def test_p1_p2_share_sft_checkpoint():
    yaml = pytest.importorskip("yaml")
    p1 = yaml.safe_load((ROOT / "configs/phase3a/p1_uvfc_fixed_group.yaml").read_text())
    p2 = yaml.safe_load((ROOT / "configs/phase3a/p2_uvfc_violation_fixed_group.yaml").read_text())
    assert p1["sft_checkpoint"] == p2["sft_checkpoint"]
    assert p1["seed"] == p2["seed"] == 42
    assert p1["mode"] == "p1" and p2["mode"] == "p2"


# K. final_holdout IDs never appear in training / monitor inputs.
@pytest.mark.integration
@pytest.mark.external_artifact(
    'outputs/phase3a_fixed_group_relative/monitor128.jsonl',
)
def test_no_holdout_leakage():
    holdout = {
        json.loads(line)["image_id"]
        for line in (ROOT / "src/vvqg/data/evaluation/final_holdout_manifest.jsonl").read_text().splitlines()
        if line.strip()
    }
    monitor = [
        json.loads(line)
        for line in (ROOT / "outputs/phase3a_fixed_group_relative/monitor128.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert monitor
    assert not any(str(row["image_id"]) in holdout for row in monitor)
    assert not any("val2014" in str(row["image_path"]) for row in monitor)
