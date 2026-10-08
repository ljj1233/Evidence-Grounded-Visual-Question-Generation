import json

import pytest
from pathlib import Path

from vvqg.baselines.seeq_faithful import (
    INTENT_NAMES, candidate_key, decide, dual_format_records, proposal_prompt,
    resolve_round_models,
)

REPO = Path(__file__).resolve().parents[1]
M0 = "m0-adapter"
M1 = "m1-adapter"

VALID_JUDGE = {"hard_constraints": {"Factuality": {"count": 0}, "Ambiguity": {"count": 0}, "Validity": {"answerable": True}}}
INVALID_JUDGE = {"hard_constraints": {"Factuality": {"count": 1}, "Ambiguity": {"count": 0}, "Validity": {"answerable": True}}}


def _decide(**kw):
    base = dict(hard_judge=VALID_JUDGE, perception_seed=2.0, perception_rew=2.0,
                reasoning_seed=2.0, reasoning_rew=2.0, seed_question="What is the man riding?",
                rewritten_question="Which object supports the man and what colour is it?",
                seen_questions=set())
    base.update(kw)
    return decide(**base)


# 1. intent prompts exist ---------------------------------------------------
def test_all_intents_and_prompts_exist():
    assert set(INTENT_NAMES) == {"direct_recognition", "comparison", "spatial_relations", "scene_understanding", "grounded_reasoning"}
    for intent in INTENT_NAMES:
        prompt = proposal_prompt(intent)
        assert "Question" in prompt and "Visual intent:" in prompt and len(prompt) > 100


# 2 + 3. round proposer/rewrite reference -----------------------------------
def test_round_models_r1_m0_and_r2_m1_with_frozen_m0_reference():
    assert resolve_round_models(1, M0, M1) == (M0, M0)
    assert resolve_round_models(2, M0, M1) == (M1, M0)  # R2 proposer M1, rewrite/filter M0


def test_round2_requires_m1():
    import pytest
    with pytest.raises(ValueError):
        resolve_round_models(2, M0, None)


# 4. acceptance rule --------------------------------------------------------
def test_accept_requires_valid_and_improvement():
    assert _decide(reasoning_rew=3.0)["accepted"] is True           # reasoning improved (delta 1.0)
    assert _decide(perception_rew=4.0)["accepted"] is True          # perception improved (delta 2.0)
    flat = _decide()
    assert flat["accepted"] is False and flat["rejection_reason"] == "difficulty_not_improved_enough"
    assert _decide(hard_judge=INVALID_JUDGE, reasoning_rew=3.0)["accepted"] is False


def test_perception_improvement_alone_needs_full_unit():
    assert _decide(perception_rew=2.5)["accepted"] is False          # delta 0.5 < 1.0
    assert _decide(perception_rew=3.0)["accepted"] is True           # delta 1.0


# 5. parser failure is never accepted ---------------------------------------
def test_parser_failure_is_never_accepted():
    out = _decide(perception_rew=None, reasoning_rew=9.0)
    assert out["accepted"] is False and out["rejection_reason"] == "filter_parse_failed"


def test_duplicate_and_high_overlap_rejected():
    assert _decide(reasoning_rew=3.0, seen_questions={"which object supports the man and what colour is it?"})["rejection_reason"] == "duplicate_simplified_question"
    assert _decide(reasoning_rew=3.0, rewritten_question="What is the man riding?")["rejection_reason"] in {"lexical_overlap_too_high", "difficulty_not_improved_enough"}


# 6. dual-format counts -----------------------------------------------------
def test_dual_format_counts_are_qg_plus_qa():
    triples = [{"image_id": str(i), "image": f"{i}.jpg", "question": f"q{i}?", "answer": f"a{i}", "intent": "comparison"} for i in range(5)]
    records = dual_format_records(triples)
    assert len(records) == 10
    assert sum(r["format"] == "qg" for r in records) == 5
    assert sum(r["format"] == "qa" for r in records) == 5
    assert all(r["target"] == r["question"] for r in records if r["format"] == "qg")
    assert all(r["target"] == r["answer"] for r in records if r["format"] == "qa")


# 7. train pool excludes dev and final_holdout ------------------------------
def _ids(path: Path) -> set[str]:
    return {str(json.loads(line)["image_id"]) for line in path.read_text().splitlines() if line.strip()}


@pytest.mark.integration
@pytest.mark.external_artifact(
    'outputs/same_backbone_baselines/manifests/development1000_images.jsonl',
    'outputs/same_backbone_baselines/manifests/train_unlabeled_images.jsonl',
)
def test_train_pool_disjoint_from_dev_and_holdout():
    train = _ids(REPO / "outputs/same_backbone_baselines/manifests/train_unlabeled_images.jsonl")
    dev = _ids(REPO / "outputs/same_backbone_baselines/manifests/development1000_images.jsonl")
    holdout = _ids(REPO / "src/vvqg/data/evaluation/final_holdout_manifest.jsonl")
    assert train and dev and holdout
    assert train & dev == set()
    assert train & holdout == set()


# 8. resume key carries intent/sample identity ------------------------------
def test_candidate_key_contains_round_intent_sample():
    key = candidate_key("86", 2, "spatial_relations", 1)
    assert key == "r2-86-spatial_relations-1"
    assert "spatial_relations" in key and key.endswith("-1") and key.startswith("r2-")


# 9. training continues the SFT adapter, never a random LoRA -----------------
def test_training_continues_start_adapter():
    source = (REPO / "scripts/vvqg/train_seeq_faithful.py").read_text()
    assert "PeftModel.from_pretrained" in source and "is_trainable=True" in source
    assert "LoraConfig(" not in source and "get_peft_model" not in source
