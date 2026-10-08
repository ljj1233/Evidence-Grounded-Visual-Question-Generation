"""Phase 4A.4 audit regression tests (lightweight, no model load)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.integration
@pytest.mark.external_artifact(
    'outputs/phase3a_grpo_pilot/preflight/rollouts_flat.jsonl',
    'outputs/phase3a_grpo_pilot/preflight/rollouts_uvfc.jsonl',
    'outputs/phase4a_uvfc_rft/data/uvfc_violation_train.jsonl',
)
def test_uvfc_violation_selection_alignment_from_raw():
    select_mod = _load(ROOT / "scripts/vvqg/phase4a_select.py", "p4a_select")
    uvfc = _read(ROOT / "outputs/phase3a_grpo_pilot/preflight/rollouts_uvfc.jsonl")
    flat = {(r["image"], int(r["generation_id"])): r["state"] for r in _read(ROOT / "outputs/phase3a_grpo_pilot/preflight/rollouts_flat.jsonl")}
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in uvfc:
        groups[row["image"]].append({**row, "state": flat[(row["image"], int(row["generation_id"]))]})
    frozen = {r["image"]: r["question"] for r in _read(ROOT / "outputs/phase4a_uvfc_rft/data/uvfc_violation_train.jsonl")}
    assert len(groups) == 2000
    assert all(len(v) == 4 for v in groups.values())
    mismatches = 0
    for image, members in groups.items():
        winner = select_mod.select(members, True)
        if winner["question"] != frozen[image]:
            mismatches += 1
    assert mismatches == 0


@pytest.mark.integration
@pytest.mark.external_artifact(
    'outputs/phase4a2_rft_replay/data/human_replay_2000.jsonl',
    'outputs/phase4a2_rft_replay/data/mixed_uvfc_violation_replay_1to1.jsonl',
    'outputs/phase4a_uvfc_rft/data/uvfc_violation_train.jsonl',
)
def test_replay_and_selected_share_image_ids_and_mixed_size():
    selected = _read(ROOT / "outputs/phase4a_uvfc_rft/data/uvfc_violation_train.jsonl")
    replay = _read(ROOT / "outputs/phase4a2_rft_replay/data/human_replay_2000.jsonl")
    mixed = _read(ROOT / "outputs/phase4a2_rft_replay/data/mixed_uvfc_violation_replay_1to1.jsonl")
    assert {r["image"] for r in selected} == {r["image"] for r in replay}
    assert len({r["image"] for r in selected}) == 2000
    assert len(mixed) == 4000
    assert sum(r["source"] == "uvfc_violation" for r in mixed) == 2000
    assert sum(r["source"] == "human_replay" for r in mixed) == 2000


@pytest.mark.integration
@pytest.mark.external_artifact(
    'outputs/phase4a2_rft_replay/generation/sampled4x1000.jsonl',
    'outputs/phase4a3_rft_replay_multiseed/seed43/generation/sampled4x1000.jsonl',
    'outputs/same_backbone_baselines/manifests/development1000_images.jsonl',
)
def test_sampled4_has_exactly_four_candidates_per_dev_image():
    dev = {str(r["image_id"]) for r in _read(ROOT / "outputs/same_backbone_baselines/manifests/development1000_images.jsonl")}
    sampled_files = [
        ROOT / "outputs/phase4a2_rft_replay/generation/sampled4x1000.jsonl",
        ROOT / "outputs/phase4a3_rft_replay_multiseed/seed43/generation/sampled4x1000.jsonl",
    ]
    for path in sampled_files:
        rows = _read(path)
        groups: dict[str, set[int]] = defaultdict(set)
        for row in rows:
            groups[str(row["image_id"])].add(int(row["generation_id"]))
        assert set(groups) == dev
        assert all(v == {0, 1, 2, 3} for v in groups.values())


def test_paired_bootstrap_is_paired_by_position():
    from vvqg.evaluation.common.bootstrap import paired_bootstrap_diff

    identical = paired_bootstrap_diff([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], 1000, 42)
    assert identical["mean_diff"] == 0.0
    assert identical["lo"] == identical["hi"] == 0.0
    a = paired_bootstrap_diff([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], 500, 42)
    b = paired_bootstrap_diff([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], 500, 42)
    assert a == b  # deterministic
    assert a["mean_diff"] == pytest.approx(2.0)


def test_support_summary_class_rates_sum_to_one():
    from vvqg.evaluation.visual_support.api import summarize
    from vvqg.rewards.support.calibration import overall_support

    def rec(premise, clarity, answer):
        return {"status": "ok", "premise_consistency": premise, "reference_clarity": clarity, "visual_answerability": answer, "failure_type": "none", "overall_visual_support": overall_support(premise, clarity, answer)}

    records = [rec("supported", "clear", "answerable"), rec("unsupported", "clear", "answerable"), rec("uncertain", "clear", "answerable"), {"status": "api_failure"}]
    summary = summarize(records)
    assert summary["valid"] == 3
    rates = summary["overall_visual_support"]["rates"]
    assert rates["supported"] + rates["unsupported"] + rates["uncertain"] == pytest.approx(1.0)


def test_seeq_failed_request_is_retried_only_with_retry_flag(tmp_path, monkeypatch):
    import asyncio

    from vvqg.evaluation.visual_centric import seeq

    monkeypatch.setattr(seeq, "load_env", lambda: {"LLM_JUDGE_BASE_URL": "http://x", "LLM_JUDGE_API_KEY": "k"})
    monkeypatch.setattr(seeq, "_audit", lambda: {"seeq_protocol": {"dimensions": [{"prompt_system": "s", "prompt_user": "u"}]}})
    monkeypatch.setattr(seeq, "image_digest", lambda path: "h")
    monkeypatch.setattr(seeq, "_image_part", lambda path: {})

    calls = {"n": 0}

    class Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            scores = {k: 3 for k in seeq.DIMENSIONS}
            return {"choices": [{"message": {"content": json.dumps(scores)}}]}

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, json=None):
            calls["n"] += 1
            return Resp()

    monkeypatch.setattr(seeq.httpx, "AsyncClient", Client)

    model = "gpt-5.6-luna"
    prompt_hash = hashlib.sha256(("s" + "u").encode()).hexdigest()
    records = [{"image_id": "a", "image": "a.jpg", "question": "q1"}, {"image_id": "b", "image": "b.jpg", "question": "q2"}]
    out = tmp_path / "scores.jsonl"
    key_a = seeq.cache_key(records[0], model, prompt_hash, "h")
    key_b = seeq.cache_key(records[1], model, prompt_hash, "h")
    rows = [{**records[0], "cache_key": key_a, "model": model, "image_hash": "h", "status": "error", "error_type": "HTTPStatusError"}, {**records[1], "cache_key": key_b, "model": model, "image_hash": "h", "status": "ok", "scores": {k: 4 for k in seeq.DIMENSIONS}}]
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    runtime = asyncio.run(seeq.evaluate(records, out, model=model, concurrency=1, allow_variant=True, retry_failures=False))
    assert calls["n"] == 0
    assert runtime["new_requests"] == 0

    runtime = asyncio.run(seeq.evaluate(records, out, model=model, concurrency=1, allow_variant=True, retry_failures=True))
    assert calls["n"] == 1
    assert runtime["ok"] == 2
    assert runtime["failed"] == 0


@pytest.mark.integration
@pytest.mark.external_artifact(
    'outputs/phase4a_uvfc_rft/data/uvfc_violation_train.jsonl',
    'outputs/same_backbone_baselines/manifests/development1000_images.jsonl',
)
def test_development_and_final_holdout_images_are_disjoint():
    dev = {str(r["image_id"]) for r in _read(ROOT / "outputs/same_backbone_baselines/manifests/development1000_images.jsonl")}
    holdout = {str(r["image_id"]) for r in _read(ROOT / "src/vvqg/data/evaluation/final_holdout_manifest.jsonl")}
    assert dev & holdout == set()
    selected = {r["image"] for r in _read(ROOT / "outputs/phase4a_uvfc_rft/data/uvfc_violation_train.jsonl")}
    assert selected & {r["image_path"] for r in _read(ROOT / "src/vvqg/data/evaluation/final_holdout_manifest.jsonl")} == set()
