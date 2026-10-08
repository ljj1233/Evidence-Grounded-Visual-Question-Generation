"""Role-aware judge env resolution: precedence, fallback, independence guard."""
from __future__ import annotations

import pytest

from vqg.evaluation.judge_roles import (
    CONSTRUCTION,
    EVALUATION,
    assert_roles_independent,
    judge_identity,
    load_judge_env,
)


def test_role_specific_overrides_legacy_per_variable():
    env = {
        "LLM_JUDGE_API_KEY": "legacy-key",
        "LLM_JUDGE_BASE_URL": "https://legacy.example/v1",
        "LLM_JUDGE_MODEL_NAME": "legacy-model",
        "LLM_JUDGE_CONCURRENCY": "4",
        "LLM_JUDGE_MAX_RETRIES": "1",
        "LLM_JUDGE_TIMEOUT": "30",
        "EVALUATION_JUDGE_API_KEY": "eval-key",
        "EVALUATION_JUDGE_MODEL_NAME": "eval-model",
        "EVALUATION_JUDGE_CONCURRENCY": "8",
        "EVALUATION_JUDGE_TIMEOUT": "120",
    }
    cfg = load_judge_env(EVALUATION, env)
    assert cfg["api_key"] == "eval-key"
    assert cfg["model_name"] == "eval-model"
    assert cfg["concurrency"] == 8
    assert cfg["timeout"] == 120.0
    # Unset role-specific vars fall back to legacy, not to the default.
    assert cfg["base_url"] == "https://legacy.example/v1"
    assert cfg["max_retries"] == 1


def test_falls_back_to_legacy_when_no_role_vars():
    env = {
        "LLM_JUDGE_API_KEY": "k",
        "LLM_JUDGE_BASE_URL": "https://legacy.example/v1",
        "LLM_JUDGE_MODEL_NAME": "m",
        "LLM_JUDGE_CONCURRENCY": "7",
        "LLM_JUDGE_MAX_RETRIES": "5",
        "LLM_JUDGE_TIMEOUT": "45",
    }
    cfg = load_judge_env(CONSTRUCTION, env)
    assert cfg == {
        "api_key": "k",
        "base_url": "https://legacy.example/v1",
        "model_name": "m",
        "concurrency": 7,
        "max_retries": 5,
        "timeout": 45.0,
    }


def test_defaults_when_nothing_is_set():
    cfg = load_judge_env(CONSTRUCTION, {})
    assert cfg["api_key"] == "" and cfg["base_url"] == "" and cfg["model_name"] == ""
    assert cfg["concurrency"] == 32
    assert cfg["max_retries"] == 3
    assert cfg["timeout"] == 60.0


def test_unknown_role_rejected():
    with pytest.raises(ValueError):
        load_judge_env("bogus", {})


def test_assert_roles_independent_raises_on_shared_model():
    with pytest.raises(RuntimeError):
        assert_roles_independent({"LLM_JUDGE_MODEL_NAME": "gpt-5.6-luna"})


def test_assert_roles_independent_passes_when_models_differ():
    env = {
        "CONSTRUCTION_JUDGE_MODEL_NAME": "constructor-model",
        "EVALUATION_JUDGE_MODEL_NAME": "evaluator-model",
    }
    assert_roles_independent(env)  # must not raise


def test_assert_roles_independent_passes_when_only_one_role_configured():
    assert_roles_independent({"EVALUATION_JUDGE_MODEL_NAME": "evaluator-model"})


def test_assert_roles_independent_explicit_required():
    with pytest.raises(RuntimeError):
        assert_roles_independent({}, required=(CONSTRUCTION, EVALUATION))
    assert_roles_independent(
        {"CONSTRUCTION_JUDGE_MODEL_NAME": "a", "EVALUATION_JUDGE_MODEL_NAME": "b"},
        required=(CONSTRUCTION, EVALUATION),
    )


def test_judge_identity_is_stable_and_secret_free():
    env = {
        "EVALUATION_JUDGE_MODEL_NAME": "eval-model",
        "EVALUATION_JUDGE_API_KEY": "SUPER-SECRET",
        "EVALUATION_JUDGE_BASE_URL": "https://secret.example/v1",
    }
    identity = judge_identity(EVALUATION, env)
    assert identity == "evaluation:eval-model"
    assert "SUPER-SECRET" not in identity
    assert "secret.example" not in identity
    assert judge_identity(EVALUATION, env) == judge_identity(EVALUATION, env)


def test_judge_identity_accepts_resolved_config():
    resolved = load_judge_env(EVALUATION, {"EVALUATION_JUDGE_MODEL_NAME": "m"})
    assert judge_identity(EVALUATION, resolved) == "evaluation:m"
