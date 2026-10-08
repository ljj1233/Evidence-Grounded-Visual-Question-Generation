"""Role-aware judge configuration resolution (construction vs evaluation).

This module is the single, stdlib-only source of truth for *which environment
variables* configure each judge **role**.  It keeps a construction-time judge
separate from the judge that scores frozen evaluation outputs, so the same model
is not both building a training signal and the sole instrument scoring it.

Two roles are defined:

``construction``
    Builds training-side signals.
``evaluation``
    Scores frozen predictions for reported metrics.

Environment variable precedence for a role, e.g. ``evaluation``::

    EVALUATION_JUDGE_API_KEY      -> LLM_JUDGE_API_KEY
    EVALUATION_JUDGE_BASE_URL     -> LLM_JUDGE_BASE_URL
    EVALUATION_JUDGE_MODEL_NAME   -> LLM_JUDGE_MODEL_NAME
    EVALUATION_JUDGE_CONCURRENCY  -> LLM_JUDGE_CONCURRENCY   (default 32)
    EVALUATION_JUDGE_MAX_RETRIES  -> LLM_JUDGE_MAX_RETRIES   (default 3)
    EVALUATION_JUDGE_TIMEOUT      -> LLM_JUDGE_TIMEOUT       (default 60)

The legacy ``LLM_JUDGE_*`` names remain the fallback, so every existing
deployment keeps working unchanged.  When neither name is set the numeric
fields fall back to the historical defaults and the string fields are ``""``;
callers remain responsible for their own "required variable" errors.

Import safety: no network calls, no file IO, no mutation of the environment.
"""

from __future__ import annotations

import os
from typing import Mapping

__all__ = [
    "CONSTRUCTION",
    "EVALUATION",
    "assert_roles_independent",
    "judge_identity",
    "load_judge_env",
]

CONSTRUCTION = "construction"
EVALUATION = "evaluation"

_ROLES = (CONSTRUCTION, EVALUATION)
_LEGACY_PREFIX = "LLM_JUDGE"

# (result key, env suffix, default-as-string)
_FIELDS = (
    ("api_key", "API_KEY", ""),
    ("base_url", "BASE_URL", ""),
    ("model_name", "MODEL_NAME", ""),
    ("concurrency", "CONCURRENCY", "32"),
    ("max_retries", "MAX_RETRIES", "3"),
    ("timeout", "TIMEOUT", "60"),
)

_DEFAULT_CONCURRENCY = 32
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_TIMEOUT = 60.0


def _normalize_role(role: str) -> str:
    normalized = str(role).strip().lower()
    if normalized not in _ROLES:
        raise ValueError(
            f"unknown judge role {role!r}; expected one of {_ROLES}")
    return normalized


def _role_env_names(role: str) -> dict[str, str]:
    prefix = f"{role.upper()}_JUDGE"
    return {field: f"{prefix}_{suffix}" for field, suffix, _ in _FIELDS}


def _legacy_env_name(suffix: str) -> str:
    return f"{_LEGACY_PREFIX}_{suffix}"


def _as_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_judge_env(role: str, environ: Mapping[str, str] | None = None) -> dict:
    """Resolve a judge role's config from environment variables.

    Precedence is per-variable: ``{ROLE}_JUDGE_*`` first, then the legacy
    ``LLM_JUDGE_*`` fallback, then the historical default.  An empty string is
    treated as "unset" so explicit blanks still fall through to the legacy
    name.

    Args:
        role: ``CONSTRUCTION`` or ``EVALUATION`` (case-insensitive).
        environ: mapping to read from; defaults to ``os.environ``.  Never
            mutated.

    Returns:
        ``dict`` with keys ``api_key`` (str), ``base_url`` (str),
        ``model_name`` (str), ``concurrency`` (int), ``max_retries`` (int),
        ``timeout`` (float).
    """
    normalized = _normalize_role(role)
    env = os.environ if environ is None else environ
    role_names = _role_env_names(normalized)

    resolved: dict[str, object] = {}
    for field, suffix, default in _FIELDS:
        value = env.get(role_names[field])
        if value is None or value == "":
            value = env.get(_legacy_env_name(suffix))
        if value is None or value == "":
            value = default
        resolved[field] = value

    resolved["concurrency"] = _as_int(resolved["concurrency"], _DEFAULT_CONCURRENCY)
    resolved["max_retries"] = _as_int(resolved["max_retries"], _DEFAULT_MAX_RETRIES)
    resolved["timeout"] = _as_float(resolved["timeout"], _DEFAULT_TIMEOUT)
    return resolved


def _resolve_model_name(role: str, env: Mapping[str, str] | None) -> str:
    """Return the model name for ``env`` (raw environ or a resolved config)."""
    if (isinstance(env, Mapping) and "model_name" in env and "api_key" in env
            and not any(str(k).startswith(_LEGACY_PREFIX) for k in env)):
        return str(env.get("model_name") or "")
    return str(load_judge_env(role, env)["model_name"])


def judge_identity(role: str, env: Mapping[str, str] | None = None) -> str:
    """Return a stable, secret-free fingerprint for a judge role.

    The fingerprint is ``"{role}:{model_name}"``.  It deliberately excludes
    ``api_key`` and ``base_url`` (credentials/endpoints must never be written
    into manifests or cache keys).  ``env`` may be a raw environment mapping or
    a dict already returned by :func:`load_judge_env`.
    """
    normalized = _normalize_role(role)
    model_name = _resolve_model_name(normalized, env)
    return f"{normalized}:{model_name}"


def assert_roles_independent(
    env: Mapping[str, str] | None = None,
    *,
    required: tuple[str, ...] | None = None,
) -> None:
    """Fail if two required judge roles resolve to the same model.

    With the default ``required=None``, "required" is inferred from the
    environment: a role is required when it resolves to a non-empty
    ``model_name``.  This is the common case: a check raises only when *both*
    construction and evaluation are configured and they resolve to the same
    model.  When at most one role is configured there is nothing to assert.

    With an explicit ``required`` iterable, every listed role must resolve to a
    non-empty model and all listed roles must use distinct model names.

    Raises:
        ValueError: on an unknown role.
        RuntimeError: on a missing required/configured model or a collision.
    """
    env_map = os.environ if env is None else env

    if required is None:
        construction = load_judge_env(CONSTRUCTION, env_map)
        evaluation = load_judge_env(EVALUATION, env_map)
        c_model = construction["model_name"]
        e_model = evaluation["model_name"]
        if c_model and e_model and c_model == e_model:
            raise RuntimeError(
                "construction and evaluation judges resolve to the same "
                f"model_name ({c_model!r}); set EVALUATION_JUDGE_MODEL_NAME "
                "(and the other EVALUATION_JUDGE_* vars) to a model independent "
                "of the construction judge (CONSTRUCTION_JUDGE_MODEL_NAME / "
                "LLM_JUDGE_MODEL_NAME).")
        return

    roles = tuple(_normalize_role(r) for r in required)
    resolved = {role: load_judge_env(role, env_map) for role in roles}
    missing = [r for r in roles if not resolved[r]["model_name"]]
    if missing:
        raise RuntimeError(
            f"required judge role(s) without a model_name: {missing}")
    models = {role: resolved[role]["model_name"] for role in roles}
    if len(set(models.values())) < len(models):
        raise RuntimeError(
            "judge roles are not independent; colliding model_name(s): "
            f"{models}")