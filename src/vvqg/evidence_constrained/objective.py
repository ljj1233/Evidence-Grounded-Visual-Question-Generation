"""CER objective terms (pure functions; torch imported lazily for CPU tests)."""

from __future__ import annotations

from typing import Any, Sequence


def token_normalized_logprob(token_logprobs: Any) -> Any:
    """Mean per-token log-probability of one completion (prompt/pad excluded upstream)."""
    return token_logprobs.mean()


def delta_counterfactual(lp_plus: Any, lp_minus: Any) -> Any:
    """Positive means the anchor question is more likely on the original image."""
    return lp_plus - lp_minus


def margin_loss(delta_cf: Any, margin: float) -> Any:
    import torch

    return torch.relu(margin - delta_cf).mean()


def kl_to_reference(current_tokens: Any, ref_tokens: Any) -> Any:
    """Repo-canonical sampled KL estimator; reference log-probs must be detached."""
    import torch

    log_ratio = ref_tokens - current_tokens
    return (torch.exp(log_ratio) - log_ratio - 1.0).mean()


def split_pairs(per_token: Sequence[Any], n_pairs: int) -> tuple[list[Any], list[Any]]:
    """Items are laid out ``[plus_0 .. plus_{n-1}, minus_0 .. minus_{n-1}]``."""
    if len(per_token) != 2 * n_pairs:
        raise ValueError(f"expected {2 * n_pairs} items, got {len(per_token)}")
    return list(per_token[:n_pairs]), list(per_token[n_pairs:])


def paired_deltas(plus: Sequence[Any], minus: Sequence[Any]) -> Any:
    import torch

    return torch.stack([token_normalized_logprob(p) - token_normalized_logprob(m) for p, m in zip(plus, minus)])
