"""Candidate-local slot assignment for Joint Soft-Evidence QSE v2.

The frozen question encoder is the Qwen3-VL language model applied to text only
(no image). Candidate embeddings are detached before assignment. Assignment is
computed independently per image with Hungarian maximum matching.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

SUPPORT_WEIGHT = {"supported": 1.0, "uncertain": 0.5, "unsupported": 0.0, None: 0.0}
TEMPERATURE = 0.07


class FrozenQuestionEncoder(nn.Module):
    """Frozen text-only Qwen3-VL encoder + frozen random projection."""

    def __init__(self, base_model, tokenizer, slot_dim: int, seed: int = 42, max_length: int = 64):
        super().__init__()
        self.lm = base_model.model.language_model
        self.lm.requires_grad_(False)
        self.tokenizer = tokenizer
        self.max_length = max_length
        hidden = int(base_model.config.text_config.hidden_size)
        generator = torch.Generator().manual_seed(seed)
        projection = torch.empty(hidden, slot_dim)
        nn.init.normal_(projection, std=hidden ** -0.5, generator=generator)
        self.register_buffer("projection", projection)

    @torch.no_grad()
    def encode(self, questions: list[str], device, batch_size: int = 32) -> torch.Tensor:
        outputs = []
        for start in range(0, len(questions), batch_size):
            chunk = questions[start : start + batch_size]
            encoded = self.tokenizer(chunk, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            mask = encoded["attention_mask"].to(device)
            hidden = self.lm(input_ids=input_ids, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
            pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            outputs.append((pooled.float() @ self.projection.float()).cpu())
        return torch.cat(outputs, 0) if outputs else torch.zeros(0, self.projection.shape[1])


def hungarian_match(similarity: torch.Tensor) -> tuple[list[int], list[int]]:
    """Maximise the matched similarity. similarity: [n_slots, n_candidates]."""
    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(-similarity.detach().cpu().numpy())
    return rows.tolist(), cols.tolist()


def assignment_loss(
    evidence: torch.Tensor,
    candidate_emb: torch.Tensor,
    candidate_weight: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    temperature: float = TEMPERATURE,
    min_candidates: int = 2,
) -> tuple[torch.Tensor, dict]:
    """Weighted matched-pair NLL, one Hungarian assignment per image.

    evidence [B,N,ds]; candidate_emb [B,m,ds]; candidate_weight [B,m]; candidate_mask [B,m].
    Only candidates with weight > 0 and a valid mask participate.
    """
    z_e = F.normalize(evidence.float(), dim=-1)
    z_q = F.normalize(candidate_emb.float(), dim=-1)
    similarity = torch.einsum("bnd,bmd->bnm", z_e, z_q)  # [B,N,m]
    n_slots = evidence.shape[1]
    losses, usage = [], torch.zeros(n_slots)
    matched_sims, shuffled_sims, entropies = [], [], []
    skipped = 0
    for b in range(evidence.shape[0]):
        valid = [j for j in range(candidate_mask.shape[1])
                 if bool(candidate_mask[b, j]) and float(candidate_weight[b, j]) > 0]
        if len(valid) < min_candidates:
            skipped += 1
            continue
        sub = similarity[b][:, valid]  # [N,m']
        rows, cols = hungarian_match(sub)
        logits = sub / temperature
        log_probs = F.log_softmax(logits, dim=1)
        weights = candidate_weight[b][valid][cols]
        matched = log_probs[rows, cols]
        denom = weights.sum().clamp_min(1e-6)
        losses.append(-(matched * weights).sum() / denom)
        for slot in rows:
            usage[slot] += 1
        matched_sims.append(float(sub[rows, cols].mean()))
        shuffled_cols = [(c + 1) % len(valid) for c in cols]
        shuffled_sims.append(float(sub[rows, shuffled_cols].mean()))
        entropies.append(float(-(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)).sum(1).mean()))
    stats = {
        "assign_skipped": skipped,
        "assign_entropy": sum(entropies) / len(entropies) if entropies else 0.0,
        "matched_sim": sum(matched_sims) / len(matched_sims) if matched_sims else 0.0,
        "shuffled_sim": sum(shuffled_sims) / len(shuffled_sims) if shuffled_sims else 0.0,
        "assign_gap": (sum(matched_sims) - sum(shuffled_sims)) / len(matched_sims) if matched_sims else 0.0,
        "slot_usage": usage,
    }
    if not losses:
        return evidence.sum() * 0.0, stats
    return torch.stack(losses).mean(), stats


def slot_diversity_loss(evidence: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Penalise only overly similar slot pairs (post-LayerNorm)."""
    z = F.normalize(evidence.float(), dim=-1)
    similarity = torch.einsum("bnd,bjd->bnj", z, z)
    n = similarity.shape[1]
    offdiag = 1.0 - torch.eye(n, device=similarity.device)
    penalty = F.relu(similarity - threshold) * offdiag
    return penalty.sum(dim=(1, 2)).mean() / (n * (n - 1))
