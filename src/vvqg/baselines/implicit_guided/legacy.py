"""Implicit-guided VQG heads using image-derived concepts and soft guide tokens."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

CATEGORIES = (
    "count", "binary", "predicate", "material", "time", "color", "attribute", "object",
    "stuff", "food", "shape", "other", "location", "animal", "spatial", "activity",
)
CATEGORY_TO_ID = {name: index for index, name in enumerate(CATEGORIES)}


@dataclass(frozen=True)
class GuidedConfig:
    k: int = 2
    temperature: float = 1.0


def hard_select(
    scores: torch.Tensor,
    k: int = 2,
    training: bool = True,
    temperature: float = 1.0,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return straight-through hard k-hot selections over candidate objects."""
    if scores.ndim != 2:
        raise ValueError("scores must be [batch, candidates]")
    valid_mask = torch.ones_like(scores, dtype=torch.bool) if valid_mask is None else valid_mask.bool()
    logits = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
    k = min(k, scores.shape[1])
    if training:
        soft = F.gumbel_softmax(logits, tau=temperature, hard=False, dim=-1)
        _, indices = torch.topk(soft, k, dim=-1)
    else:
        soft = logits.softmax(-1)
        _, indices = torch.topk(logits, k, dim=-1)
    soft = soft * valid_mask
    soft = soft / soft.sum(-1, keepdim=True).clamp_min(torch.finfo(soft.dtype).eps)
    hard = torch.zeros_like(scores).scatter_(1, indices, 1.0) * valid_mask
    return hard - soft.detach() + soft


class ImplicitGuidanceHead(nn.Module):
    def __init__(self, object_dim: int, hidden_dim: int, guide_tokens: int = 4, config: GuidedConfig = GuidedConfig()):
        super().__init__()
        self.config = config
        self.selector = nn.Sequential(nn.Linear(object_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.category = nn.Linear(object_dim, len(CATEGORIES))
        self.category_embedding = nn.Embedding(len(CATEGORIES), hidden_dim)
        self.guide = nn.Sequential(nn.Linear(object_dim + hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, guide_tokens * hidden_dim))
        self.guide_tokens = guide_tokens
        self.hidden_dim = hidden_dim

    def forward(
        self,
        objects: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        training: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        if training is None:
            training = self.training
        scores = self.selector(objects).squeeze(-1)
        selected = hard_select(scores, self.config.k, training, self.config.temperature, valid_mask)
        pooled = (objects * selected.unsqueeze(-1)).sum(1) / selected.sum(1, keepdim=True).clamp_min(1.0)
        if valid_mask is None:
            global_pooled = objects.mean(1)
        else:
            global_pooled = (objects * valid_mask.unsqueeze(-1)).sum(1) / valid_mask.sum(1, keepdim=True).clamp_min(1)
        category_logits = self.category(global_pooled + pooled)
        category_ids = category_logits.argmax(-1)
        category_vec = self.category_embedding(category_ids)
        guides = self.guide(torch.cat((pooled, category_vec), -1)).view(-1, self.guide_tokens, self.hidden_dim)
        return {"scores": scores, "selection": selected, "category_logits": category_logits, "category_ids": category_ids, "guides": guides}

    def losses(self, output: dict[str, torch.Tensor], category_targets: torch.Tensor, object_targets: torch.Tensor, question_ce: torch.Tensor) -> dict[str, torch.Tensor]:
        category_ce = F.cross_entropy(output["category_logits"], category_targets)
        object_bce = F.binary_cross_entropy_with_logits(output["scores"], object_targets.to(output["scores"].dtype))
        return {"loss": question_ce + category_ce + object_bce, "question": question_ce.detach(), "category": category_ce.detach(), "object": object_bce.detach()}


def category_id(name: str) -> int:
    return CATEGORY_TO_ID[name]


def inject_guides(inputs_embeds: torch.Tensor, guides: torch.Tensor, position: int = 0) -> torch.Tensor:
    return torch.cat((inputs_embeds[:, :position], guides, inputs_embeds[:, position:]), dim=1)
