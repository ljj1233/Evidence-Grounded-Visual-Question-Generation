"""B2 cache and model components."""

from .model import ContrastiveGroundingQwen

__all__ = ["ContrastiveGroundingQwen"]
"""Contrastive grounding baseline components."""

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .model import ContrastiveGroundingQwen


@dataclass(frozen=True)
class ContrastiveConfig:
    alpha: float = 0.2
    margin: float = 0.5


class ContrastiveGroundingHead(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int | ContrastiveConfig = 256,
                 config: ContrastiveConfig = ContrastiveConfig()):
        super().__init__()
        if isinstance(embedding_dim, ContrastiveConfig):
            config, embedding_dim = embedding_dim, 256
        self.config = config
        self.projection = nn.Linear(hidden_dim, int(embedding_dim))

    def forward(self, question, image, text):
        q = F.normalize(self.projection(question), dim=-1)
        i = F.normalize(self.projection(image), dim=-1)
        t = F.normalize(self.projection(text), dim=-1)
        positive = (q * i).sum(-1) + (q * t).sum(-1)
        negative = torch.roll((q * i).sum(-1), 1)
        loss = F.relu(self.config.margin - positive / 2 + negative).mean()
        return {"loss": loss, "positive": positive.mean(), "negative": negative.mean()}

    def loss(self, hidden, target_mask, z_gt, z_img, z_txt, beta, step=0):
        question = F.normalize(self.projection((hidden * target_mask.unsqueeze(-1)).sum(1) /
                                                target_mask.sum(1, keepdim=True).clamp_min(1)), dim=-1)
        z_gt, z_img, z_txt = (F.normalize(x.to(question), dim=-1) for x in (z_gt, z_img, z_txt))
        positive = ((question * z_img).sum(-1) + (question * z_txt).sum(-1)) / 2
        negative = torch.roll(positive, 1, 0)
        contrastive = F.relu(self.config.margin - positive + negative).mean()
        return {"loss": contrastive * self.config.alpha, "contrastive": contrastive.detach(),
                "positive": positive.mean().detach(), "negative": negative.mean().detach()}


__all__ = ["ContrastiveGroundingQwen", "ContrastiveGroundingHead", "ContrastiveConfig"]
