"""ConVQG-inspired contrastive grounding for Qwen3-VL."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .common import cosine_distance, masked_mean, normalize


@dataclass(frozen=True)
class ContrastiveConfig:
    alpha: float = 0.2
    margin: float = 0.5
    beta_start: float = 10.0
    beta_step: float = 10.0

    def beta(self, epoch: int) -> float:
        return self.beta_start + self.beta_step * epoch


class ContrastiveGroundingHead(nn.Module):
    """Project Qwen target-token states into the frozen embedding space."""

    def __init__(self, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, embedding_dim)

    def joint_embedding(self, hidden_states: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        return normalize(self.projection(masked_mean(hidden_states, target_mask)))

    def loss(
        self,
        hidden_states: torch.Tensor,
        target_mask: torch.Tensor,
        z_gt: torch.Tensor,
        z_img: torch.Tensor,
        z_txt: torch.Tensor,
        ce: torch.Tensor,
        epoch: int,
        config: ContrastiveConfig = ContrastiveConfig(),
    ) -> dict[str, torch.Tensor]:
        z_joint = self.joint_embedding(hidden_states, target_mask)
        cl_img = (cosine_distance(z_joint, z_gt) - cosine_distance(z_joint, z_img) + config.margin).relu().mean()
        cl_txt = (cosine_distance(z_joint, z_gt) - cosine_distance(z_joint, z_txt) + config.margin).relu().mean()
        cl = config.alpha * cl_txt + (1.0 - config.alpha) * cl_img
        beta = config.beta(epoch)
        total = (beta * cl + ce) / 2.0
        return {"loss": total, "ce": ce.detach(), "cl": cl.detach(), "weighted_cl": (beta * cl).detach(), "cl_img": cl_img.detach(), "cl_txt": cl_txt.detach(), "z_joint": z_joint}


def caption_prompt() -> str:
    return "Describe only factual, visibly supported content in this image. Do not speculate."


def text_only_question_prompt(caption: str) -> str:
    return f"Image description: {caption}\nGenerate one visual question about the described scene."
