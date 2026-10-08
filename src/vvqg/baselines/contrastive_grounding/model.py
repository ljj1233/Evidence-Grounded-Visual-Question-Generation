"""Qwen3-VL wrapper for the ConVQG-inspired B2 objective."""

from __future__ import annotations

import torch
from torch import nn

from ..common import cosine_distance, masked_mean, normalize


class ContrastiveGroundingQwen(nn.Module):
    def __init__(self, backbone: nn.Module, embedding_dim: int, alpha: float = 0.2, margin: float = 0.5):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        hidden_dim = int(backbone.config.text_config.hidden_size)
        self.projection = nn.Linear(hidden_dim, int(embedding_dim))
        self.alpha = alpha
        self.margin = margin

    def gradient_checkpointing_enable(self, **kwargs):
        return self.backbone.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        return self.backbone.gradient_checkpointing_disable()

    def enable_input_require_grads(self):
        return self.backbone.enable_input_require_grads()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def save_baseline_pretrained(self, output):
        output = __import__("pathlib").Path(output)
        output.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(str(output / "backbone"))
        torch.save(self.projection.state_dict(), output / "contrastive_projection.pt")
        (output / "contrastive_config.json").write_text(
            __import__("json").dumps({"embedding_dim": self.projection.out_features, "alpha": self.alpha, "margin": self.margin}) + "\n",
            encoding="utf-8",
        )

    def forward(self, *, z_gt: torch.Tensor, z_img: torch.Tensor, z_txt: torch.Tensor, beta: float = 10.0, **inputs):
        outputs = self.backbone(**inputs, output_hidden_states=True, return_dict=True)
        labels = inputs["labels"]
        target_mask = labels.ne(-100)
        h_q = masked_mean(outputs.hidden_states[-1], target_mask)
        z_joint = normalize(self.projection(h_q))
        z_gt, z_img, z_txt = (z.to(device=z_joint.device, dtype=z_joint.dtype) for z in (z_gt, z_img, z_txt))
        d_gt = cosine_distance(z_joint, z_gt)
        cl_img = (d_gt - cosine_distance(z_joint, z_img) + self.margin).relu()
        cl_txt = (d_gt - cosine_distance(z_joint, z_txt) + self.margin).relu()
        cl = (1.0 - self.alpha) * cl_img.mean() + self.alpha * cl_txt.mean()
        loss = (outputs.loss + float(beta) * cl) / 2.0
        return {"loss": loss, "ce": outputs.loss.detach(), "cl": cl.detach(), "cl_img": cl_img.mean().detach(), "cl_txt": cl_txt.mean().detach(), "weighted_cl": (float(beta) * cl).detach(), "z_joint": z_joint}


def make_b2_messages(question: str, caption: str) -> list[dict]:
    return [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"Factual visible caption: {caption}\nGenerate a question grounded in this image and caption."}]},
        {"role": "assistant", "content": question},
    ]
