"""Soft evidence branch over a shared Qwen3-VL backbone.

Evidence slots are produced by learned queries cross-attending to the shared
vision representation (the merged image tokens emitted by the vision tower) and
are fused into the text decoder by a gated cross-attention residual. The gate is
initialized at zero so E1 starts exactly at the control. The original image tokens
remain in the sequence, so the decoder always keeps a direct image bypass.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)


class SoftEvidenceQwen(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        n_slots: int = 4,
        slot_dim: int = 512,
        n_heads: int = 8,
        lambda_evidence: float = 0.10,
        lambda_swap: float = 0.10,
        lambda_slot: float = 0.01,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        base = backbone.get_base_model() if hasattr(backbone, "get_base_model") else backbone
        self._base = base
        self._visual = base.model.visual
        self._layers = list(base.model.language_model.layers)
        hidden = int(base.config.text_config.hidden_size)
        self.merge_size = int(getattr(base.model.visual, "spatial_merge_size", 2))
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.lambda_evidence = lambda_evidence
        self.lambda_swap = lambda_swap
        self.lambda_slot = lambda_slot
        self.temperature = temperature

        self.img_proj = nn.Linear(hidden, slot_dim)
        self.slot_queries = nn.Parameter(torch.randn(n_slots, slot_dim) * slot_dim ** -0.5)
        self.evidence_attn = nn.MultiheadAttention(slot_dim, n_heads, batch_first=True)
        self.fuse_q = nn.Linear(hidden, slot_dim)
        self.fuse_attn = nn.MultiheadAttention(slot_dim, n_heads, batch_first=True)
        self.out_proj = nn.Linear(slot_dim, hidden)
        self.gate = nn.Parameter(torch.zeros(1))
        self.support_head = nn.Sequential(nn.Linear(slot_dim, slot_dim), nn.GELU(), nn.Linear(slot_dim, 1))
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self._grid: torch.Tensor | None = None
        self._evidence: torch.Tensor | None = None
        self._slot_conf: torch.Tensor | None = None

    # ---- backbone passthroughs ------------------------------------------
    def gradient_checkpointing_enable(self, **kwargs):
        return self.backbone.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        return self.backbone.gradient_checkpointing_disable()

    def enable_input_require_grads(self):
        return self.backbone.enable_input_require_grads()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def parameters(self, recurse: bool = True):
        return super().parameters(recurse)

    # ---- evidence computation -------------------------------------------
    def _compute_evidence(self, per_image: list[torch.Tensor]):
        batch = len(per_image)
        queries = self.slot_queries.unsqueeze(0).expand(batch, -1, -1)
        evidence, confidence = [], []
        for index, tokens in enumerate(per_image):
            kv = self.img_proj(tokens).unsqueeze(0)
            out, weights = self.evidence_attn(
                queries[index : index + 1], kv, kv, need_weights=True, average_attn_weights=True
            )
            evidence.append(out[0])
            confidence.append(weights[0].amax(dim=-1))
        return torch.stack(evidence, 0), torch.stack(confidence, 0)

    def _capture_image(self, _module, _inputs, output):
        if self._grid is None:
            return output
        tokens = output.pooler_output
        if isinstance(tokens, (list, tuple)):
            tokens = torch.cat(list(tokens), dim=0)
        split = (self._grid.prod(-1) // (self.merge_size ** 2)).tolist()
        per_image = torch.split(tokens.to(self.img_proj.weight.dtype), split, dim=0)
        self._evidence, self._slot_conf = self._compute_evidence(list(per_image))
        return output

    def _inject(self, _module, _inputs, output):
        if self._evidence is None:
            return output
        hidden = output if torch.is_tensor(output) else output[0]
        if self._evidence.shape[0] != hidden.shape[0]:
            return output
        query = self.fuse_q(hidden.float())
        attn, _ = self.fuse_attn(query, self._evidence, self._evidence)
        fused = hidden + (torch.sigmoid(self.gate) * self.out_proj(attn)).to(hidden.dtype)
        return fused if torch.is_tensor(output) else (fused,) + tuple(output[1:])

    def _hooks(self):
        handles = [layer.register_forward_hook(self._inject) for layer in self._layers]
        handles.append(self._visual.register_forward_hook(self._capture_image))
        return handles

    @staticmethod
    def _remove(handles) -> None:
        for handle in handles:
            handle.remove()

    # ---- forward ---------------------------------------------------------
    def forward(self, **inputs):
        self._grid = inputs.get("image_grid_thw")
        self._evidence = None
        self._slot_conf = None
        handles = self._hooks()
        try:
            outputs = self.backbone(**inputs, output_hidden_states=True, return_dict=True)
        finally:
            self._remove(handles)
            self._grid = None
        loss_task = outputs.loss
        labels = inputs.get("labels")
        evidence = self._evidence
        if labels is None or evidence is None:
            return {"loss": loss_task, "ce": loss_task.detach()}
        hidden = outputs.hidden_states[-1]
        q_repr = self.fuse_q(masked_mean(hidden, labels.ne(-100)).float())  # [B, ds]
        e_repr = evidence.mean(1)  # [B, ds]
        batch = q_repr.shape[0]
        index = torch.arange(batch, device=q_repr.device)

        zq = F.normalize(q_repr, dim=-1)
        ze = F.normalize(e_repr, dim=-1)
        logits = zq @ ze.t() / self.temperature
        loss_align = 0.5 * (F.cross_entropy(logits, index) + F.cross_entropy(logits.t(), index))

        pairs = q_repr.unsqueeze(1) * e_repr.unsqueeze(0)  # [B, B, ds]
        scores = self.support_head(pairs).squeeze(-1)
        target = torch.eye(batch, device=scores.device)
        loss_swap = F.binary_cross_entropy_with_logits(scores, target)

        zslot = F.normalize(evidence, dim=-1)
        sim = (zslot @ zslot.transpose(1, 2)).clamp_min(0)
        confidence = self._slot_conf / self._slot_conf.sum(-1, keepdim=True).clamp_min(1e-6)
        weight = confidence.unsqueeze(2) * confidence.unsqueeze(1)
        offdiag = 1.0 - torch.eye(self.n_slots, device=sim.device)
        loss_slot = (sim * offdiag * weight).sum(dim=(1, 2)).mean()

        loss = loss_task + self.lambda_evidence * loss_align + self.lambda_swap * loss_swap + self.lambda_slot * loss_slot
        with torch.no_grad():
            pos = scores.diagonal().mean()
            neg = (scores * (1 - target)).sum() / max(batch * (batch - 1), 1)
        return {
            "loss": loss, "ce": loss_task.detach(), "align": loss_align.detach(),
            "swap": loss_swap.detach(), "slot": loss_slot.detach(),
            "swap_pos": pos, "swap_neg": neg, "gate": torch.sigmoid(self.gate).detach(),
        }

    # ---- generation ------------------------------------------------------
    def generate(self, **inputs):
        self._grid = inputs.get("image_grid_thw")
        self._evidence = None
        self._slot_conf = None
        handles = self._hooks()
        try:
            return self.backbone.generate(**inputs)
        finally:
            self._remove(handles)
            self._grid = None

    # ---- persistence -----------------------------------------------------
    def evidence_state(self) -> dict:
        keys = ("img_proj", "slot_queries", "evidence_attn", "fuse_q", "fuse_attn",
                "out_proj", "gate", "support_head")
        return {key: getattr(self, key).state_dict() if hasattr(getattr(self, key), "state_dict") else getattr(self, key)
                for key in keys}

    def save_pretrained(self, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(str(output / "backbone"))
        torch.save(self.evidence_state(), output / "evidence_head.pt")
        (output / "evidence_config.json").write_text(json.dumps({
            "n_slots": self.n_slots, "slot_dim": self.slot_dim,
            "lambda_evidence": self.lambda_evidence, "lambda_swap": self.lambda_swap,
            "lambda_slot": self.lambda_slot, "temperature": self.temperature,
        }, indent=2) + "\n")

    @classmethod
    def from_pretrained(cls, backbone: nn.Module, checkpoint) -> "SoftEvidenceQwen":
        checkpoint = Path(checkpoint)
        config = json.loads((checkpoint / "evidence_config.json").read_text())
        model = cls(backbone, **config)
        state = torch.load(checkpoint / "evidence_head.pt", map_location="cpu")
        for key, value in state.items():
            module = getattr(model, key)
            if hasattr(module, "load_state_dict"):
                module.load_state_dict(value)
            else:
                with torch.no_grad():
                    module.copy_(value)
        return model

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
