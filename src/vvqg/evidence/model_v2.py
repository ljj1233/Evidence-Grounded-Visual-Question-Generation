"""Joint Soft-Evidence QSE v2 — candidate-assigned soft visual slots.

Fixes the v1 slot collapse by (a) giving every slot its own query projection,
(b) supervising each slot against a distinct candidate question through a
candidate-local Hungarian assignment, and (c) never mean-pooling over slots.
The decoder keeps the original image tokens (bypass); the evidence path is added
as a small gated residual that has non-zero gradient from the first step.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from vvqg.evidence.assignment import assignment_loss, slot_diversity_loss, TEMPERATURE


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)


class SoftEvidenceQwenV2(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        n_slots: int = 4,
        slot_dim: int = 512,
        n_heads: int = 8,
        lambda_question: float = 1.0,
        lambda_assign: float = 0.20,
        lambda_swap: float = 0.05,
        lambda_slot: float = 0.01,
        temperature: float = TEMPERATURE,
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
        self.lambda_question = lambda_question
        self.lambda_assign = lambda_assign
        self.lambda_swap = lambda_swap
        self.lambda_slot = lambda_slot
        self.temperature = temperature

        self.img_proj = nn.Linear(hidden, slot_dim)
        self.slot_queries = nn.Parameter(torch.randn(n_slots, slot_dim) * 0.02)
        self.query_proj = nn.ModuleList([nn.Linear(slot_dim, slot_dim, bias=False) for _ in range(n_slots)])
        self.key_proj = nn.Linear(slot_dim, slot_dim, bias=False)
        self.value_proj = nn.Linear(slot_dim, slot_dim, bias=False)
        self.slot_norm = nn.LayerNorm(slot_dim)
        self.attn_scale = nn.Parameter(torch.tensor(math.sqrt(slot_dim)))  # unit-scale logits
        self.fuse_q = nn.Linear(hidden, slot_dim)
        self.fuse_attn = nn.MultiheadAttention(slot_dim, n_heads, batch_first=True)
        self.out_proj = nn.Linear(slot_dim, hidden)
        self.gate = nn.Parameter(torch.tensor([-4.595]))  # sigmoid ~= 0.01
        self.support_head = nn.Sequential(nn.Linear(slot_dim, slot_dim), nn.GELU(), nn.Linear(slot_dim, 1))
        nn.init.normal_(self.out_proj.weight, std=1e-4)
        nn.init.zeros_(self.out_proj.bias)

        self._grid: torch.Tensor | None = None
        self._evidence: torch.Tensor | None = None
        self._slot_conf: torch.Tensor | None = None

    # ---- passthroughs ----------------------------------------------------
    def gradient_checkpointing_enable(self, **kwargs):
        return self.backbone.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        return self.backbone.gradient_checkpointing_disable()

    def enable_input_require_grads(self):
        return self.backbone.enable_input_require_grads()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    # ---- evidence --------------------------------------------------------
    def _compute_evidence(self, per_image: list[torch.Tensor]):
        batch = len(per_image)
        evidence, confidence = [], []
        for tokens in per_image:
            projected = self.img_proj(tokens)
            keys = F.normalize(self.key_proj(projected), dim=-1).unsqueeze(0)  # [1,S,ds]
            values = self.value_proj(projected).unsqueeze(0)                    # [1,S,ds]
            queries = torch.stack([F.normalize(self.query_proj[i](self.slot_queries[i]), dim=-1)
                                   for i in range(self.n_slots)]).unsqueeze(0)  # [1,N,ds]
            logits = torch.matmul(queries, keys.transpose(1, 2)) * self.attn_scale
            weights = F.softmax(logits, dim=-1)               # [1,N,S]
            slots = torch.matmul(weights, values)[0]          # [N,ds]
            evidence.append(self.slot_norm(slots))
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
        candidate_emb = inputs.pop("candidate_emb", None)
        candidate_weight = inputs.pop("candidate_weight", None)
        candidate_mask = inputs.pop("candidate_mask", None)
        self._grid = inputs.get("image_grid_thw")
        self._evidence = None
        self._slot_conf = None
        handles = self._hooks()
        try:
            outputs = self.backbone(**inputs, output_hidden_states=True, return_dict=True)
        finally:
            self._remove(handles)
            self._grid = None
        loss_question = outputs.loss
        labels = inputs.get("labels")
        evidence = self._evidence
        result = {"loss": loss_question, "ce": loss_question.detach()}
        if labels is None or evidence is None:
            return result

        hidden = outputs.hidden_states[-1]
        q_repr = self.fuse_q(masked_mean(hidden, labels.ne(-100)).float())
        batch = q_repr.shape[0]

        loss_slot = slot_diversity_loss(evidence)
        result["slot"] = loss_slot.detach()
        with torch.no_grad():
            z = F.normalize(evidence.float(), dim=-1)
            sim = torch.einsum("bnd,bjd->bnj", z, z)
            off = 1 - torch.eye(self.n_slots, device=sim.device)
            result["slot_cosine"] = (sim * off).sum((1, 2)).mean() / (self.n_slots * (self.n_slots - 1))
            result["slot_std"] = evidence.float().std(dim=1).mean()

        loss_assign = torch.zeros((), device=evidence.device)
        result["assign"] = torch.zeros((), device=evidence.device)
        if candidate_emb is not None and candidate_weight is not None and candidate_mask is not None and batch > 1:
            loss_assign, stats = assignment_loss(
                evidence, candidate_emb.to(evidence.device), candidate_weight.to(evidence.device),
                candidate_mask.to(evidence.device), temperature=self.temperature,
            )
            result["assign"] = loss_assign.detach()
            result["assign_skipped"] = stats["assign_skipped"]
            result["assign_entropy"] = stats["assign_entropy"]
            result["matched_sim"] = stats["matched_sim"]
            result["shuffled_sim"] = stats["shuffled_sim"]
            result["assign_gap"] = stats["assign_gap"]
            result["slot_usage"] = stats["slot_usage"]

        # swap objective: per-image image-question matching, max over slots (no mean)
        e_image = evidence.float().max(dim=1).values
        pairs = q_repr.unsqueeze(1) * e_image.unsqueeze(0)
        scores = self.support_head(pairs).squeeze(-1)
        target = torch.eye(batch, device=scores.device)
        loss_swap = F.binary_cross_entropy_with_logits(scores, target)
        result["swap"] = loss_swap.detach()
        with torch.no_grad():
            result["swap_pos"] = scores.diagonal().mean()
            result["swap_neg"] = (scores * (1 - target)).sum() / max(batch * (batch - 1), 1)
            result["gate"] = torch.sigmoid(self.gate).detach()

        loss = (self.lambda_question * loss_question + self.lambda_assign * loss_assign
                + self.lambda_swap * loss_swap + self.lambda_slot * loss_slot)
        result["loss"] = loss
        return result

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
    _STATE_KEYS = ("img_proj", "slot_queries", "query_proj", "key_proj", "value_proj",
                   "slot_norm", "attn_scale", "fuse_q", "fuse_attn", "out_proj", "gate", "support_head")

    def evidence_state(self) -> dict:
        state = {}
        for key in self._STATE_KEYS:
            module = getattr(self, key)
            state[key] = module.state_dict() if hasattr(module, "state_dict") else module
        return state

    def save_pretrained(self, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(str(output / "backbone"))
        torch.save(self.evidence_state(), output / "evidence_head.pt")
        (output / "evidence_config.json").write_text(json.dumps({
            "n_slots": self.n_slots, "slot_dim": self.slot_dim,
            "lambda_question": self.lambda_question, "lambda_assign": self.lambda_assign,
            "lambda_swap": self.lambda_swap, "lambda_slot": self.lambda_slot,
            "temperature": self.temperature,
        }, indent=2) + "\n")

    @classmethod
    def from_pretrained(cls, backbone: nn.Module, checkpoint) -> "SoftEvidenceQwenV2":
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
