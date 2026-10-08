"""Qwen-facing B3 guide injection with differentiable guide slots."""

from __future__ import annotations

import torch
from torch import nn

from .implicit_guided.legacy import ImplicitGuidanceHead

GUIDE_TOKENS = ("<|vguide_1|>", "<|vguide_2|>", "<|cguide|>")


def register_guide_tokens(tokenizer, backbone: nn.Module) -> dict[str, int]:
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else getattr(tokenizer, "vocab", {})
    before = {token: vocab.get(token) for token in GUIDE_TOKENS}
    unknown = getattr(tokenizer, "unk_token_id", None)
    missing = [token for token, token_id in before.items() if token_id is None or token_id == unknown]
    if missing:
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
        backbone.resize_token_embeddings(len(tokenizer))
    return {name[2:-2]: tokenizer.convert_tokens_to_ids(name) for name in GUIDE_TOKENS}


class GuidedQwenAdapter(nn.Module):
    """Inject predicted guides through Qwen's normal token embedding lookup."""
    def __init__(self, backbone: nn.Module, object_dim: int, hidden_dim: int, guide_tokens: int = 4):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        # Three reserved slots are injected into the normal Qwen token stream:
        # two selected-region guides and one predicted-category guide.
        self.guidance = ImplicitGuidanceHead(object_dim, hidden_dim, guide_tokens=3)
        config = getattr(backbone, "config", None)
        text_config = getattr(config, "text_config", config)
        self.guide_projection = nn.Linear(hidden_dim, int(getattr(text_config, "hidden_size", hidden_dim)))

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def gradient_checkpointing_enable(self, **kwargs):
        return self.backbone.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        return self.backbone.gradient_checkpointing_disable()

    def enable_input_require_grads(self):
        return self.backbone.enable_input_require_grads()

    def save_baseline_pretrained(self, output):
        output = __import__("pathlib").Path(output); output.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(str(output / "backbone"))
        torch.save({"guidance": self.guidance.state_dict(), "guide_projection": self.guide_projection.state_dict()}, output / "guided_head.pt")

    def _guide_vectors(self, state: dict[str, torch.Tensor], guide_token_ids: dict[str, int]):
        guides = state["guides"]
        vectors = {}
        for name in guide_token_ids:
            if name.startswith("vguide_"):
                vectors[name] = self.guide_projection(guides[:, int(name.rsplit("_", 1)[-1]) - 1])
            elif name == "cguide":
                vectors[name] = self.guide_projection(guides[:, 2])
            else:
                raise KeyError(f"unknown B3 guide slot: {name}")
        return vectors

    def forward(self, objects: torch.Tensor, input_ids: torch.Tensor, guide_token_ids: dict[str, int], object_mask: torch.Tensor | None = None, **inputs):
        state = self.guidance(objects, valid_mask=object_mask)
        vectors = self._guide_vectors(state, guide_token_ids)
        embedding = self.backbone.get_input_embeddings()

        def replace_guides(_module, _args, output):
            result = output
            token_ids = _args[0]
            for name, token_id in guide_token_ids.items():
                result = torch.where(token_ids.eq(token_id).unsqueeze(-1), vectors[name].to(result.dtype).unsqueeze(1), result)
            return result

        hook = embedding.register_forward_hook(replace_guides)
        try:
            outputs = self.backbone(input_ids=input_ids, output_hidden_states=True, **inputs)
        finally:
            hook.remove()
        return {"outputs": outputs, "guidance": state, "guide_vectors": vectors, "hidden_states": outputs.hidden_states[-1] if getattr(outputs, "hidden_states", None) else None}

    @torch.inference_mode()
    def generate_with_guides(self, objects: torch.Tensor, input_ids: torch.Tensor, guide_token_ids: dict[str, int], object_mask: torch.Tensor | None = None, **inputs):
        """Generate through the backbone while replacing reserved guide slots."""
        state = self.guidance(objects, valid_mask=object_mask, training=False)
        vectors = self._guide_vectors(state, guide_token_ids)
        embedding = self.backbone.get_input_embeddings()

        def replace_guides(_module, _args, output):
            result = output
            token_ids = _args[0]
            for name, token_id in guide_token_ids.items():
                result = torch.where(token_ids.eq(token_id).unsqueeze(-1), vectors[name].to(result.dtype).unsqueeze(1), result)
            return result

        hook = embedding.register_forward_hook(replace_guides)
        try:
            return self.backbone.generate(input_ids=input_ids, **inputs)
        finally:
            hook.remove()


__all__ = ["GUIDE_TOKENS", "GuidedQwenAdapter", "register_guide_tokens"]
