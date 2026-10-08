"""CLIPScore — **raw-question adaptation** (DIAGNOSTIC only).

This is NOT an exact reproduction of Hessel et al. 2021. The reference
implementation prepends a text prefix such as ``"A photo depicts"`` before
encoding; this implementation encodes the question text **as-is**. Everything
else follows the same recipe:

- backbone ``openai/clip-vit-base-patch32`` (local cache used when present),
- L2-normalized image and text features,
- cosine similarity, ``2.5 * max(cos, 0)``,
- no use of the CLIP ``logit_scale``.

Because the text protocol differs, active reports must call this
``raw-question CLIPScore adaptation`` and must not present it as the original
CLIPScore. Image features are cached by image path (in-memory) with the
backbone fixed by the resolved checkpoint path.
"""

from __future__ import annotations

from vvqg.paths import local_model_path

import os
from pathlib import Path
from typing import Optional, Sequence

import torch

BACKBONE = "openai/clip-vit-base-patch32"
LOCAL_BACKBONE = os.path.expanduser(local_model_path('clip-vit-base-patch32'))
SCALE = 2.5
TEXT_PROTOCOL = "raw_question_no_prefix"


def resolve_backbone(backbone: Optional[str] = None) -> str:
    if backbone:
        return backbone
    if Path(LOCAL_BACKBONE).is_dir():
        return LOCAL_BACKBONE
    return BACKBONE


class ClipScorer:
    name = "clipscore_raw_question"

    def __init__(
        self,
        backbone: Optional[str] = None,
        device: str = "cpu",
        batch_size: int = 64,
        local_files_only: bool = True,
    ) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.batch_size = batch_size
        self.backbone = resolve_backbone(backbone)
        self.model = CLIPModel.from_pretrained(self.backbone, local_files_only=local_files_only).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(self.backbone, local_files_only=local_files_only)
        self._image_cache: dict[str, torch.Tensor] = {}

    def image_embedding(self, image_path: str) -> torch.Tensor:
        key = str(image_path)
        if key in self._image_cache:
            return self._image_cache[key]
        from PIL import Image

        with Image.open(image_path) as source:
            image = source.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            embedding = self.model.get_image_features(**inputs)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        self._image_cache[key] = embedding
        return embedding

    def text_embeddings(self, texts: Sequence[str]) -> torch.Tensor:
        chunks = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.inference_mode():
                embedding = self.model.get_text_features(**inputs)
            chunks.append(embedding / embedding.norm(dim=-1, keepdim=True))
        return torch.cat(chunks, dim=0)

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """pairs: list of (image_path, question)."""
        images = torch.cat([self.image_embedding(path) for path, _ in pairs], dim=0)
        texts = self.text_embeddings([question for _, question in pairs])
        cosine = (images * texts).sum(dim=-1)
        return (SCALE * torch.clamp(cosine, min=0.0)).tolist()


def clipscore_from_embeddings(image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
    """CLIPScore from L2-normalized embeddings: ``2.5 * max(cos, 0)``."""
    cosine = (image_embeddings * text_embeddings).sum(dim=-1)
    return SCALE * torch.clamp(cosine, min=0.0)
