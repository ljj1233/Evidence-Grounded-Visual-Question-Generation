"""On-policy wrapper around the frozen Flat verifier.

The prompt, parser, threshold and aggregation are the frozen Phase-2.1
implementation (`frozen_verifier`), used exactly as in
`scripts/vvqg/phase3a_flat_rollouts.py`; this only batches fresh rollouts.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from vvqg.rewards.support.frozen_verifier import frozen_prompts, parse_prediction

STATES = {
    "supported": "NO_CONFIRMED_VIOLATION",
    "unsupported": "CONFIRMED_VIOLATION",
    "uncertain": "UNCERTAIN",
}


class OnlineFlatVerifier:
    def __init__(self, model_path: str, device: Any, batch_size: int = 32, max_new_tokens: int = 256):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.device, self.batch_size, self.max_new_tokens = device, batch_size, max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.processor.tokenizer.padding_side = "left"
        self.model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, dtype=torch.bfloat16, device_map=str(device), local_files_only=True
            )
            .eval()
            .requires_grad_(False)
        )
        self.prompt = frozen_prompts()["flat"]

    def _image(self, path: str):
        from pathlib import Path

        from PIL import Image

        candidate = Path(path)
        root = Path(__file__).resolve().parents[4]
        with Image.open(candidate if candidate.is_absolute() else root / candidate) as source:
            return source.convert("RGB")

    @torch.inference_mode()
    def verify(self, rows: Sequence[dict[str, Any]]) -> list[str]:
        states: list[str] = []
        for start in range(0, len(rows), self.batch_size):
            chunk = rows[start : start + self.batch_size]
            texts = [
                self.processor.apply_chat_template(
                    [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.prompt + "\nQuestion: " + r["question"]}]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for r in chunk
            ]
            inputs = self.processor(
                text=texts, images=[self._image(r["image"]) for r in chunk], padding=True, return_tensors="pt"
            ).to(self.model.device)
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            raws = self.processor.tokenizer.batch_decode(
                generated[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
            )
            for raw in raws:
                parsed = parse_prediction(raw, "flat")
                state = (
                    STATES.get(parsed.get("overall_visual_support"), "FAILED")
                    if parsed.get("status") == "ok"
                    else "FAILED"
                )
                states.append(state)
        return states
