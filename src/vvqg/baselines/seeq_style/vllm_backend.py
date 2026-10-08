"""vLLM inference backend for resumable SeeQ generation.

Uses vLLM multimodal prefix caching so the same image vision features can be
reused across answer, rewrite, and filter stages. LoRA is applied to the
language layers; this project's adapter contains language-layer targets only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .pipeline import ANSWER_PROMPT, FILTER_PROMPT, PROPOSAL_PROMPT, REWRITE_PROMPT


class VLLMQwen3VLBackend:
    def __init__(
        self,
        *,
        base_model: str | Path,
        adapter: str | Path,
        reference_adapter: str | Path | None = None,
        batch_size: int = 64,
        gpu_memory_utilization: float = 0.7,
        max_model_len: int = 4096,
        min_pixels: int = 4096,
        max_pixels: int = 147456,
    ) -> None:
        import os
        import torch

        # The container has an 8-core CPU quota; torch's default thread pool
        # oversubscribes it and cgroup throttling makes CPU multimodal
        # preprocessing ~10x slower and erratic.  Pin CPU threads to 1.
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        torch.set_num_threads(1)

        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        self.batch_size = int(batch_size)
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        self._image_store: dict[str, Any] = {}
        self._SamplingParams = SamplingParams
        self.processor = AutoProcessor.from_pretrained(
            str(base_model), local_files_only=True, use_fast=False
        )
        self.llm = LLM(
            model=str(base_model),
            dtype="bfloat16",
            gpu_memory_utilization=float(gpu_memory_utilization),
            max_model_len=int(max_model_len),
            limit_mm_per_prompt={"image": 1},
            enable_lora=True,
            max_lora_rank=16,
            mm_processor_kwargs={
                "min_pixels": int(min_pixels),
                "max_pixels": int(max_pixels),
            },
        )
        self._proposer_lora = LoRARequest("seeq-proposer", 1, str(adapter))
        ref = str(reference_adapter or adapter)
        self._reference_lora = (
            self._proposer_lora
            if ref == str(adapter)
            else LoRARequest("seeq-reference", 2, ref)
        )

    def _image(self, value: Any) -> Any:
        if not isinstance(value, (str, Path)):
            return value
        from PIL import Image
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
        with Image.open(value) as image:
            image = image.convert("RGB").copy()
        h, w = smart_resize(
            image.height, image.width, min_pixels=self.min_pixels, max_pixels=self.max_pixels
        )
        if (h, w) != (image.height, image.width):
            image = image.resize((w, h), Image.Resampling.BICUBIC)
        return image

    def _requests(self, prompts: Sequence[str], images: Sequence[Any] | None) -> list[dict]:
        import hashlib
        requests = []
        image_cache = self._image_store
        for index, prompt in enumerate(prompts):
            image = None if images is None else images[index]
            if image is None:
                text = self.processor.apply_chat_template(
                    [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                requests.append({"prompt": text})
                continue
            key = str(image) if isinstance(image, (str, Path)) else id(image)
            if key not in image_cache:
                image_cache[key] = self._image(image)
            image_value = image_cache[key]
            text = self.processor.apply_chat_template(
                [{"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": prompt}
                ]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            stable_uuid = hashlib.sha256(str(image).encode("utf-8")).hexdigest()
            requests.append({
                "prompt": text,
                "multi_modal_data": {"image": image_value},
                "multi_modal_uuids": {"image": [stable_uuid]},
            })
        return requests

    def generate_role(self, prompts, images=None, *, reference=False, do_sample=False,
                      temperature=0.0, top_p=1.0, top_k=-1, max_new_tokens=256):
        sampling = self._SamplingParams(
            temperature=(float(temperature) if do_sample else 0.0),
            top_p=float(top_p), top_k=int(top_k), max_tokens=int(max_new_tokens),
        )
        lora = self._reference_lora if reference else self._proposer_lora
        outputs: list[str] = []
        for start in range(0, len(prompts), self.batch_size):
            chunk = prompts[start:start + self.batch_size]
            image_chunk = None if images is None else images[start:start + self.batch_size]
            result = self.llm.generate(self._requests(chunk, image_chunk), sampling, lora_request=lora, use_tqdm=False)
            outputs.extend(str(item.outputs[0].text).strip() for item in result)
        return outputs

    def _generate(
        self,
        prompts: Sequence[str],
        images: Sequence[Any],
        *,
        max_new_tokens: int,
        reference: bool = False,
    ) -> list[str]:
        sampling = self._SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        lora = self._reference_lora if reference else self._proposer_lora
        outputs: list[str] = []
        for start in range(0, len(prompts), self.batch_size):
            req = self._requests(
                prompts[start:start + self.batch_size],
                images[start:start + self.batch_size],
            )
            result = self.llm.generate(req, sampling, lora_request=lora, use_tqdm=False)
            outputs.extend(str(item.outputs[0].text).strip() for item in result)
        return outputs

    def propose(self, images, tasks):
        return [
            _json_output(x)
            for x in self._generate(
                [PROPOSAL_PROMPT + " Task family: " + str(t) for t in tasks],
                images, max_new_tokens=256,
            )
        ]

    def answer(self, images, questions):
        return self._generate(
            [ANSWER_PROMPT + " Question: " + str(q) for q in questions],
            images, max_new_tokens=128,
        )

    def rewrite(self, images, questions, answers):
        return [
            _rewrite_output(x)
            for x in self._generate(
                [REWRITE_PROMPT + f"\nQuestion: {q}\nAnswer: {a}"
                 for q, a in zip(questions, answers)],
                images, max_new_tokens=256, reference=True,
            )
        ]

    def filter(self, images, questions, answers):
        return [
            _json_output(x)
            for x in self._generate(
                [FILTER_PROMPT + f"\nQuestion: {q}\nAnswer: {a}"
                 for q, a in zip(questions, answers)],
                images, max_new_tokens=512, reference=True,
            )
        ]


def _json_output(text: Any) -> Any:
    import ast
    import json
    import re
    if isinstance(text, (dict, list)):
        return text
    match = re.search(r"\{.*\}", str(text), re.S)
    if not match:
        return {"text": str(text).strip()}
    payload = match.group(0)
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(payload)
        except (SyntaxError, ValueError):
            # Qwen sometimes emits a Python-style dict whose single-quoted
            # value contains an apostrophe (for example, "man's"). Extract
            # the requested field without treating that apostrophe as syntax.
            for key in ("Question", "Rewritten question"):
                match = re.search(
                    rf"[\"']{re.escape(key)}[\"']\s*:\s*(['\"])(.*?)\1\s*(?:,\s*[\"']|}})",
                    payload,
                    re.S,
                )
                if match:
                    return {key: match.group(2).strip()}
            return {"text": str(text).strip()}


def _rewrite_output(text: Any) -> Any:
    """Normalize the rewrite role: a bare question is the rewritten question."""
    out = _json_output(text)
    if isinstance(out, dict) and out.keys() == {"text"}:
        return {"Rewritten question": out["text"]}
    return out
