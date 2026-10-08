"""Frozen answer-token likelihood scorer for visual-information diagnostics."""
from __future__ import annotations
import hashlib
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
PROMPT = "Answer the question with a short answer only.\nQuestion: {question}"
VERSION = "vic-scorer-v1"

def cache_key(item: dict[str, Any], model_path: str, prompt: str = PROMPT) -> str:
    payload = {"version": VERSION, "model": str(model_path), "condition": item["condition"], "image": item.get("image"), "question": item["question"], "answer": item["answer"], "prompt": prompt}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def answer_logprobs(logits: torch.Tensor, input_ids: torch.Tensor, counts: torch.Tensor):
    width = int(counts.max())
    mask = torch.arange(width, device=counts.device)[None, :] >= width - counts[:, None]
    targets = input_ids[:, -width:]
    token_lp = logits[:, -width - 1:-1].float().log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (token_lp * mask).sum(-1) / counts, token_lp, mask

class VisualInformationScorer:
    """One frozen model with cached image features and tokenized prompts."""
    def __init__(self, model_path: str | Path, root: str | Path, *, device: str = "cuda:0", cache_size: int = 128):
        torch.set_num_threads(4)
        torch.manual_seed(42)
        self.model_path, self.root, self.device = str(model_path), Path(root), torch.device(device)
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True, use_fast=False)
        self.processor.tokenizer.padding_side = "left"
        start = time.perf_counter()
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_path, dtype=torch.bfloat16, device_map=str(self.device), local_files_only=True, attn_implementation="sdpa").eval().requires_grad_(False)
        if self.device.type == "cuda": torch.cuda.synchronize(self.device)
        self.load_seconds = time.perf_counter() - start
        self.visual_cache: OrderedDict[tuple[str, str], tuple[torch.Tensor, list[int], tuple[int, int]]] = OrderedDict()
        self.token_cache: dict[str, list[int]] = {}
        self.cache_size = cache_size
        self.runtime = {c: {"seconds": 0.0, "items": 0} for c in ("real", "text", "blank")}
        self.preprocessing_seconds = self.preprocessing_items = self.preprocessing_cache_hits = 0.0
        if self.device.type == "cuda": torch.cuda.reset_peak_memory_stats(self.device)

    def prompt(self, question: str, multimodal: bool) -> str:
        content = ([{"type": "image"}] if multimodal else []) + [{"type": "text", "text": PROMPT.format(question=question)}]
        return self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)

    def tokens(self, text: str) -> list[int]:
        if text not in self.token_cache: self.token_cache[text] = self.processor.tokenizer.encode(text, add_special_tokens=False)
        return self.token_cache[text]

    @torch.inference_mode()
    def visuals(self, items: list[dict[str, Any]]):
        keys, images, item_keys = [], [], []
        start = time.perf_counter()
        for item in items:
            key = (item["condition"], item["image"]); item_keys.append(key)
            if key in self.visual_cache or key in keys:
                self.preprocessing_cache_hits += 1; continue
            with Image.open(self.root / item["image"]) as im: image = im.convert("RGB")
            keys.append(key); images.append(Image.new("RGB", image.size, (127, 127, 127)) if key[0] == "blank" else image)
        if images:
            encoded = self.processor.image_processor(images=images, return_tensors="pt").to(self.device)
            features = self.model.get_image_features(**encoded).pooler_output
            grids = encoded["image_grid_thw"].cpu().tolist()
            for key, feat, grid, image in zip(keys, features, grids, images): self.visual_cache[key] = (feat, grid, image.size)
            self.preprocessing_items += len(images)
        self.preprocessing_seconds += time.perf_counter() - start
        result = [self.visual_cache[key] for key in item_keys]
        for key in item_keys: self.visual_cache.move_to_end(key)
        while len(self.visual_cache) > self.cache_size: self.visual_cache.popitem(last=False)
        return result

    @torch.inference_mode()
    def score_batch(self, items: list[dict[str, Any]], audit: bool = False) -> list[dict[str, Any]]:
        if not items: return []
        start = time.perf_counter(); multimodal = items[0]["condition"] != "text"
        assert all((r["condition"] != "text") == multimodal for r in items)
        prompts = [self.prompt(r["question"], multimodal) for r in items]; visual = self.visuals(items) if multimodal else None
        if visual:
            merge = self.processor.image_processor.merge_size ** 2
            prompts = [p.replace(self.processor.image_token, self.processor.image_token * (g[0] * g[1] * g[2] // merge)) for p, (_, g, _) in zip(prompts, visual)]
        prompt_ids, answer_ids = [self.tokens(p) for p in prompts], [self.tokens(r["answer"]) for r in items]
        assert all(answer_ids), "Empty answer token sequence"
        if audit: assert all(self.tokens(p + r["answer"]) == ids + ans for p, r, ids, ans in zip(prompts, items, prompt_ids, answer_ids))
        full_ids = [p + a for p, a in zip(prompt_ids, answer_ids)]
        full = self.processor.tokenizer.pad({"input_ids": full_ids}, padding=True, return_tensors="pt").to(self.device)
        counts = torch.tensor([len(a) for a in answer_ids], device=self.device); embeddings = self.model.get_input_embeddings()(full.input_ids); kwargs = dict(full)
        if visual:
            mask = (full.input_ids == self.model.config.image_token_id).unsqueeze(-1); embeddings = embeddings.masked_scatter(mask, torch.cat([f for f, _, _ in visual])); kwargs["image_grid_thw"] = torch.tensor([g for _, g, _ in visual], device=self.device); kwargs["mm_token_type_ids"] = mask.squeeze(-1).long()
        else:
            pos = full.attention_mask.cumsum(-1) - 1; kwargs["position_ids"] = pos.masked_fill(full.attention_mask == 0, 0).unsqueeze(0).expand(3, -1, -1)
        logits = self.model(**kwargs, inputs_embeds=embeddings, use_cache=False, logits_to_keep=int(counts.max()) + 1).logits
        means, token_lp, mask = answer_logprobs(logits, full.input_ids, counts)
        if self.device.type == "cuda": torch.cuda.synchronize(self.device)
        condition = items[0]["condition"]; self.runtime[condition]["seconds"] += time.perf_counter() - start; self.runtime[condition]["items"] += len(items)
        means, token_lp = means.cpu().tolist(), token_lp.cpu().tolist(); out = []
        for i, (row, mean, prompt, ans) in enumerate(zip(items, means, prompt_ids, answer_ids)):
            record = {**row, "L": mean, "answer_token_count": len(ans), "cache_key": cache_key(row, self.model_path)}
            if audit: record.update(prompt_text=prompts[i], prompt_token_count=len(prompt), answer_token_ids=ans, full_token_ids=prompt + ans, answer_mask=[0] * len(prompt) + [1] * len(ans), answer_token_logprobs=token_lp[i][-len(ans):], padding_offset=full.input_ids.shape[1] - len(prompt) - len(ans), scored_unpadded_positions=list(range(len(prompt) - 1, len(prompt) + len(ans) - 1)), image_grid_thw=visual[i][1] if visual else None, image_size=visual[i][2] if visual else None, independent_processor_max_error=None)
            out.append(record)
        return out

    def score_many(self, items: Iterable[dict[str, Any]], batch_size: int = 16, audit: bool = False) -> list[dict[str, Any]]:
        rows, result = list(items), []
        for condition in ("real", "text", "blank"):
            group, pos = [r for r in rows if r["condition"] == condition], 0
            while pos < len(group):
                size = min(batch_size, len(group) - pos)
                while True:
                    try: result.extend(self.score_batch(group[pos:pos + size], audit=audit)); break
                    except torch.cuda.OutOfMemoryError:
                        if size == 1: raise
                        size //= 2
                        if self.device.type == "cuda": torch.cuda.empty_cache()
                pos += size
        index = {cache_key(r, self.model_path): i for i, r in enumerate(rows)}; result.sort(key=lambda r: index[r["cache_key"]]); return result

    def report(self, batch_size: int) -> dict[str, Any]:
        conditions = {k: {**v, "sec_per_likelihood": v["seconds"] / v["items"] if v["items"] else None} for k, v in self.runtime.items()}
        report = {"version": VERSION, "model": self.model_path, "batch_size": batch_size, "conditions": conditions, "preprocessing_seconds": self.preprocessing_seconds, "preprocessing_items": self.preprocessing_items, "preprocessing_cache_hits": self.preprocessing_cache_hits, "model_load_seconds": self.load_seconds, "dtype": "bfloat16", "backend": "pil", "attention": "sdpa"}
        if self.device.type == "cuda": report.update(peak_allocated_gib=torch.cuda.max_memory_allocated(self.device) / 2**30, peak_reserved_gib=torch.cuda.max_memory_reserved(self.device) / 2**30, gpu=torch.cuda.get_device_name(self.device))
        return report

TeacherForcedScorer = VisualInformationScorer
