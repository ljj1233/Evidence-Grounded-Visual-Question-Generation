#!/usr/bin/env python3
"""Build resumable, image-deduplicated B2 frozen-model caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vvqg.paths import local_model_path

from vvqg.baselines.common import SEED, stable_hash
from vvqg.baselines.contrastive_grounding.model import make_b2_messages

CAPTION_PROMPT = "Describe only directly visible content in this image. Be concise, factual, and do not speculate or use external knowledge."
QUESTION_PROMPT = "Generate one visual question about this image."
TEXT_QUESTION_PROMPT = "Generate one visual-style question that could be asked using only this textual description. Output only one concise question ending with a question mark."


def model_hash(path: Path) -> str:
    config = path / "config.json"
    return hashlib.sha256(config.read_bytes() + str(path).encode()).hexdigest()


def load_rows(path: Path, limit: int | None) -> list[dict]:
    rows = []
    seen = set()
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        image = row.get("image_path") or row.get("image")
        iid = str(row.get("image_id") or Path(image).stem.rsplit("_", 1)[-1].lstrip("0") or "0")
        if iid in seen:
            continue
        seen.add(iid)
        rows.append({"image_id": iid, "image": image})
    rows.sort(key=lambda row: int(row["image_id"]))
    return rows[:limit] if limit else rows


def read_cache(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    rows = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            rows[str(row["image_id"])] = row
    return rows


def write_cache(path: Path, rows: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in sorted(rows.values(), key=lambda item: int(item["image_id"])):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()] if path.is_file() else []


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def generate_image_batch(model, processor, rows: list[dict], prompt: str, max_new_tokens: int, device: str) -> list[str]:
    import torch
    from PIL import Image

    images = []
    for row in rows:
        with Image.open(ROOT / row["image"]) as source:
            images.append(source.convert("RGB"))
    messages = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}] for _ in rows]
    texts = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    prompt_len = inputs["input_ids"].shape[1]
    return [processor.tokenizer.decode(output[prompt_len:], skip_special_tokens=True).strip() for output in outputs]


def generate_text_batch(model, processor, captions: list[str], device: str, max_new_tokens: int) -> list[str]:
    import torch

    messages = [[{"role": "user", "content": [{"type": "text", "text": f"{TEXT_QUESTION_PROMPT}\nDescription: {caption}"}]}] for caption in captions]
    texts = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
    inputs = processor(text=texts, padding=True, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    prompt_len = inputs["input_ids"].shape[1]
    return [processor.tokenizer.decode(output[prompt_len:], skip_special_tokens=True).strip() for output in outputs]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/same_backbone_baselines/manifests/train50k_images.jsonl")
    parser.add_argument("--records", type=Path, help="Optional record manifest; deduplicate its image paths for a smoke cache.")
    parser.add_argument("--development-manifest", type=Path, default=ROOT / "outputs/same_backbone_baselines/manifests/development1000_images.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/same_backbone_baselines/b2_contrastive/cache")
    parser.add_argument("--model", type=Path, default=Path(local_model_path('Qwen3-VL-8B-Instruct')).expanduser())
    parser.add_argument("--stage", choices=("captions", "qimg", "qtxt", "all"), default="all")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    rows = load_rows(args.records or args.manifest, args.limit)
    if args.development_manifest.is_file():
        rows += [row for row in load_rows(args.development_manifest, args.limit) if row["image_id"] not in {r["image_id"] for r in rows}]
    rows.sort(key=lambda row: int(row["image_id"]))
    if args.limit:
        rows = rows[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    mhash = model_hash(args.model)
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True, use_fast=False)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(str(args.model), dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa").eval().to(args.device)
    started = time.perf_counter()

    captions = read_cache(args.output / "captions.jsonl")
    if args.stage in {"captions", "all"}:
        for start in range(0, len(rows), args.batch_size):
            batch = [row for row in rows[start:start + args.batch_size] if row["image_id"] not in captions]
            if batch:
                texts = generate_image_batch(model, processor, batch, CAPTION_PROMPT, args.max_new_tokens, args.device)
                for row, text in zip(batch, texts):
                    captions[row["image_id"]] = {"image_id": row["image_id"], "image": row["image"], "caption": text, "model": str(args.model), "model_hash": mhash, "prompt_hash": stable_hash(CAPTION_PROMPT), "seed": SEED}
                write_cache(args.output / "captions.jsonl", captions)

    qimg = read_cache(args.output / "q_img.jsonl")
    if args.stage in {"qimg", "all"}:
        for start in range(0, len(rows), args.batch_size):
            batch = [row for row in rows[start:start + args.batch_size] if row["image_id"] not in qimg]
            if batch:
                texts = generate_image_batch(model, processor, batch, QUESTION_PROMPT, args.max_new_tokens, args.device)
                for row, text in zip(batch, texts):
                    qimg[row["image_id"]] = {"image_id": row["image_id"], "image": row["image"], "question": text, "model": str(args.model), "model_hash": mhash, "prompt_hash": stable_hash(QUESTION_PROMPT), "seed": SEED}
                write_cache(args.output / "q_img.jsonl", qimg)

    qtxt_rows = read_rows(args.output / "q_txt.jsonl")
    qtxt = {str(row["cache_key"]): row for row in qtxt_rows}
    if args.stage in {"qtxt", "all"}:
        caption_rows = {}
        for row in rows:
            caption = captions[row["image_id"]]["caption"]
            key = stable_hash({"caption": caption, "model_hash": mhash, "prompt_hash": stable_hash(TEXT_QUESTION_PROMPT)})
            caption_rows.setdefault(key, (row, caption))
        pending = [(key, row, caption) for key, (row, caption) in caption_rows.items()]
        for start in range(0, len(pending), args.batch_size):
            batch = [(row, cap, key) for key, row, cap in pending[start:start + args.batch_size] if key not in qtxt]
            if batch:   
                texts = generate_text_batch(model, processor, [cap for _, cap, _ in batch], args.device, args.max_new_tokens)
                for (row, cap, key), text in zip(batch, texts):
                    qtxt[key] = {"image_id": row["image_id"], "question": text, "caption_hash": stable_hash(cap), "cache_key": key, "model": str(args.model), "model_hash": mhash, "prompt_hash": stable_hash(TEXT_QUESTION_PROMPT), "seed": SEED}
                write_rows(args.output / "q_txt.jsonl", sorted(qtxt.values(), key=lambda item: item["cache_key"]))
    runtime = {"seed": SEED, "images": len(rows), "captions": len(captions), "q_img": len(qimg), "q_txt": len(qtxt), "seconds": time.perf_counter() - started, "device": args.device, "model": str(args.model)}
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(runtime, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
