#!/usr/bin/env python3
"""Deduplicate and embed B2 question strings with Qwen3-Embedding-4B."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vvqg.paths import local_model_path
from vvqg.baselines.common import SEED, normalize, stable_hash
from vvqg.baselines.contrastive_grounding.data import text_hash

import torch


def last_token_pool(hidden, mask):
    lengths = mask.sum(-1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=ROOT / "outputs/same_backbone_baselines/manifests/train50k_records.jsonl")
    parser.add_argument("--image-cache", type=Path, default=ROOT / "outputs/same_backbone_baselines/b2_contrastive/cache")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/same_backbone_baselines/b2_contrastive/cache/embeddings.json")
    parser.add_argument("--model", type=Path, default=Path(local_model_path('Qwen3-Embedding-4B')).expanduser())
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--filter-image-cache", action="store_true", help="Embed only q_gt rows whose image is in --image-cache (smoke runs).")
    args = parser.parse_args()
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    image_ids = None
    if args.filter_image_cache:
        image_ids = {str(json.loads(line)["image_id"]) for line in (args.image_cache / "captions.jsonl").open(encoding="utf-8")}
    questions = set()
    for index, line in enumerate(args.records.open(encoding="utf-8")):
        if args.limit is not None and index >= args.limit:
            break
        row = json.loads(line)
        iid = str(Path(row["image"]).stem.rsplit("_", 1)[-1].lstrip("0") or "0")
        if image_ids is None or iid in image_ids:
            questions.add(row["question"])
    for name in ("q_img.jsonl", "q_txt.jsonl"):
        path = args.image_cache / name
        if path.is_file():
            for line in path.open(encoding="utf-8"):
                questions.add(json.loads(line)["question"])
    texts = sorted(questions)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, padding_side="left")
    model = AutoModel.from_pretrained(str(args.model), local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").eval().to(args.device)
    vectors = {}
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_size):
            batch = texts[start:start + args.batch_size]
            tokens = tokenizer(batch, padding=True, truncation=True, max_length=8192, return_tensors="pt").to(args.device)
            pooled = last_token_pool(model(**tokens).last_hidden_state, tokens["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=-1).float().cpu()
            vectors.update({text_hash(text): vec.tolist() for text, vec in zip(batch, pooled)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "b2-text-embedding-cache-v1", "seed": SEED, "model": str(args.model), "embedding_dim": len(next(iter(vectors.values()))), "vectors": vectors, "text_hashes": {text_hash(text): text for text in texts}, "seconds": time.perf_counter() - started, "source_hash": stable_hash(texts)}
    args.output.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("schema", "embedding_dim", "seconds", "source_hash")}, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
