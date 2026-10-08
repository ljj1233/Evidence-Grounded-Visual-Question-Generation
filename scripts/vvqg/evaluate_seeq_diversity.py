#!/usr/bin/env python3
"""Evaluate SeeQ's four-question within-image embedding diversity protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


def last_token_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    lengths = mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda row: (str(row["image_id"]), int(row["generation_id"])))
    assert len(rows) == 4000
    assert all(int(row["generation_id"]) == i % 4 for i, row in enumerate(rows))
    return rows


def embed_questions(texts: list[str], model_path: Path, cache_path: Path, batch_size: int, label: str = "Semantic diversity") -> tuple[np.ndarray, bool]:
    keys = np.array([hashlib.sha256(text.encode()).hexdigest() for text in texts])
    if cache_path.is_file():
        cache = np.load(cache_path, allow_pickle=False)
        if np.array_equal(cache["keys"], keys):
            return cache["embeddings"], True
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    output = np.empty((len(texts), model.config.hidden_size), dtype=np.float32)
    bar = tqdm(total=len(texts), desc=label, unit="q", dynamic_ncols=True)
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True, max_length=8192, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            pooled = last_token_pool(model(**batch).last_hidden_state, batch["attention_mask"])
            output[start:start + len(batch["input_ids"])] = F.normalize(pooled, p=2, dim=1).float().cpu().numpy()
        bar.update(len(batch["input_ids"]))
    bar.close()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, keys=keys, embeddings=output)
    return output, False


def score(rows: list[dict], embeddings: np.ndarray) -> list[dict]:
    groups = np.arange(len(rows)).reshape(-1, 4)
    sims = np.einsum("nkd,nld->nkl", embeddings[groups], embeddings[groups])
    pair_distances = 1.0 - sims[:, (0, 0, 0, 1, 1, 2), (1, 2, 3, 2, 3, 3)]
    questions = np.array([row["question"].casefold().strip() for row in rows]).reshape(-1, 4)
    return [
        {
            "image_id": str(rows[i * 4]["image_id"]),
            "num_questions": 4,
            "unique_question_rate": float(len(set(questions[i])) / 4),
            "mean_pairwise_cosine_distance": float(pair_distances[i].mean()),
            "mean_pairwise_cosine_similarity": float(1.0 - pair_distances[i].mean()),
            "min_pairwise_cosine_distance": float(pair_distances[i].min()),
            "max_pairwise_cosine_distance": float(pair_distances[i].max()),
        }
        for i in range(len(groups))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--zero-shot", type=Path, required=True)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--label", default="Semantic diversity")
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = {"zero_shot": load_rows(args.zero_shot), "sft": load_rows(args.sft)}
    texts = [row["question"] for system in inputs.values() for row in system]
    embeddings, cache_hit = embed_questions(texts, args.model, args.output / "embedding_cache.npz", args.batch_size, label=args.label)
    summaries = {}
    offset = 0
    rng = np.random.default_rng(42)
    for system, rows in inputs.items():
        per_image = score(rows, embeddings[offset:offset + len(rows)])
        offset += len(rows)
        values = np.array([row["mean_pairwise_cosine_distance"] for row in per_image])
        bootstrap = values[rng.integers(0, len(values), (5000, len(values)))].mean(axis=1)
        summaries[system] = {"images": len(per_image), "questions": len(rows), "questions_per_image": 4, "mean": float(values.mean()), "std": float(values.std(ddof=1)), "bootstrap_95ci": [float(np.quantile(bootstrap, .025)), float(np.quantile(bootstrap, .975))], "mean_unique_question_rate": float(np.mean([row["unique_question_rate"] for row in per_image]))}
        (args.output / f"{system}_per_image.jsonl").write_text("\n".join(json.dumps(row) for row in per_image) + "\n")
    (args.output / "summary.json").write_text(json.dumps({"schema": "phase24-seeq-embedding-diversity-summary-v2", "embedding_model": str(args.model), "seed": 42, "systems": summaries}, indent=2) + "\n")
    (args.output / "runtime.json").write_text(json.dumps({"schema": "phase24-seeq-diversity-runtime-v2", "model_loads": int(not cache_hit), "batch_size": args.batch_size, "questions": len(texts), "wall_seconds": time.perf_counter() - started, "cache": str(args.output / "embedding_cache.npz"), "cache_hit": cache_hit}, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
