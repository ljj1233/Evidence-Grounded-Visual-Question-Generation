"""Lexical/semantic diversity metrics shared by the SeeQ-faithful preflight and tables."""
from __future__ import annotations

from vvqg.paths import local_model_path

import re
from collections import defaultdict

import numpy as np


def norm(q: str) -> str:
    return re.sub(r"\s+", " ", str(q).strip().casefold())


def tokenize(q: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", str(q).casefold())


def uniq_rate(questions) -> float:
    qs = [norm(q) for q in questions]
    return len(set(qs)) / len(qs) if qs else 0.0


def distinct_2(questions) -> float:
    bigrams, total = set(), 0
    for q in questions:
        toks = tokenize(q)
        for a, b in zip(toks, toks[1:]):
            bigrams.add((a, b)); total += 1
    return len(bigrams) / total if total else 0.0


def per_image_questions(rows: list[dict], key: str) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        groups[str(row["image_id"])].append(row[key])
    return groups


def lexical_metrics(rows: list[dict], key: str) -> dict:
    questions = [r[key] for r in rows]
    groups = per_image_questions(rows, key)
    within = [len({norm(q) for q in qs}) for qs in groups.values()]
    lengths = [len(tokenize(q)) for q in questions]
    return {
        "n": len(questions),
        "images": len(groups),
        "normalized_unique_rate": round(uniq_rate(questions), 4),
        "distinct_2": round(distinct_2(questions), 4),
        "normalized_duplicate_rate": round(1.0 - uniq_rate(questions), 4),
        "within_image_unique_mean": round(float(np.mean(within)), 3) if within else 0.0,
        "within_image_unique_rate": round(float(np.mean([u / len(groups[i]) for i, u in zip(groups, within)])), 4) if within else 0.0,
        "mean_length": round(float(np.mean(lengths)), 2) if lengths else 0.0,
    }


def semantic_diversity(groups: dict[str, list[str]], embed_fn) -> float:
    texts = [q for g in groups.values() for q in g]
    if not texts:
        return 0.0
    embeddings, _ = embed_fn(texts)
    distances, offset = [], 0
    for g in groups.values():
        block = embeddings[offset:offset + len(g)]; offset += len(g)
        if len(g) > 1:
            distances.append(float((1.0 - (block @ block.T)[np.triu_indices(len(g), 1)]).mean()))
    return round(float(np.mean(distances)), 4) if distances else 0.0


def load_embedder(model_path: str = local_model_path('Qwen3-Embedding-4B')):
    from pathlib import Path

    import torch
    from tqdm.auto import tqdm
    from transformers import AutoModel, AutoTokenizer

    path = Path(model_path).expanduser()
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
    model = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto").eval()

    def embed(texts: list[str], label: str = "SemDiv") -> tuple[np.ndarray, list[str]]:
        out = np.empty((len(texts), model.config.hidden_size), dtype=np.float32)
        with torch.inference_mode():
            for start in tqdm(range(0, len(texts), 64), desc=label, unit="batch"):
                batch = tokenizer(texts[start:start + 64], padding=True, truncation=True, max_length=512, return_tensors="pt").to(model.device)
                lengths = batch["attention_mask"].sum(dim=1) - 1
                last = model(**batch).last_hidden_state[torch.arange(batch["input_ids"].shape[0], device=model.device), lengths]
                out[start:start + len(batch["input_ids"])] = torch.nn.functional.normalize(last, p=2, dim=1).float().cpu().numpy()
        return out, texts

    return embed
