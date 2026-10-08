"""Frozen text-semantic alignment replacing the released GloVe lookup."""
from __future__ import annotations
import re
import torch
import torch.nn.functional as F

def content_tokens(text: str) -> list[str]:
    stop = {"a", "an", "the", "is", "are", "was", "were", "what", "which", "who", "of", "on", "in", "to", "and", "or", "does", "do"}
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in stop]

def semantic_object_target(labels: list[str], question: str, answer: str, embed) -> list[float]:
    terms = content_tokens(f"{question} {answer}")
    phrases = [" ".join(str(label).lower().split()) for label in labels]
    if not terms or not phrases:
        return [0.0] * len(labels)
    vectors = embed(phrases + terms)
    scores = vectors[:len(phrases)] @ vectors[len(phrases):].T
    k = min(3, len(labels))
    valid = torch.tensor([label != "<empty>" for label in labels])
    ranking = scores.mean(-1).masked_fill(~valid, -torch.inf)
    selected = ranking.topk(min(k, int(valid.sum()))).indices if valid.any() else torch.empty(0, dtype=torch.long)
    target = torch.zeros(len(labels), dtype=torch.float32)
    target[selected] = 1.0
    return target.tolist()

def make_embedder(model_path, device="cuda", batch_size=32):
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, padding_side="left")
    model = AutoModel.from_pretrained(str(model_path), local_files_only=True, dtype=torch.bfloat16).eval().to(device)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    def embed(texts):
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True, return_tensors="pt").to(device)
                hidden = model(**batch).last_hidden_state
                index = batch["attention_mask"].sum(-1) - 1
                pooled = hidden[torch.arange(hidden.shape[0], device=device), index]
                vectors.append(F.normalize(pooled.float(), dim=-1).cpu())
        return torch.cat(vectors)
    return embed

def semantic_object_targets_batch(items, embed):
    """Compute released-style object targets with deduplicated batched math."""
    prepared = []
    unique = []
    seen = set()
    for labels, question, answer in items:
        terms = content_tokens(f"{question} {answer}")
        phrases = [" ".join(str(label).lower().split()) for label in labels]
        prepared.append((phrases, terms))
        for text in phrases + terms:
            if text and text not in seen:
                seen.add(text); unique.append(text)
    vectors = embed(unique)
    lookup = {text: vectors[i] for i, text in enumerate(unique)}
    outputs = []
    for begin in range(0, len(prepared), 512):
        chunk = prepared[begin:begin + 512]
        max_terms = max((len(terms) for _, terms in chunk), default=0)
        if not max_terms:
            outputs.extend([[0.0] * len(phrases) for phrases, _ in chunk]); continue
        pv = torch.stack([torch.stack([lookup[x] for x in phrases]) for phrases, _ in chunk])
        tv = torch.zeros((len(chunk), max_terms, pv.shape[-1]), dtype=pv.dtype)
        tm = torch.zeros((len(chunk), max_terms), dtype=torch.bool)
        for i, (_, terms) in enumerate(chunk):
            if terms:
                tv[i, :len(terms)] = torch.stack([lookup[x] for x in terms]); tm[i, :len(terms)] = True
        scores = torch.einsum("bkd,bmd->bkm", pv, tv)
        means = scores.sum(-1) / tm.sum(-1).clamp_min(1).to(scores.dtype).unsqueeze(1)
        for i, (phrases, _) in enumerate(chunk):
            valid = torch.tensor([label != "<empty>" for label in phrases], dtype=torch.bool)
            ranking = means[i].masked_fill(~valid, -torch.inf)
            k = min(3, len(phrases))
            selected = ranking.topk(min(k, int(valid.sum()))).indices if valid.any() else torch.empty(0, dtype=torch.long)
            target = torch.zeros(len(phrases), dtype=torch.float32); target[selected] = 1.0
            outputs.append(target.tolist())
    return outputs
