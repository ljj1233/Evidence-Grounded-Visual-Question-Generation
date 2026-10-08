"""Frozen latent coverage reward (Phase 5A).

``z_vis(I, q) = L2Norm(h(image, q) - h(q-only))`` measured with the frozen SFT
VLM (last hidden layer, mean over question content tokens).  A fixed K-means
bank over the *training* SFT greedy questions gives under-covered weights
``w_k = minmax(1 / sqrt(n_k + 1))`` and a P95 distance gate.  This is an
empirical image-conditioned representation difference, not a causal or
mutual-information claim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch

INSTRUCTION = "Generate a question about this image."
REPO_ROOT = Path(__file__).resolve().parents[4]
BANK_KEYS = ("centers", "counts", "weights", "d95", "k", "p5", "p95")


def load_image(path: str):
    from PIL import Image

    candidate = Path(path)
    with Image.open(candidate if candidate.is_absolute() else REPO_ROOT / candidate) as source:
        return source.convert("RGB")


def _messages(processor: Any, question: str, with_image: bool, instruction: str) -> str:
    content = [{"type": "image"}] if with_image else []
    content.append({"type": "text", "text": instruction})
    messages = [
        {"role": "user", "content": content},
        {"role": "assistant", "content": question},
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


@torch.inference_mode()
def _mean_question_hidden(
    model: Any, processor: Any, texts: Sequence[str], images: Sequence[Any] | None,
    question_ids: Sequence[list[int]], device: Any,
) -> torch.Tensor:
    if images is not None:
        encoded = processor(text=list(texts), images=list(images), padding=True, return_tensors="pt")
    else:
        encoded = processor(text=list(texts), padding=True, return_tensors="pt")
    batch = {key: (value.to(device) if hasattr(value, "to") else value) for key, value in encoded.items()}
    hidden = model(**batch, use_cache=False, output_hidden_states=True).hidden_states[-1]
    vectors = []
    for index, ids in enumerate(question_ids):
        length = int(encoded["attention_mask"][index].sum())
        sequence = encoded["input_ids"][index, :length].tolist()
        start = next(
            (j for j in range(len(sequence) - len(ids) + 1) if sequence[j : j + len(ids)] == ids), None
        )
        if start is None:
            raise RuntimeError("question token alignment mismatch")
        positions = torch.arange(start, start + len(ids), device=device)
        vectors.append(hidden[index, positions].float().mean(dim=0))
    del hidden, batch, encoded
    return torch.stack(vectors)


@torch.inference_mode()
def visual_difference_repr(
    model: Any, processor: Any, items: Sequence[tuple[str, str]], device: Any,
    batch_size: int = 8, instruction: str = INSTRUCTION,
) -> torch.Tensor:
    """Return L2-normalized ``h(image, q) - h(q-only)`` for each ``(image, question)``."""
    previous = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "right"
    outputs = []
    try:
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            images = [load_image(image) for image, _ in chunk]
            multimodal = [_messages(processor, q, True, instruction) for _, q in chunk]
            text_only = [_messages(processor, q, False, instruction) for _, q in chunk]
            question_ids = [processor.tokenizer.encode(q, add_special_tokens=False) for _, q in chunk]
            hidden_mm = _mean_question_hidden(model, processor, multimodal, images, question_ids, device)
            hidden_t = _mean_question_hidden(model, processor, text_only, None, question_ids, device)
            for image in images:
                image.close()
            difference = (hidden_mm - hidden_t).float()
            outputs.append(difference / difference.norm(dim=-1, keepdim=True).clamp_min(1e-8))
    finally:
        processor.tokenizer.padding_side = previous
    return torch.cat(outputs).cpu()


def kmeans_plusplus(x: torch.Tensor, k: int, generator: torch.Generator) -> torch.Tensor:
    first = torch.randint(len(x), (1,), generator=generator, device=x.device).item()
    centers = [x[first]]
    closest = (x - centers[0]).pow(2).sum(dim=1)
    for _ in range(k - 1):
        probability = closest / closest.sum().clamp_min(1e-12)
        index = torch.multinomial(probability, 1, generator=generator).item()
        centers.append(x[index])
        closest = torch.minimum(closest, (x - x[index]).pow(2).sum(dim=1))
    return torch.stack(centers)


def kmeans(x: torch.Tensor, k: int, seed: int = 42, iters: int = 100) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixed-seed Lloyd K-means; returns (centers, assignment, distance-to-center)."""
    x = x.float()
    generator = torch.Generator(device=x.device).manual_seed(seed)
    centers = kmeans_plusplus(x, k, generator)
    assignment = torch.zeros(len(x), dtype=torch.long, device=x.device)
    distances = torch.zeros(len(x), device=x.device)
    for _ in range(iters):
        distances, indices = torch.cdist(x, centers).min(dim=1)
        assignment = indices
        updated = torch.zeros_like(centers)
        counts = torch.zeros(k, device=x.device)
        updated.index_add_(0, assignment, x)
        counts.index_add_(0, assignment, torch.ones(len(x), device=x.device))
        empty = counts == 0
        updated = updated / counts.clamp_min(1.0)[:, None]
        centers = torch.where(empty[:, None], centers, updated)
    distances = torch.cdist(x, centers).min(dim=1).values
    return centers, assignment, distances


def coverage_weights(counts: torch.Tensor) -> torch.Tensor:
    weights = 1.0 / (counts.float() + 1.0).sqrt()
    span = (weights.max() - weights.min()).clamp_min(1e-12)
    return (weights - weights.min()) / span


def assign_clusters(z: torch.Tensor, centers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    distances = torch.cdist(z.float().cpu(), centers.float().cpu())
    distance, cluster = distances.min(dim=1)
    return cluster, distance


def save_coverage(
    out_dir: Path, centers: torch.Tensor, counts: torch.Tensor, weights: torch.Tensor,
    d95: float, p5: float, p95: float, extra: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np

    np.save(out_dir / "cluster_centers.npy", centers.float().cpu().numpy())
    (out_dir / "cluster_counts.json").write_text(
        json.dumps({str(k): int(v) for k, v in enumerate(counts.tolist())}, indent=2) + "\n", encoding="utf-8"
    )
    config = {
        "k": int(len(centers)),
        "weights": [float(v) for v in weights.tolist()],
        "d95": float(d95),
        "uvfc_p5": float(p5),
        "uvfc_p95": float(p95),
        **extra,
    }
    (out_dir / "cluster_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def load_coverage(coverage_dir: str | Path, device: Any) -> dict[str, Any]:
    import numpy as np

    coverage_dir = Path(coverage_dir)
    config = json.loads((coverage_dir / "cluster_config.json").read_text(encoding="utf-8"))
    counts = json.loads((coverage_dir / "cluster_counts.json").read_text(encoding="utf-8"))
    centers = torch.tensor(np.load(coverage_dir / "cluster_centers.npy"), dtype=torch.float32)
    return {
        "centers": centers.to(device),
        "counts": torch.tensor([counts[str(k)] for k in range(int(config["k"]))], dtype=torch.long),
        "weights": torch.tensor(config["weights"], dtype=torch.float32, device=device),
        "d95": float(config["d95"]),
        "k": int(config["k"]),
        "p5": float(config["uvfc_p5"]),
        "p95": float(config["uvfc_p95"]),
    }


@torch.inference_mode()
def coverage_scores(
    model: Any, processor: Any, rows: Sequence[dict[str, Any]], coverage: dict[str, Any],
    device: Any, batch_size: int = 8,
) -> list[dict[str, Any]]:
    """Attach coverage cluster/weight/distance to each rollout row."""
    items = [(row["image"], row["question"]) for row in rows]
    z = visual_difference_repr(model, processor, items, device, batch_size=batch_size)
    cluster, distance = assign_clusters(z, coverage["centers"].cpu())
    raw = coverage["weights"].cpu()[cluster]
    gated = torch.where(distance <= coverage["d95"], raw, torch.zeros_like(raw))
    occupancy = coverage.get("counts")
    return [
        {
            "coverage_cluster": int(cluster[i]),
            "cluster_occupancy": int(occupancy[cluster[i]]) if occupancy is not None else None,
            "coverage_distance": float(distance[i]),
            "coverage_reward_raw": float(raw[i]),
            "coverage_reward_gated": float(gated[i]),
            "coverage_tail": bool(distance[i] > coverage["d95"]),
        }
        for i in range(len(rows))
    ]
