"""Pure statistics for the CER counterfactual-signal audit.

Image-level (not candidate-level) bootstrap is used everywhere so that images
with many candidates do not dominate.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Sequence

import numpy as np

POS = "supported"
NEG = "unsupported"


def image_level_bootstrap(values: Sequence[float], n: int = 5000, seed: int = 42) -> dict[str, float]:
    values = np.asarray([value for value in values if value is not None], dtype=float)
    if values.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, values.size, size=(n, values.size))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lo": float(np.percentile(draws, 2.5)),
        "hi": float(np.percentile(draws, 97.5)),
        "n": int(values.size),
    }


def _groups_by_image(records: Sequence[dict[str, Any]], label_key: str) -> dict[str, dict[str, list[float]]]:
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        label = record.get(label_key)
        if label in (POS, NEG):
            groups[str(record["image_id"])][label].append(record)
    return groups


def within_image_separation(
    records: Sequence[dict[str, Any]],
    signal: Callable[[dict[str, Any]], float],
    label_key: str = "external_support",
) -> dict[str, Any]:
    """Per-image supported-minus-unsupported mean difference and pairwise concordance."""
    groups = _groups_by_image(records, label_key)
    deltas, concordances, per_image = [], [], []
    for image_id, labels in sorted(groups.items()):
        positives, negatives = labels.get(POS, []), labels.get(NEG, [])
        if not positives or not negatives:
            continue
        pos_values = [signal(record) for record in positives]
        neg_values = [signal(record) for record in negatives]
        pos_values = [value for value in pos_values if value is not None]
        neg_values = [value for value in neg_values if value is not None]
        if not pos_values or not neg_values:
            continue
        mean_delta = float(np.mean(pos_values) - np.mean(neg_values))
        pairs = [1.0 if p > n else 0.5 if p == n else 0.0 for p in pos_values for n in neg_values]
        concordance = float(np.mean(pairs))
        deltas.append(mean_delta)
        concordances.append(concordance)
        per_image.append({"image_id": image_id, "d_i": mean_delta, "concordance": concordance,
                          "n_supported": len(pos_values), "n_unsupported": len(neg_values)})
    return {
        "mixed_label_images": len(deltas),
        "delta": image_level_bootstrap(deltas),
        "concordance": image_level_bootstrap(concordances),
        "per_image": per_image,
    }


def label_distribution(records: Sequence[dict[str, Any]], signal: Callable[[dict[str, Any]], float],
                       label_key: str = "external_support") -> list[dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_label[record.get(label_key)].append(record)
    rows = []
    for label in (POS, "uncertain", NEG, None):
        subset = by_label.get(label)
        if not subset:
            continue
        values = np.asarray([signal(record) for record in subset if signal(record) is not None], dtype=float)
        if values.size == 0:
            continue
        rows.append({
            "label": label,
            "n_candidates": len(subset),
            "n_images": len({str(record["image_id"]) for record in subset}),
            "gap_mean": float(values.mean()),
            "gap_std": float(values.std()),
            "gap_p10": float(np.percentile(values, 10)),
            "gap_p50": float(np.percentile(values, 50)),
            "gap_p90": float(np.percentile(values, 90)),
            "lp_plus_mean": float(np.mean([record.get("lp_plus", float("nan")) for record in subset])),
            "lp_minus_mean": float(np.mean([record.get("lp_minus", float("nan")) for record in subset])),
        })
    return rows


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    denominator = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / denominator) if denominator else float("nan")
