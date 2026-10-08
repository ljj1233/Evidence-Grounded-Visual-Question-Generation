from __future__ import annotations

from vvqg.evidence_constrained.audit import (
    image_level_bootstrap,
    label_distribution,
    spearman,
    within_image_separation,
)


def _records():
    return [
        {"image_id": "A", "external_support": "supported", "v": 2.0},
        {"image_id": "A", "external_support": "supported", "v": 2.0},
        {"image_id": "A", "external_support": "unsupported", "v": 0.0},
        {"image_id": "B", "external_support": "supported", "v": 1.0},
        {"image_id": "B", "external_support": "unsupported", "v": 3.0},
        {"image_id": "B", "external_support": "uncertain", "v": 5.0},
    ]


def test_within_image_separation_equal_image_weight():
    stats = within_image_separation(_records(), signal=lambda row: row["v"])
    assert stats["mixed_label_images"] == 2
    assert stats["delta"]["mean"] == 0.0  # (2 + (-2)) / 2
    assert stats["concordance"]["mean"] == 0.5  # (1.0 + 0.0) / 2
    per_image = {row["image_id"]: row for row in stats["per_image"]}
    assert per_image["A"]["d_i"] == 2.0
    assert per_image["A"]["concordance"] == 1.0
    assert per_image["B"]["d_i"] == -2.0
    assert per_image["B"]["concordance"] == 0.0


def test_bootstrap_deterministic_and_brackets_mean():
    first = image_level_bootstrap([1.0, 2.0, 3.0, 4.0], n=500, seed=42)
    second = image_level_bootstrap([1.0, 2.0, 3.0, 4.0], n=500, seed=42)
    assert first == second
    assert first["lo"] <= first["mean"] <= first["hi"]


def test_label_distribution_counts():
    rows = label_distribution(_records(), signal=lambda row: row["v"])
    by_label = {row["label"]: row for row in rows}
    assert by_label["supported"]["n_candidates"] == 3
    assert by_label["unsupported"]["n_candidates"] == 2
    assert by_label["uncertain"]["n_candidates"] == 1


def test_spearman_monotone():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) > 0.99
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) < -0.99
