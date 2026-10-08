"""Active evaluation metric tests: golden fixtures + differential checks.

CPU-only, no API, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vvqg.evaluation import metrics  # noqa: E402
from vvqg.evaluation.audit import (  # noqa: E402
    K4_FIXTURE_CASES,
    K4_FIXTURE_EXPECTED,
    coco_caption_bleu4,
    k4_rates,
)
from vvqg.evaluation.metrics import (  # noqa: E402
    corpus_bleu4,
    distinct_2,
    duplicate_rate,
    normalized_excess_duplicate_rate,
    qbleu4_variant,
    question_type,
    repeated_group_membership_rate,
)


def test_k4_golden_fixture_matches_hand_computed_values():
    got = k4_rates(K4_FIXTURE_CASES)
    for key, expected in K4_FIXTURE_EXPECTED.items():
        assert got[key] == pytest.approx(expected, abs=1e-12), key


def test_k4_zero_source_applicable_denominator_is_null():
    got = k4_rates([{"source": 0, "applicable_negatives": 2}])
    assert got["conditional_any_negative_rate"] is None
    assert got["conditional_mean_negative_fraction"] is None


def test_bleu4_perfect_and_empty():
    assert corpus_bleu4(["a cat sits on a mat"], [["a cat sits on a mat"]]) == pytest.approx(1.0)
    assert corpus_bleu4([""], [["a cat sits on a mat"]]) == 0.0


def test_bleu4_matches_coco_caption_when_all_orders_present():
    pycoco = pytest.importorskip("pycocoevalcap")
    assert pycoco is not None
    cands = ["a cat sits on a mat", "the dog runs in the park", "a man rides a bike"]
    refs = [["a cat sits on a mat"], ["the dog runs in the park"], ["a man rides a bike"]]
    assert corpus_bleu4(cands, refs) == pytest.approx(coco_caption_bleu4(cands, refs), abs=1e-6)


def test_bleu4_smoothing_is_method1_variant_on_zero_count_orders():
    """Documented divergence: zero-count orders use method-1 smoothing, not coco additive smoothing."""
    cands = ["a cat sits on a mat", "the dog runs in the park", "a man rides a bike"]
    refs = [["the cat is on the mat"], ["a dog runs in a park"], ["a person rides a bicycle"]]
    ours = corpus_bleu4(cands, refs)
    assert ours > 0.1  # method-1 keeps the metric alive
    try:
        from pycocoevalcap.bleu.bleu import Bleu

        gts = {i: r for i, r in enumerate(refs)}
        res = {i: [c] for i, c in enumerate(cands)}
        coco = float(Bleu(4).compute_score(gts, res, verbose=0)[0][3])
        assert abs(coco - ours) > 1e-3
    except ModuleNotFoundError:
        pass


def test_distinct_2_is_corpus_level_over_all_bigrams():
    # bigrams: (a,b),(a,c),(a,b) -> unique 2 of 3
    assert distinct_2(["a b", "a c", "a b"]) == pytest.approx(2 / 3)
    assert distinct_2(["one"]) == 0.0
    assert distinct_2([]) == 0.0


def test_duplicate_rate_is_normalized_excess_rate():
    assert duplicate_rate(["A", "A", "B"]) == pytest.approx(1 / 3)
    assert normalized_excess_duplicate_rate(["A", "A", "B"]) == pytest.approx(1 / 3)
    assert repeated_group_membership_rate(["A", "A", "B"]) == pytest.approx(2 / 3)
    assert duplicate_rate([" a", "A "]) == pytest.approx(0.5)  # strip+lower normalization


def test_qbleu_variant_range_and_question_type():
    assert qbleu4_variant(["what color is the cat"], [["what color is the cat"]]) == pytest.approx(1.0)
    assert 0.0 <= qbleu4_variant(["how many dogs"], [["what color is the cat"]]) <= 1.0
    assert question_type("How many dogs are there?") == "how"
    assert question_type("Is there a dog?") == "be"


def test_excluded_metrics_absent_from_active_package():
    names = set(dir(metrics))
    for excluded in ("bertscore", "bert_score", "pacscore", "vqascore"):
        assert excluded not in names
