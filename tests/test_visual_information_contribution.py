import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path('src').resolve()))
from vvqg.evaluation.metrics.visual_information import normalize_answer, spearman, pair_bootstrap, stage_a_statistics, auc


def test_vqa_normalization():
    for raw, expected in [("The two cats.", "2 cats"), ("don't", "don't"), ("3.5", "3.5"), ("1,000", "1000"), ("none", "0"), ("twenty", "twenty"), ("well", "well"), ("2.", "2")]:
        assert normalize_answer(raw) == expected


def test_tied_spearman():
    assert np.isclose(spearman(np.array([1, 1, 3]), np.array([1, 2, 3])), np.sqrt(3) / 2)


def test_pair_bootstrap_keeps_both_directions_and_direction_median():
    margins = np.array([[1., -1.], [10., -2.]])
    gt = np.array([[1., 2.], [3., 4.]])
    result = pair_bootstrap(margins, gt, gt, n=50)
    assert set(result['switch_accuracy']) == {.5}
    assert stage_a_statistics(margins, gt, gt)['switch_median_margin'] == 0
    assert set(result['switch_median_margin']) <= {0, 4}
    assert auc([1, 2], [2, 3]) == .875
