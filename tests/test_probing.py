"""Probe machinery on synthetic data: leakage-free folds, planted signal found, noise not, control and G2 logic.
Needs numpy/scikit-learn (the project .venv lacks them; run under the shared venv with the project site-packages on PYTHONPATH)."""
import pytest

np = pytest.importorskip('numpy')
pytest.importorskip('sklearn')

from egvqg.probe_run import evaluate_task, g2_verdict
from egvqg.probing import bootstrap_groups, grouped_folds, interval, oof, r2, score, to_control_binary


def synthetic(n_mat=40, per=6, signal=2.0, seed=0, d=30):
    rng = np.random.default_rng(seed)
    materials, skills, X, y = [], [], [], []
    for m in range(n_mat):
        lab = int(rng.random() < 0.5)
        for _ in range(per):
            materials.append(f'm{m}')
            skills.append(f's{m % 10}')
            v = rng.normal(size=d)
            v[0] += signal * (lab - 0.5)
            X.append(v)
            y.append(lab)
    return np.array(X), np.array(y), skills, materials


def test_folds_keep_each_group_in_one_fold_and_seed_changes_assignment():
    groups = [f'g{i % 12}' for i in range(120)]
    folds = grouped_folds(groups, 5, 0)
    assert all(len({folds[i] for i, g in enumerate(groups) if g == name}) == 1 for name in set(groups))
    assert not np.array_equal(folds, grouped_folds(groups, 5, 1))


def test_probe_finds_planted_signal_on_held_out_skills_and_not_pure_noise():
    X, y, skills, _ = synthetic(signal=3.0)
    assert score(y, oof(X, y, skills, 'binary', 0), 'binary') > 0.8
    Xn, yn, sk, _ = synthetic(signal=0.0, seed=1)
    assert abs(score(yn, oof(Xn, yn, sk, 'binary', 0), 'binary') - 0.5) < 0.2


def test_g2_passes_with_real_signal_and_fails_on_noise():
    X, y, skills, mats = synthetic(signal=3.0, n_mat=60)
    base = {'noise': np.random.default_rng(5).normal(size=(len(y), 3))}
    res = evaluate_task(X, y, skills, mats, 'binary', base, B=200)
    assert res['point']['probe'] > res['point']['noise'] and res['selectivity']['ci'][0] > 0 and g2_verdict(res)['passed']
    Xn, yn, sk, ma = synthetic(signal=0.0, n_mat=60, seed=3)
    assert not g2_verdict(evaluate_task(Xn, yn, sk, ma, 'binary', base, B=200))['passed']


def test_control_labels_are_constant_within_material_and_bootstrap_resamples_whole_groups():
    mats = ['a', 'a', 'b', 'b', 'c', 'c']
    c = to_control_binary(mats, np.array([1, 0, 1, 1, 0, 0]), 0)
    assert c[0] == c[1] and c[2] == c[3] and c[4] == c[5]
    boot = bootstrap_groups(mats, {'n': lambda idx: float(len(idx))}, B=40)
    assert set(boot['n'].tolist()) == {6.0}                          # always 3 groups of size 2
    assert all(np.isnan(v) for v in interval(np.array([0.0] * 5)))   # too few samples -> no interval


def test_r2_of_constant_target_is_undefined():
    assert np.isnan(r2([1, 1, 1], [1, 2, 3])) and r2([1, 2, 3], [1, 2, 3]) == 1.0
