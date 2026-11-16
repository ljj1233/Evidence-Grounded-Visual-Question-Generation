"""Variance decomposition recovers planted components and gives no skill/material share to pure candidate noise."""
import pytest

np = pytest.importorskip('numpy')

from egvqg.variance_decomposition import components, loo_material_r2


def world(sd_skill, sd_mat, sd_cand, n_skill=40, mats=6, per=8, seed=0):
    rng = np.random.default_rng(seed)
    y, sk, mt = [], [], []
    for s in range(n_skill):
        a = rng.normal(0, sd_skill)
        for m in range(mats):
            b = rng.normal(0, sd_mat)
            for _ in range(per):
                y.append(a + b + rng.normal(0, sd_cand)); sk.append(f's{s}'); mt.append(f's{s}m{m}')
    return y, sk, mt


def test_planted_components_are_recovered_approximately():
    c = components(*world(1.0, 0.5, 0.5))
    assert c['share']['skill'] == pytest.approx(1.0 / 1.5, abs=0.12)
    assert c['share']['material'] == pytest.approx(0.25 / 1.5, abs=0.08)
    assert c['share']['candidate'] == pytest.approx(0.25 / 1.5, abs=0.05)


def test_pure_candidate_noise_has_no_skill_or_material_share_and_skill_means_do_not_predict():
    y, sk, mt = world(0.0, 0.0, 1.0)
    c = components(y, sk, mt)
    assert c['share']['candidate'] > 0.9
    assert loo_material_r2(y, sk, mt)['r2_oos'] < 0.1


def test_strong_skill_effect_is_predictive_out_of_sample_and_too_few_groups_returns_none():
    y, sk, mt = world(1.0, 0.1, 0.3)
    assert loo_material_r2(y, sk, mt)['r2_oos'] > 0.5
    assert components([1, 0, 1, 0], ['a', 'a', 'b', 'b'], ['m1', 'm1', 'm2', 'm2']) is None
