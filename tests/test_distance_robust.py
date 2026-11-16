"""R1/R2 must reject a score that only encodes the template and accept one that carries signal inside templates."""
import pytest

np = pytest.importorskip('numpy')
pytest.importorskip('sklearn')

from egvqg.distance_robust import r1_within_skill, r2_adjusted, residualize, usable_skills


def world(template_only, n_skill=8, mats=6, per=4, seed=0):
    rng = np.random.default_rng(seed)
    y, x, sk, mt, A = [], [], [], [], []
    for s in range(n_skill):
        rate = 0.1 + 0.8 * (s / (n_skill - 1))                 # label prevalence depends on the template
        for m in range(mats):
            for _ in range(per):
                label = int(rng.random() < rate)
                score = 3.0 * s + rng.normal() if template_only else 3.0 * s + 1.5 * label + rng.normal()
                y.append(label); x.append(score); sk.append(f's{s}'); mt.append(f's{s}m{m}')
                A.append([rng.normal(), rng.normal(), 100, 100, 100, 10, 0.5])
    return np.array(x), np.array(y), sk, mt, np.array(A, float)


def test_template_only_score_has_high_pooled_auc_but_no_within_skill_or_adjusted_signal():
    from egvqg.probing import auc
    x, y, sk, mt, A = world(True)
    assert auc(y, x) > 0.7                                              # the confound that fools a pooled AUC
    r1 = r1_within_skill(x, y, sk, mt, B=100)
    assert r1['status'] == 'ok' and r1['ci'][0] < 0.5 < r1['ci'][1]
    r2 = r2_adjusted(x.reshape(-1, 1), A, y, sk, mt, B=100)
    assert r2['gain_ci'][0] < 0 < r2['gain_ci'][1] or abs(r2['gain']) < 0.05


def test_signal_inside_templates_passes_both_checks():
    x, y, sk, mt, A = world(False)
    r1 = r1_within_skill(x, y, sk, mt, B=100)
    assert r1['ci'][0] > 0.5
    r2 = r2_adjusted(x.reshape(-1, 1), A, y, sk, mt, B=100)
    assert r2['gain'] > 0.05 and r2['gain_ci'][0] > 0


def test_too_few_skills_or_positives_is_not_evaluable_and_residualize_centres_each_skill():
    x, y, sk, mt, A = world(False, n_skill=2)
    assert r1_within_skill(x, y, sk, mt, B=50)['status'] == 'not_evaluable'
    assert usable_skills([0, 1, 0, 0], ['a', 'a', 'b', 'b']) == {'a'}
    res = residualize([1, 3, 10, 12], ['a', 'a', 'b', 'b'])
    assert list(res) == [-1, 1, -1, 1]
