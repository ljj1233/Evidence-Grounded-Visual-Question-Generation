"""Steering: hook touches only the intended positions, directions are clean of skill offsets and language, controls match in norm, screening and G3 logic."""
import pytest

np = pytest.importorskip('numpy')
torch = pytest.importorskip('torch')

from egvqg.intervene import (demean_by_group, g3_verdict, gaussian_direction, paired_effect, random_manifold_direction, real_direction,
                             screen_conditions, steering)


class Layer(torch.nn.Module):
    def forward(self, x):
        return x


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.layers = torch.nn.ModuleList([Layer(), Layer()])

    def forward(self, x):
        for layer in self.model.language_model.layers:
            x = layer(x)
        return x


def test_hook_adds_only_to_the_last_prompt_position_and_to_every_decode_position_and_is_removed_afterwards():
    model, v = Toy(), np.ones(4, dtype=np.float32)
    prefill, decode = torch.zeros(1, 5, 4), torch.zeros(1, 1, 4)
    with steering(model, 1, v, alpha=2.0):
        out_pre, out_dec = model(prefill), model(decode)
    assert out_pre[0, -1].tolist() == [2.0] * 4 and float(out_pre[0, :-1].abs().sum()) == 0.0
    assert out_dec[0, 0].tolist() == [2.0] * 4
    assert float(model(prefill).abs().sum()) == 0.0                         # hook removed on exit
    assert float(prefill.abs().sum()) == 0.0                                # input not modified in place


def planted(n_skill=6, per=30, h=40, seed=0):
    rng = np.random.default_rng(seed)
    d_true, lang_dir = np.zeros(h), np.zeros(h)
    d_true[0], lang_dir[1] = 1.0, 1.0
    X, y, sk, zh = [], [], [], []
    for s in range(n_skill):
        offset = rng.normal(size=h) * 3
        for _ in range(per):
            label, z = int(rng.random() < 0.4), int(rng.random() < 0.5)
            x = offset + rng.normal(size=h) * 0.3 + 1.5 * label * d_true + 2.0 * z * lang_dir + 1.0 * label * lang_dir * z
            X.append(x); y.append(label); sk.append(f's{s}'); zh.append(z)
    return np.array(X), np.array(y), sk, np.array(zh), d_true, lang_dir


def test_direction_ignores_skill_offsets_and_language_and_finds_the_planted_axis():
    X, y, sk, zh, d_true, lang = planted()
    d, sigma, info = real_direction(X, y, sk, zh)
    assert abs(d @ d_true) > 0.9 and abs(d @ lang) < 0.15 and sigma > 0 and info['n_pos'] > 0     # orthogonal to the *estimated* language axis
    assert np.allclose(demean_by_group(X, sk).mean(0), 0, atol=1e-8)


def test_controls_are_unit_norm_and_nearly_orthogonal_to_the_real_direction():
    X, y, sk, zh, d_true, _ = planted()
    d, _, _ = real_direction(X, y, sk, zh)
    r, g = random_manifold_direction(X, y, sk, zh, 1), gaussian_direction(X.shape[1], 2)
    assert abs(np.linalg.norm(r) - 1) < 1e-6 and abs(np.linalg.norm(g) - 1) < 1e-6
    assert abs(d @ r) < 0.7 and abs(d @ g) < 0.7


def rows(n_mat, ended=1.0, tokens=10, zh=1.0, success=0.1, valid=1.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for m in range(n_mat):
        for j in range(4):
            ok = rng.random() < valid
            out.append({'material_id': f'm{m}', 'plan': {'complete': rng.random() < ended}, 'n_generated_tokens': tokens, 'zh_share': float(zh),
                        'question': f'q{m}{j}{seed}', 'valid': 1 if ok else 0, 'redundant': 0 if ok and rng.random() < success else (1 if ok else None)})
    return out


def test_screening_drops_gibberish_length_blowup_and_language_flip():
    conds = {'ctrl': rows(20), 'good': rows(20, seed=1), 'broken': rows(20, ended=0.3, seed=2), 'long': rows(20, tokens=60, seed=3), 'flip': rows(20, zh=0.0, seed=4)}
    res = screen_conditions(conds)
    assert res['good']['survives'] and not res['broken']['survives'] and not res['long']['survives'] and not res['flip']['survives']


def test_g3_needs_the_real_direction_to_beat_both_controls_and_keep_validity():
    conds = {'ctrl': rows(60, success=0.1), 'dir': rows(60, success=0.5, seed=1), 'rand': rows(60, success=0.1, seed=2), 'gauss': rows(60, success=0.1, seed=3)}
    eff = paired_effect(conds, B=200)
    assert g3_verdict(eff, 'dir', ['rand', 'gauss'])['passed']
    conds['rand'] = rows(60, success=0.5, seed=5)
    assert not g3_verdict(paired_effect(conds, B=200), 'dir', ['rand', 'gauss'])['passed']          # a random control works as well
    conds2 = {'ctrl': rows(60, success=0.1), 'dir': rows(60, success=0.5, valid=0.5, seed=1), 'rand': rows(60, seed=2), 'gauss': rows(60, seed=3)}
    assert not g3_verdict(paired_effect(conds2, B=200), 'dir', ['rand', 'gauss'])['passed']        # validity collapsed
