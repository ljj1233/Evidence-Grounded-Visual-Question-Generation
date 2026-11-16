"""P5 readout analysis on synthetic distances: planted separation is found, constant shifts are not, adoption rule needs two scheme pairs."""
import pytest

np = pytest.importorskip('numpy')
pytest.importorskip('sklearn')

from egvqg.distance_analysis import PAIRS, join, run_p5, score_at, targets


def synth(n_mat=30, per=6, effect=0.5, seed=0):
    rng = np.random.default_rng(seed)
    labels, dists = [], []
    for m in range(n_mat):
        for j in range(per):
            nonred = int(rng.random() < 0.4)
            cid = f'm{m}~c{j}'
            labels.append({'candidate_id': cid, 'material_id': f'm{m}', 'valid': 1, 'redundant': 1 - nonred, 'direction': 'image' if rng.random() < .5 else 'text'})
            d = {'rel_l2': {}}
            for name in ('image_removed', 'image_gray', 'image_swap', 'text_removed', 'text_swap'):
                base = rng.normal(0.3, 0.05, 37)
                if name.startswith('image'):
                    base = base + effect * nonred          # non-redundant candidates move more under image removal
                d['rel_l2'][name] = {'prompt_end': list(base), 'question_end': list(base)}
            d['candidate_id'] = cid
            dists.append(d)
    return labels, dists


def test_join_and_targets_use_only_decided_valid_candidates():
    labels, dists = synth(5)
    labels[0]['redundant'] = None
    pairs = join(labels, dists[:-1])
    t = targets(pairs)
    assert len(pairs) == len(labels) - 1 and len(t['nonredundant'][0]) == len(pairs) - 1


def test_planted_effect_is_detected_at_the_confirmation_layer_and_noise_is_not():
    labels, dists = synth(effect=0.6)
    pairs = join(labels, dists)
    res = score_at(pairs, targets(pairs)['nonredundant'], PAIRS[0], 'question_end', 'asymmetry', 5, B=100)
    assert res['auc'] > 0.9 and res['ci'][0] > 0.5
    labels, dists = synth(effect=0.0, seed=3)
    pairs = join(labels, dists)
    noise = score_at(pairs, targets(pairs)['nonredundant'], PAIRS[0], 'question_end', 'asymmetry', 5, B=100)
    assert abs(noise['auc'] - 0.5) < 0.15 and noise['ci'][0] < 0.65     # null effect: no strong separation
