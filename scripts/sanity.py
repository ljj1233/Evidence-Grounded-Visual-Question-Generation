"""Single CPU sanity check for the VQAG math (synthetic data, seconds).

Checks: (1) nested variance-share recovery within 5%; (2) a candidate-level direction signal survives material
centering while a skill-only redundancy signal drops to chance; (3) the alpha=1 steering clamp lands on b and
leaves the orthogonal complement untouched; (4) the paired bootstrap covers 0 on null paired data.
Run: .venv/bin/python scripts/sanity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from egvqg import configure_threads  # noqa: E402

configure_threads()

import numpy as np  # noqa: E402

from egvqg.centering import center, projection  # noqa: E402
from egvqg.diag_curve import _metric, oof_predict  # noqa: E402
from egvqg.diag_template import decomposition  # noqa: E402
from egvqg.stats import paired_bootstrap  # noqa: E402
from egvqg.steer import steer_numpy  # noqa: E402


def synthetic(seed=0, n_skills=12, n_mats=6, n_cands=8, dim=16, shares=(0.5, 0.3, 0.2)):
    rng = np.random.default_rng(seed)
    skill_ids, mat_ids = [], []
    for s in range(n_skills):
        for m in range(n_mats):
            for _ in range(n_cands):
                skill_ids.append(s); mat_ids.append(f's{s}m{m}')
    skill_ids, mat_ids = np.array(skill_ids), np.array(mat_ids)

    skill_vec = rng.normal(size=(n_skills, dim))
    skill_vec -= skill_vec.mean(0)
    mat_vec = rng.normal(size=(n_skills * n_mats, dim))
    mat_of, s_of = {}, {}
    k = 0
    for s in range(n_skills):
        block = mat_vec[k:k + n_mats]
        mat_vec[k:k + n_mats] -= block.mean(0)
        for m in range(n_mats):
            mat_of[f's{s}m{m}'] = k + m; s_of[f's{s}m{m}'] = s
        k += n_mats
    cand_vec = rng.normal(size=(len(mat_ids), dim))
    for m in np.unique(mat_ids):
        idx = np.where(mat_ids == m)[0]
        cand_vec[idx] -= cand_vec[idx].mean(0)

    rows_skill = np.array([skill_vec[s] for s in skill_ids])
    rows_mat = np.array([mat_vec[mat_of[m]] for m in mat_ids])
    parts = [rows_skill, rows_mat, cand_vec]
    for part, target in zip(parts, shares):
        ss = float(((part - part.mean(0)) ** 2).sum())
        part *= np.sqrt(target / ss)
    X = rows_skill + rows_mat + cand_vec + rng.normal(scale=0.001, size=(len(mat_ids), dim))
    return X, skill_ids, mat_ids


def check(name, ok, detail):
    print(f'[{"PASS" if ok else "FAIL"}] {name}: {detail}')
    return bool(ok)


def main():
    ok = True
    X, skill_ids, mat_ids = synthetic()
    dec = decomposition(X, skill_ids, mat_ids)
    errors = {k: abs(dec['share'][k] - t) for k, t in zip(('skill', 'material', 'candidate'), (0.5, 0.3, 0.2))}
    ok &= check('variance shares within 5%', max(errors.values()) < 0.05,
                ', '.join(f'{k}={dec["share"][k]:.3f} (err {v:.3f})' for k, v in errors.items()))

    u = np.random.default_rng(7).normal(size=X.shape[1]); u /= np.linalg.norm(u)
    delta = X - np.stack([X[mat_ids == m].mean(0) for m in mat_ids])       # material-centred residual
    y_dir = (delta @ u > 0).astype(int)
    y_red = (skill_ids % 2).astype(int)
    meta = {'skill': skill_ids, 'material_id': mat_ids, 'parent_id': mat_ids}
    Xc = center(X, meta, 'material')
    pred_dir = oof_predict(Xc, y_dir, meta['parent_id'], 'binary', seeds=(0,))
    pred_red = oof_predict(Xc, y_red, meta['parent_id'], 'binary', seeds=(0,))
    auc_dir, auc_red = _metric(y_dir, pred_dir), _metric(y_red, pred_red)
    ok &= check('candidate-level direction survives material centering', auc_dir > 0.6, f'AUC={auc_dir:.3f}')
    ok &= check('skill-only redundancy drops to chance', abs(auc_red - 0.5) < 0.1, f'AUC={auc_red:.3f}')

    w = np.random.default_rng(3).normal(size=X.shape[1]); w /= np.linalg.norm(w)
    h = np.random.default_rng(4).normal(size=(5, X.shape[1]))
    b = 0.37
    out = steer_numpy(h, w, alpha=1.0, b=b)
    proj_err = float(np.abs(projection(out, w) - b).max())
    perp = float(np.abs((out - h) - np.outer((out - h) @ w, w)).max())
    ok &= check('alpha=1 clamp sets projection to b', proj_err < 1e-10, f'max|proj-b|={proj_err:.2e}')
    ok &= check('clamp leaves orthogonal components unchanged', perp < 1e-10, f'max deviation={perp:.2e}')

    rng = np.random.default_rng(11)
    groups = np.repeat(np.arange(40), 5)
    null = rng.normal(size=len(groups)) * 0.5
    boot = paired_bootstrap(lambda idx: float(null[idx].mean()), groups, n=2000)
    ok &= check('paired bootstrap covers 0 on null data', boot['ci'][0] <= 0 <= boot['ci'][1],
                f'point={boot["point"]:+.3f} CI=[{boot["ci"][0]:+.3f}, {boot["ci"][1]:+.3f}]')

    print('\nSANITY', 'PASSED' if ok else 'FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
