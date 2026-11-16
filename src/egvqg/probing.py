"""P3 linear probes on captured hidden states (analysis only; needs numpy, scipy, scikit-learn, safetensors/torch).

Everything follows the pre-registered protocol in docs/implementation_plan.md 7.7:
 - folds hold out whole skills (templates); candidates of one material never straddle train/test;
 - fixed hyperparameters (no tuning): standardised L2 logistic regression C=0.01, ridge alpha = number of features;
 - baselines: input features (A) and lexical surface of the generated tokens (Q); control task = material-level random labels;
 - intervals: bootstrap over materials (paired for differences); 3 seeds change only the group-to-fold assignment.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from .records import read_jsonl, write_json
from .whitebox import has_question_states

C_LOGISTIC = 0.01
POSITIONS = ('prompt_end', 'early', 'question_end')


# ---------- folds, metrics, bootstrap ----------

def grouped_folds(groups, n_folds, seed):
    """Fold id per sample; whole groups share a fold, group-to-fold assignment is shuffled by `seed` and balanced by group count."""
    uniq = sorted(set(groups))
    order = random.Random(seed).sample(uniq, len(uniq))
    fold_of = {g: i % n_folds for i, g in enumerate(order)}
    return np.array([fold_of[g] for g in groups])


def auc(y, score):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return float('nan')
    if score.ndim == 2 and score.shape[1] > 2:
        present = sorted(set(y.tolist()))
        if len(present) < score.shape[1]:
            return float('nan')
        return float(roc_auc_score(y, score, multi_class='ovr', average='macro'))
    return float(roc_auc_score(y, score if score.ndim == 1 else score[:, 1]))


def r2(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    denom = ((y - y.mean()) ** 2).sum()
    return float('nan') if denom == 0 else float(1 - ((y - pred) ** 2).sum() / denom)


def bootstrap_groups(groups, metric_fns, B=1000, seed=0):
    """Paired bootstrap over groups. metric_fns: dict name -> f(index_array) -> float. Returns {name: samples}."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in uniq}
    out = {n: [] for n in metric_fns}
    for _ in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[g] for g in pick])
        for n, f in metric_fns.items():
            out[n].append(f(idx))
    return {n: np.array(v) for n, v in out.items()}


def interval(samples, level=0.95):
    s = samples[~np.isnan(samples)]
    if len(s) < 20:
        return [float('nan'), float('nan')]
    lo, hi = np.quantile(s, [(1 - level) / 2, 1 - (1 - level) / 2])
    return [float(lo), float(hi)]


# ---------- probes ----------

def fit_predict(Xtr, ytr, Xte, kind):
    """kind 'binary'|'multiclass' -> class scores; 'reg' -> predictions. Standardised with training statistics."""
    from sklearn.linear_model import LogisticRegression, Ridge
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-6] = 1.0
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    if kind == 'reg':
        return Ridge(alpha=float(Xtr.shape[1])).fit(A, ytr).predict(B)
    model = LogisticRegression(C=C_LOGISTIC, max_iter=300).fit(A, ytr)
    scores = model.predict_proba(B)
    full = np.zeros((len(Xte), int(max(ytr.max(), 1)) + 1)) if kind != 'reg' else None
    full[:, model.classes_] = scores
    return full


def oof(X, y, skills, kind, seed, n_folds=5, folds=None):
    """Out-of-fold predictions with whole skills held out. `folds` may be supplied (e.g. label-balanced)."""
    if folds is None:
        folds = grouped_folds(skills, n_folds, seed)
    y = np.asarray(y)
    out = None
    for f in range(n_folds):
        te = folds == f
        tr = ~te
        if te.sum() == 0 or tr.sum() == 0:
            continue
        if kind != 'reg' and len(set(y[tr].tolist())) < 2:
            continue
        pred = fit_predict(X[tr], y[tr], X[te], kind)
        if out is None:
            out = np.full((len(y),) + pred.shape[1:], np.nan)
        out[te] = pred
    return out


def metric_for(kind):
    return r2 if kind == 'reg' else auc


def score(y, pred, kind):
    ok = ~np.isnan(pred).any(axis=tuple(range(1, pred.ndim))) if pred.ndim > 1 else ~np.isnan(pred)
    return metric_for(kind)(np.asarray(y)[ok], pred[ok])


def control_labels(material_ids, y, seed):
    """Control task: every material gets one random label (matching the marginal frequency) shared by all its candidates (binary),
    or, for regression, the material-level targets are permuted among materials."""
    rng = np.random.default_rng(10_000 + seed)
    uniq = sorted(set(material_ids))
    y = np.asarray(y, float)
    first = {m: y[[i for i, mm in enumerate(material_ids) if mm == m]].mean() for m in uniq}
    values = rng.permutation([first[m] for m in uniq])
    mapped = dict(zip(uniq, values))
    return np.array([mapped[m] for m in material_ids])


def to_control_binary(material_ids, y, seed):
    rng = np.random.default_rng(20_000 + seed)
    uniq = sorted(set(material_ids))
    p = float(np.mean(y))
    draw = {m: int(rng.random() < p) for m in uniq}
    return np.array([draw[m] for m in material_ids])


# ---------- features ----------

def input_features(rows, with_language):
    cols = []
    for r in rows:
        f = [r['text_chars'], r.get('n_choices') or 0, r['image']['width'], r['image']['height'], r['image']['mean'], r['image']['std']]
        if with_language:
            f.append(r['zh_share'])
        cols.append(f)
    return np.array(cols, float)


def lexical_features(rows, n_tokens=None, extra=True):
    """Bag of generated token ids (first n_tokens if given) plus length and language share, as a dense hashed matrix."""
    dim = 512
    X = np.zeros((len(rows), dim + (2 if extra else 0)))
    for i, r in enumerate(rows):
        ids = r['generated_token_ids'][:-1] if r['finish'] == 'stop' else r['generated_token_ids']
        ids = ids[:n_tokens] if n_tokens else ids
        for t in ids:
            X[i, t % dim] += 1
        if extra:
            X[i, dim], X[i, dim + 1] = len(r['generated_token_ids']), r['zh_share']
    return X


# ---------- data ----------

def load_candidates(capture_runs, label_files, *, project_root='.', output_root='outputs', layers=None):
    """Rows (captures joined with labels) and state arrays: prompt_end [n,37,H], early_mean [n,37,H], question_end [n,37,H].

    State conversion is thread parallel: the per-file bf16 -> float32 conversion is ~0.5 s with torch's default
    64-thread pool, but ~0.02 s per file when 8 files are processed concurrently (bit-identical tensors)."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    from safetensors.torch import load_file
    base = Path(project_root).resolve() / output_root
    work = []
    for run, label_file in zip(capture_runs, label_files):
        labels = {r['candidate_id']: r for r in read_jsonl(base / run / label_file)}
        for c in read_jsonl(base / run / 'captures.jsonl'):
            lab = labels[c['candidate_id']]
            if not has_question_states(c):
                continue                                            # no question states (empty question)
            work.append((run, c, lab))
    sel = (lambda a: a) if layers is None else (lambda a: a[..., layers, :])

    def one(item):
        run, c, lab = item
        t = load_file(str(base / run / c['states_file']))
        row = {**c, **{k: lab[k] for k in ('valid', 'redundant', 'direction', 'decided', 'evaluated')}, 'run': run}
        return (row, sel(t['prompt_end'].float().numpy()), sel(t['early'].float().numpy().mean(0)),
                sel(t['question_end'].float().numpy()))

    with ThreadPoolExecutor(max_workers=int(os.environ.get('EGVQG_IO_THREADS', '8'))) as pool:
        loaded = list(pool.map(one, work))
    rows = [x[0] for x in loaded]
    pe = [x[1] for x in loaded]
    em = [x[2] for x in loaded]
    qe = [x[3] for x in loaded]
    return rows, np.stack(pe), np.stack(em), np.stack(qe)


def direction_codes(rows):
    order = {'image': 0, 'text': 1, 'both': 2}
    return np.array([order.get(r['direction'], -1) for r in rows])
