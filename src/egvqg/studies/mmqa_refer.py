"""Compare two MMQA question-generation conditions under the same label file format.

Reads candidate tables with `label_consensus` / `label_new` (label-repair output), recomputes inline
features with `inline_analysis.inline_features`, and reports fixed-denominator rates with Wilson
intervals plus a material-clustered bootstrap of the treatment-minus-control difference.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from .inline_analysis import inline_features
from ..core.records import read_jsonl, utc_now, write_json
from ..core.stats import wilson

DEFAULT_MATERIALS = 'data/processed/multimodalqa/mmqa-pool-v1/pvs60-materials.jsonl'


def _condition(rows, materials):
    enriched = []
    for row in rows:
        material = materials.get(row['material_id'])
        if material is None:
            continue
        verdict = (row.get('label_consensus') or {}).get('verdict')
        new = row.get('label_new') or {}
        groups = (new.get('evidence') or {}).get('groups') or {}
        judge = {name: (g.get('verdict')) for name, g in groups.items()}
        features = inline_features(row['question'], material['text'], min_digits=2, use_lexical=True)
        decided = verdict is not None and verdict != 'undetermined'
        enriched.append({**row, **features, 'verdict': verdict, 'decided': decided,
                         # text redundancy = the text brings no unique information = the image alone suffices.
                         'text_redundant': None if not decided else verdict in ('image_suff', 'both_suff'),
                         'image_redundant': None if not decided else verdict in ('text_suff', 'both_suff'),
                         'judge': judge, 'new_verdict': new.get('verdict')})
    n = len(enriched)
    decided = [r for r in enriched if r['decided']]
    dual = wilson(sum(r['verdict'] == 'dual_necessary' for r in enriched), n)
    dual_decided = wilson(sum(r['verdict'] == 'dual_necessary' for r in decided), len(decided))
    redundant = wilson(sum(bool(r['text_redundant']) for r in decided), len(decided))
    image_redundant = wilson(sum(bool(r['image_redundant']) for r in decided), len(decided))
    inline = wilson(sum(r['inline_any'] for r in enriched), n)
    inline_num = wilson(sum(r['inline_number'] for r in enriched), n)
    inline_lex = wilson(sum(r['inline_lexical'] for r in enriched), n)
    inline_rows = [r for r in enriched if r['inline_any'] and r['decided']]
    noninline_rows = [r for r in enriched if not r['inline_any'] and r['decided']]
    new_counts = Counter(r['new_verdict'] for r in enriched)
    judge_dual = {name: sum(v == 'dual_necessary' for v in (r['judge'].get(name) for r in enriched))
                  for name in sorted({k for r in enriched for k in r['judge']})}
    return {'n': n, 'decided': len(decided), 'consensus_verdicts': dict(Counter(r['verdict'] for r in enriched)),
            'label_new_verdicts': dict(new_counts), 'judge_dual': judge_dual,
            'dual_rate': dual, 'dual_rate_decided': dual_decided,
            'text_redundancy': redundant,
            'image_redundancy': image_redundant,
            'text_redundancy_conservative': wilson(sum(bool(r['text_redundant']) for r in enriched), n),
            'image_redundancy_conservative': wilson(sum(bool(r['image_redundant']) for r in enriched), n),
            'inline_any': inline, 'inline_number': inline_num, 'inline_lexical': inline_lex,
            'text_redundancy_inline': wilson(sum(bool(r['text_redundant']) for r in inline_rows), len(inline_rows)),
            'text_redundancy_non_inline': wilson(sum(bool(r['text_redundant']) for r in noninline_rows), len(noninline_rows)),
            'image_redundancy_inline': wilson(sum(bool(r['image_redundant']) for r in inline_rows), len(inline_rows)),
            'image_redundancy_non_inline': wilson(sum(bool(r['image_redundant']) for r in noninline_rows), len(noninline_rows)),
            'rows': enriched}


def _material_bootstrap(control, treatment, *, n=10000, seed=20261115):
    """Resample materials with replacement within each condition; unpaired because the questions differ."""
    def per_material(rows, value_fn, denom_fn):
        acc = {}
        for r in rows:
            key = r['material_id']
            num, den = acc.get(key, (0, 0))
            acc[key] = (num + value_fn(r), den + denom_fn(r))
        return acc
    out = {}
    for name, value_fn, denom_fn in (
            ('dual_rate', lambda r: int(r['verdict'] == 'dual_necessary'), lambda r: 1),
            ('text_redundancy', lambda r: int(bool(r['text_redundant'])), lambda r: int(bool(r['decided']))),
            ('image_redundancy', lambda r: int(bool(r['image_redundant'])), lambda r: int(bool(r['decided']))),
            ('inline_any', lambda r: int(r['inline_any']), lambda r: 1)):
        c = per_material(control['rows'], value_fn, denom_fn)
        t = per_material(treatment['rows'], value_fn, denom_fn)
        ids = sorted(set(c) & set(t))
        if not ids:
            out[name] = {'point': None, 'ci': [None, None]}
            continue
        rng = np.random.default_rng(seed)
        diffs = []
        for _ in range(n):
            pick = rng.choice(ids, size=len(ids), replace=True)
            cn = sum(c[m][0] for m in pick); cd = sum(c[m][1] for m in pick)
            tn = sum(t[m][0] for m in pick); td = sum(t[m][1] for m in pick)
            diffs.append((tn / td if td else np.nan) - (cn / cd if cd else np.nan))
        arr = np.asarray(diffs, float); arr = arr[~np.isnan(arr)]
        lo, hi = np.quantile(arr, [0.025, 0.975]) if len(arr) >= 20 else (np.nan, np.nan)
        cp = sum(c[m][0] for m in ids) / sum(c[m][1] for m in ids)
        tp = sum(t[m][0] for m in ids) / sum(t[m][1] for m in ids)
        out[name] = {'control': cp, 'treatment': tp, 'difference': tp - cp,
                     'ci95': [float(lo), float(hi)], 'n_boot': int(len(arr)), 'n_materials': len(ids)}
    return out


def _local_validity(path):
    if not path or not Path(path).exists():
        return None
    report = json.loads(Path(path).read_text())
    parsed = report.get('parsed') or 0
    n = report.get('n') or 0
    valid = report.get('valid_rate')
    return {'n': n, 'parsed': parsed, 'model_valid_rate_parsed': valid,
            'model_valid_conservative': (round(valid * parsed / n, 6) if valid is not None and n else None),
            'programmatic_valid_rate': report.get('programmatic_valid_rate')}


def _cascade_condition(path, materials):
    rows = []
    for cand_id, item in json.loads(Path(path).read_text()).items():
        material = materials.get(item['material_id'])
        if material is None:
            continue
        features = inline_features(item['question'], material['text'], min_digits=2, use_lexical=True)
        valid = item['answerability'] == 1
        rows.append({'cand_id': cand_id, 'material_id': item['material_id'],
                     'question': item['question'], 'valid': valid,
                     'dual': bool(item['dual']) and valid,
                     'text_redundant': (None if not valid else item['text_suff'] in ('image_suff', 'both_suff')),
                     'image_redundant': (None if not valid else item['text_suff'] in ('text_suff', 'both_suff')),
                     'verdict': item['verdict'], **features})
    n = len(rows)
    valid = [r for r in rows if r['valid']]
    return {'n': n, 'valid': len(valid),
            'validity': wilson(sum(r['valid'] for r in rows), n),
            'dual_rate_all': wilson(sum(r['dual'] for r in rows), n),
            'dual_rate_valid': wilson(sum(r['dual'] for r in valid), len(valid)),
            'text_redundancy_valid': wilson(sum(bool(r['text_redundant']) for r in valid), len(valid)),
            'image_redundancy_valid': wilson(sum(bool(r['image_redundant']) for r in valid), len(valid)),
            'inline_any': wilson(sum(r['inline_any'] for r in rows), n),
            'verdicts': dict(Counter(r['verdict'] for r in rows)), 'rows': rows}


def _cascade_bootstrap(control, treatment, *, n=10000, seed=20261115):
    def per_material(rows, value_fn, denom_fn):
        acc = {}
        for r in rows:
            num, den = acc.get(r['material_id'], (0, 0))
            acc[r['material_id']] = (num + value_fn(r), den + denom_fn(r))
        return acc
    out = {}
    for name, value_fn, denom_fn in (
            ('dual_rate_all', lambda r: int(r['dual']), lambda r: 1),
            ('dual_rate_valid', lambda r: int(r['dual']), lambda r: int(r['valid'])),
            ('validity', lambda r: int(r['valid']), lambda r: 1),
            ('text_redundancy_valid', lambda r: int(bool(r['text_redundant'])), lambda r: int(r['valid'])),
            ('image_redundancy_valid', lambda r: int(bool(r['image_redundant'])), lambda r: int(r['valid']))):
        c = per_material(control['rows'], value_fn, denom_fn)
        t = per_material(treatment['rows'], value_fn, denom_fn)
        ids = sorted(set(c) & set(t)); rng = np.random.default_rng(seed); diffs = []
        for _ in range(n):
            pick = rng.choice(ids, size=len(ids), replace=True)
            cd = sum(c[m][1] for m in pick); td = sum(t[m][1] for m in pick)
            diffs.append((sum(t[m][0] for m in pick) / td if td else np.nan) -
                         (sum(c[m][0] for m in pick) / cd if cd else np.nan))
        arr = np.asarray(diffs, float); arr = arr[~np.isnan(arr)]
        lo, hi = np.quantile(arr, [0.025, 0.975]) if len(arr) >= 20 else (np.nan, np.nan)
        cp = sum(c[m][0] for m in ids) / sum(c[m][1] for m in ids)
        tp = sum(t[m][0] for m in ids) / sum(t[m][1] for m in ids)
        out[name] = {'control': cp, 'treatment': tp, 'difference': tp - cp,
                     'ci95': [float(lo), float(hi)], 'n_boot': int(len(arr)), 'n_materials': len(ids)}
    return out


def compare_mmqa_refer(*, run_id, control, treatment, materials_path=DEFAULT_MATERIALS,
                       control_validity=None, treatment_validity=None, control_cascade=None,
                       treatment_cascade=None, bootstrap=10000, project_root='.', output_root='outputs'):
    root = Path(project_root).resolve()
    materials = {m['material_id']: m for m in read_jsonl(root / materials_path)}
    c = _condition(read_jsonl(root / control), materials)
    t = _condition(read_jsonl(root / treatment), materials)
    result = {
        'created_at': utc_now(), 'run_id': run_id, 'dataset': 'mmqa',
        'materials_path': materials_path, 'control_path': control, 'treatment_path': treatment,
        'label_rule': ('label-repair two independent judges (gpt-6-luna, gpt-5.6-terra) on the project '
                       '`ensemble-sufficiency-v1` screens; consensus requires both judges to return the same verdict, '
                       'disagreements and unresolved calls stay undetermined; dual_necessary = both single-modality screens fail'),
        'denominator': 'all planned candidate rows (n=360); undetermined is not a dual count',
        'control': {k: v for k, v in c.items() if k != 'rows'},
        'treatment': {k: v for k, v in t.items() if k != 'rows'},
        'bootstrap_material_clustered': _material_bootstrap(c, t, n=bootstrap),
        'local_validity': {'control': _local_validity(control_validity),
                           'treatment': _local_validity(treatment_validity),
                           'note': 'local-validity-v1 is a local Qwen AI review, not the official cascade question_answerability'},
    }
    if control_cascade and treatment_cascade:
        cc = _cascade_condition(control_cascade, materials)
        tc = _cascade_condition(treatment_cascade, materials)
        result['official_cascade'] = {
            'note': ('single-model gpt-6-luna cascade (review-question-faithfulness-v1 + ensemble-sufficiency-v1 screens); '
                     'question_answerability gates the screens; dual_necessary = both screens determinable=False; '
                     'cascade leaves both_required=None for survivors, so the label is read from the two screen outputs'),
            'sample': 'unique (material, question) pairs evaluated per condition, mapped back to all 360 candidate rows',
            'validity_denominator': 'all planned candidate rows (n=360); unresolved calls stay not-valid',
            'control': {k: v for k, v in cc.items() if k != 'rows'},
            'treatment': {k: v for k, v in tc.items() if k != 'rows'},
            'bootstrap_material_clustered': _cascade_bootstrap(cc, tc, n=bootstrap),
        }
    out = root / output_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'comparison.json', result)
    write_jsonl_rows = []
    for label, cond in (('control', c), ('treatment', t)):
        for r in cond['rows']:
            write_jsonl_rows.append({'condition': label, 'cand_id': r['cand_id'], 'material_id': r['material_id'],
                                     'question': r['question'], 'verdict': r['verdict'],
                                     'new_verdict': r['new_verdict'], 'judge': r['judge'],
                                     'inline_any': r['inline_any'], 'inline_number': r['inline_number'],
                                     'inline_lexical': r['inline_lexical'], 'text_redundant': r['text_redundant'],
                                     'image_redundant': r['image_redundant']})
    (out / 'comparison_rows.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in write_jsonl_rows) + '\n')
    return result
