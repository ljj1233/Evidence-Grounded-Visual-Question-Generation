"""2c / baseline: keep only candidates whose answer changes (or becomes undeterminable) under BOTH
counterfactuals — text removed and image removed.

Source: the cascade evaluation records (`items/<attempt_id>/record.json`, `screen_image_only` /
`screen_text_only` with the model's answer). A side "passes" when the answer is not determinable, or when the
determinable answer differs from the full image+text answer. Kept = both sides pass. No new calls, no GPU.
The rewriting branch is left as a signature only.
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

from . import candidate_table as CT
from .records import read_jsonl, write_json
from .stats import paired_bootstrap, wilson


def _norm(text):
    return ' '.join((text or '').lower().split())


def side_changed(output, full_answer):
    """True when this counterfactual's answer is undeterminable or differs from the full answer."""
    if not isinstance(output, dict) or 'determinable' not in output:
        return False
    if not output.get('determinable'):
        return True
    return _norm(output.get('answer')) != _norm(full_answer)


def load_records(eval_run, project_root='.', output_root='outputs'):
    base = Path(project_root).resolve() / output_root / eval_run
    out = {}
    for path in glob.glob(str(base / 'items' / '*' / 'record.json')):
        rec = json.loads(Path(path).read_text())
        out[rec['material_id']] = rec
    return out


def load_pseudo_map(capture_run, project_root='.', output_root='outputs'):
    base = Path(project_root).resolve() / output_root / capture_run
    files = sorted(glob.glob(str(base / 'source_map_*.json')))
    if not files:
        raise FileNotFoundError('source_map_<source_run>.json not found next to the capture run')
    table = json.loads(Path(files[-1]).read_text())
    return {cid: pid for pid, v in table.items() for cid in v['candidate_ids']}, table


def run_filter_baseline(*, capture_run, label_file, eval_run, project_root='.', output_root='outputs', run_id=None,
                        n_materials=20, label_new=None, bootstrap=2000, dry_run=False):
    root = Path(project_root).resolve()
    base = root / output_root
    rows = CT.build_table(capture_run=capture_run, label_file=label_file, project_root=project_root,
                          output_root=output_root, label_new=label_new)
    parents = sorted({r['parent_id'] for r in rows})[:n_materials]
    rows = [r for r in rows if r['parent_id'] in set(parents)]
    cand_pseudo, _ = load_pseudo_map(capture_run, project_root, output_root)
    records = load_records(eval_run, project_root, output_root)
    source = {r['material_id']: r for r in read_jsonl(base / eval_run / 'source_questions.jsonl')}
    decisions = []
    for r in rows:
        pid = cand_pseudo.get(r['cand_id'])
        rec, src = records.get(pid), source.get(pid)
        if not rec or not src:
            decisions.append((r, None, None))
            continue
        full = ((src.get('responses') or {}).get('TI') or {}).get('answer')
        outs = rec.get('outputs') or {}
        img = (outs.get('screen_image_only') or {}).get('output')
        txt = (outs.get('screen_text_only') or {}).get('output')
        img_changed = side_changed(img, full) if img is not None else None
        txt_changed = side_changed(txt, full) if txt is not None else None
        decisions.append((r, img_changed, txt_changed))
    kept = [r for r, i, t in decisions if i is True and t is True]
    evaluated = [(r, i, t) for r, i, t in decisions if i is not None and t is not None]
    if dry_run:
        print(f'[dry-run] filter-baseline: {len(rows)} candidates from {len(parents)} parents, {len(evaluated)} with both screens; '
              f'would keep {len(kept)} ({len(kept) / max(len(evaluated), 1):.1%}); no API/GPU used')
        return {'dry_run': True, 'candidates': len(rows), 'evaluated': len(evaluated), 'kept': len(kept),
                'retention': len(kept) / max(len(evaluated), 1)}

    def success(r):
        new = (r.get('label_new') or {}).get('verdict')
        ok = (new == 'dual_necessary') if new else (r['label_old'].get('redundant') == 0)
        return 1 if r.get('valid') == 1 and ok else 0

    by = defaultdict(list)
    for r, i, t in evaluated:
        by[r['material_id']].append((r, i is True and t is True))
    recs = []
    for m, items in by.items():
        sel = [success(r) for r, keep in items if keep]
        if not sel:
            continue
        recs.append({'material_id': m, 'parent_id': items[0][0].get('parent_id') or m,
                     'kept_success': sum(sel) / len(sel), 'all_success': sum(success(r) for r, _ in items) / len(items),
                     'n_kept': len(sel), 'n_all': len(items)})
    groups = [r['parent_id'] for r in recs]
    diff = paired_bootstrap(lambda idx: sum(recs[i]['kept_success'] - recs[i]['all_success'] for i in idx) / len(idx), groups, n=bootstrap) if len(recs) >= 5 else None
    result = {'candidates': len(rows), 'evaluated': len(evaluated), 'kept': len(kept),
              'retention': len(kept) / max(len(evaluated), 1),
              'kept_dual_wilson': wilson(sum(success(r) for r in kept), len(kept)),
              'all_dual_wilson': wilson(sum(success(r) for r, _, _ in evaluated), len(evaluated)),
              'n_materials': len(by), 'paired_diff_kept_minus_all': diff,
              'outcome': 'valid and label_new dual_necessary (old non-redundant only when new labels are absent)',
              'rule': 'keep iff both image-only and text-only answers are undeterminable or differ from the full TI answer'}
    out = base / run_id
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'filter_baseline.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'rule'}, indent=2, default=float))
    return result


def rewrite_candidates(*, kept, rows, model, caller, output_path):
    """Reserved: rewrite kept candidates to remove the redundant side. Not implemented in this round."""
    raise NotImplementedError('rewrite branch is reserved; this round implements the filter baseline only')
