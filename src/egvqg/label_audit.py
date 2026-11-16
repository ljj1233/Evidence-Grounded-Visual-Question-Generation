"""AI audit of cascade labels: stratified sample of candidates labelled redundant, rendered as sheets (image + material text +
question) for review by Claude. This is an AI review, not an independent human annotation.

The audit question is the one the screen answers: does the claimed modality alone give a unique full answer? A candidate is
`agree` when the reviewer finds at least one modality alone sufficient (what `redundant` asserts); `direction_agree` when the
reviewer also confirms the claimed direction (image / text / both).
"""
from __future__ import annotations

import random
import textwrap
from collections import defaultdict
from pathlib import Path

from .stats import wilson_lower
from .records import read_jsonl, write_json, write_jsonl

DIRECTION_QUOTA = (('image', 14), ('text', 12), ('both', 14))


def sample_two_class(rows, *, seed, n_redundant=40, n_nonredundant=40, quota=DIRECTION_QUOTA, per_material=2):
    """Unique pseudo-items for a two-class audit: up to `n_redundant` redundant (stratified by direction)
    and up to `n_nonredundant` non-redundant items (all distinct questions when fewer exist)."""
    rng = random.Random(seed)
    seen, red_pool, nonred_pool = set(), defaultdict(list), []
    for r in sorted(rows, key=lambda r: r['candidate_id']):
        if r['valid'] != 1 or r['pseudo_id'] in seen:
            continue
        seen.add(r['pseudo_id'])
        if r['redundant'] == 1 and r['direction']:
            red_pool[r['direction']].append(r)
        elif r['redundant'] == 0:
            nonred_pool.append(r)
    picked = []
    used = defaultdict(int)
    for direction, n in quota:
        items = red_pool[direction]
        rng.shuffle(items)
        for r in items:
            if len(picked) >= n_redundant:
                break
            if used[r['material_id']] < per_material and sum(p['direction'] == direction for p in picked) < n:
                used[r['material_id']] += 1
                picked.append({**r, 'audit_class': 'redundant'})
    rng.shuffle(nonred_pool)
    used = defaultdict(int)
    extra = []
    for r in nonred_pool:
        if len(extra) >= n_nonredundant:
            break
        if used[r['material_id']] < per_material:
            used[r['material_id']] += 1
            extra.append({**r, 'audit_class': 'nonredundant'})
    picked.extend(extra)
    rng.shuffle(picked)
    return picked


def render_two_class_sheets(items, materials, root, out_dir, per_sheet=2, width=1600):
    """Readable sheets for the two-class audit: full material text, question, and a large image."""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.load_default(size=18)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in range(0, len(items), per_sheet):
        tiles = []
        for n, r in enumerate(items[s:s + per_sheet], start=s):
            m = materials[r['material_id']]
            tile = Image.new('RGB', (width - 10, 760), 'white')
            d = ImageDraw.Draw(tile)
            text_lines = textwrap.wrap(m['text'], 100)[:8]
            for j, line in enumerate(text_lines):
                d.text((6, 4 + 20 * j), line, fill='black', font=font)
            qy = 8 + 20 * len(text_lines)
            for j, line in enumerate(textwrap.wrap('Q: ' + r['question'], 100)[:3]):
                d.text((6, qy + 20 * j), line, fill='blue', font=font)
            with Image.open(root / m['image_path']) as im:
                im = im.convert('RGB')
                scale = min(1.0, 1000 / im.width, 560 / im.height) if max(im.size) > 1000 else min(2.0, 1000 / im.width, 560 / im.height)
                im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))))
                tile.paste(im, (6, qy + 70))
            tiles.append(tile)
        sheet = Image.new('RGB', (width, 770 * len(tiles)), 'white')
        for i, t in enumerate(tiles):
            sheet.paste(t, (5, i * 770))
        path = out_dir / f'sheet_{s // per_sheet:02d}.png'
        sheet.save(path)
        paths.append(path)
    return paths


def sample_audit(rows, *, seed, quota=DIRECTION_QUOTA, per_material=2):
    """Unique (pseudo-item) redundant candidates with a direction, stratified by direction, at most `per_material` per material."""
    rng = random.Random(seed)
    pool = defaultdict(list)
    seen = set()
    for r in sorted(rows, key=lambda r: r['candidate_id']):
        if r['redundant'] == 1 and r['direction'] and r['pseudo_id'] not in seen:
            seen.add(r['pseudo_id'])
            pool[r['direction']].append(r)
    picked = []
    for direction, n in quota:
        items = pool[direction]
        rng.shuffle(items)
        used = defaultdict(int)
        for r in items:
            if used[r['material_id']] < per_material and sum(p['direction'] == direction for p in picked) < n:
                used[r['material_id']] += 1
                picked.append(r)
    rng.shuffle(picked)
    return picked


def render_sheets(items, materials, root, out_dir, per_sheet=4, width=1500):
    """Sheets of `per_sheet` items: id, material text, question, image. Returns written paths."""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.load_default(size=18)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in range(0, len(items), per_sheet):
        tiles = []
        for n, r in enumerate(items[s:s + per_sheet], start=s):
            m = materials[r['material_id']]
            with Image.open(root / m['image_path']) as im:
                im = im.convert('RGB')
                scale = min(1.0, 700 / im.width, 380 / im.height) if max(im.size) > 700 else min(2.0, 700 / im.width, 380 / im.height)
                im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))))
                tile = Image.new('RGB', (width // 2 - 10, 460), 'white')
                tile.paste(im, (5, 70 + (380 - im.height) // 2))
            d = ImageDraw.Draw(tile)
            d.text((5, 2), f'#{n}  TEXT: ' + ' '.join(textwrap.wrap(m['text'], 90))[:170], fill='black', font=font)
            d.text((5, 30), f'Q: ' + r['question'][:90], fill='blue', font=font)
            if len(r['question']) > 90:
                d.text((5, 50), r['question'][90:180], fill='blue', font=font)
            tiles.append(tile)
        sheet = Image.new('RGB', (width, 480 * ((len(tiles) + 1) // 2)), 'white')
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % 2) * (width // 2), (i // 2) * 480))
        path = out_dir / f'sheet_{s // per_sheet:02d}.png'
        sheet.save(path)
        paths.append(path)
    return paths


def audit_summary(reviews):
    """reviews: dicts with 'claimed' direction, 'reviewer_image_alone', 'reviewer_text_alone' (bool). Wilson bounds on agreement."""
    n = len(reviews)
    agree = sum(bool(r['reviewer_image_alone'] or r['reviewer_text_alone']) for r in reviews)
    direction = 0
    for r in reviews:
        claimed = {'image': (True, False), 'text': (False, True), 'both': (True, True)}[r['claimed']]
        direction += (bool(r['reviewer_image_alone']), bool(r['reviewer_text_alone'])) == claimed
    return {'n': n, 'agree_redundant': agree, 'agree_redundant_rate': agree / n if n else None, 'wilson_lower_95': wilson_lower(agree, n),
            'direction_agree': direction, 'direction_agree_rate': direction / n if n else None, 'direction_wilson_lower_95': wilson_lower(direction, n),
            'disagreements': [r['n'] for r in reviews if not (r['reviewer_image_alone'] or r['reviewer_text_alone'])]}


def wilson_interval(k, n, z=1.959963984540054):
    import math
    if not n:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def two_class_summary(reviews, *, p_label_redundant, p_label_nonredundant, seed=20261113):
    """AI audit of both label classes.

    Redundant-labelled items agree when at least one modality alone is sufficient; non-redundant-labelled
    items agree when neither modality alone is sufficient. Recall of the non-redundant label is estimated
    from the false-omission rate among redundant-labelled items and the label prevalence; its interval comes
    from resampling the two reviewed strata.
    """
    red = [r for r in reviews if r['audit_class'] == 'redundant']
    non = [r for r in reviews if r['audit_class'] == 'nonredundant']
    red_agree = sum(bool(r['reviewer_image_alone'] or r['reviewer_text_alone']) for r in red)
    non_agree = sum(not (r['reviewer_image_alone'] or r['reviewer_text_alone']) for r in non)
    prec_red = red_agree / len(red) if red else None
    prec_non = non_agree / len(non) if non else None
    q = 1 - prec_red if prec_red is not None else None          # P(truly non-redundant | labelled redundant)

    def recall_from(prec_r, prec_n):
        if prec_r is None or prec_n is None:
            return None
        true_non = p_label_nonredundant * prec_n + p_label_redundant * (1 - prec_r)
        return p_label_nonredundant * prec_n / true_non if true_non else None

    recall = recall_from(prec_red, prec_non)
    rng = random.Random(seed)
    samples = []
    for _ in range(1000):
        rb = [red[rng.randrange(len(red))] for _ in red] if red else []
        nb = [non[rng.randrange(len(non))] for _ in non] if non else []
        pr = sum(bool(r['reviewer_image_alone'] or r['reviewer_text_alone']) for r in rb) / len(rb) if rb else None
        pn = sum(not (r['reviewer_image_alone'] or r['reviewer_text_alone']) for r in nb) / len(nb) if nb else None
        value = recall_from(pr, pn)
        if value is not None:
            samples.append(value)
    samples.sort()
    interval = (samples[int(0.025 * len(samples))], samples[int(0.975 * len(samples)) - 1]) if samples else (None, None)
    return {'n_redundant': len(red), 'n_nonredundant': len(non),
            'redundant_agree': red_agree, 'redundant_precision': prec_red, 'redundant_precision_wilson_95': wilson_interval(red_agree, len(red)),
            'nonredundant_agree': non_agree, 'nonredundant_precision': prec_non, 'nonredundant_precision_wilson_95': wilson_interval(non_agree, len(non)),
            'false_omission_rate_among_redundant': q, 'nonredundant_recall': recall, 'nonredundant_recall_bootstrap_95': interval,
            'prevalence': {'p_label_redundant': p_label_redundant, 'p_label_nonredundant': p_label_nonredundant},
            'redundant_disagreements': [r['n'] for r in red if not (r['reviewer_image_alone'] or r['reviewer_text_alone'])],
            'nonredundant_disagreements': [r['n'] for r in non if (r['reviewer_image_alone'] or r['reviewer_text_alone'])]}


def prepare_two_class_audit(*, capture_run, eval_run, reference_run, project_root='.', output_root='outputs', seed=20261113):
    root = Path(project_root).resolve()
    cap = root / output_root / capture_run
    rows = read_jsonl(cap / f'labels_{eval_run}.jsonl')
    materials = {m['material_id']: m for m in read_jsonl(root / output_root / reference_run / 'materials.jsonl')}
    items = sample_two_class(rows, seed=seed)
    valid = [r for r in rows if r['valid'] == 1]
    p_red = sum(r['redundant'] == 1 for r in valid) / len(valid)
    p_non = sum(r['redundant'] == 0 for r in valid) / len(valid)
    write_jsonl(cap / 'audit_sample_two_class.jsonl',
                [{'n': n, 'candidate_id': r['candidate_id'], 'pseudo_id': r['pseudo_id'], 'material_id': r['material_id'],
                  'skill': r['skill'], 'question': r['question'], 'audit_class': r['audit_class'],
                  'direction': r['direction'] if r['audit_class'] == 'redundant' else None,
                  'p_label_redundant': p_red, 'p_label_nonredundant': p_non} for n, r in enumerate(items)])
    paths = render_two_class_sheets(items, materials, root, cap / 'audit_sheets_two_class')
    return {'items': len(items), 'redundant': sum(r['audit_class'] == 'redundant' for r in items),
            'nonredundant': sum(r['audit_class'] == 'nonredundant' for r in items),
            'sheets': [str(p.relative_to(root)) for p in paths]}


def prepare_audit(*, capture_run, eval_run, reference_run, project_root='.', output_root='outputs', seed=20261113):
    root = Path(project_root).resolve()
    cap = root / output_root / capture_run
    rows = read_jsonl(cap / f'labels_{eval_run}.jsonl')
    materials = {m['material_id']: m for m in read_jsonl(root / output_root / reference_run / 'materials.jsonl')}
    items = sample_audit(rows, seed=seed)
    write_jsonl(cap / 'audit_sample.jsonl', [{'n': n, 'candidate_id': r['candidate_id'], 'pseudo_id': r['pseudo_id'], 'material_id': r['material_id'],
                                              'skill': r['skill'], 'question': r['question'], 'claimed': r['direction']} for n, r in enumerate(items)])
    paths = render_sheets(items, materials, root, cap / 'audit_sheets')
    return {'items': len(items), 'sheets': [str(p.relative_to(root)) for p in paths]}


def finish_audit(*, capture_run, project_root='.', output_root='outputs'):
    cap = Path(project_root).resolve() / output_root / capture_run
    if (cap / 'audit_sample_two_class.jsonl').exists():
        sample = {r['n']: r for r in read_jsonl(cap / 'audit_sample_two_class.jsonl')}
        reviews = [{**sample[r['n']], **r} for r in read_jsonl(cap / 'audit_reviews_two_class.jsonl')]
        first = next(iter(sample.values()))
        result = two_class_summary(reviews, p_label_redundant=first['p_label_redundant'],
                                   p_label_nonredundant=first['p_label_nonredundant'])
        write_json(cap / 'audit_summary_two_class.json', {**result, 'reviewer': 'Claude (AI review, not independent human annotation)'})
        return result
    sample = {r['n']: r for r in read_jsonl(cap / 'audit_sample.jsonl')}
    reviews = [{**sample[r['n']], **r} for r in read_jsonl(cap / 'audit_reviews.jsonl')]
    result = audit_summary(reviews)
    write_json(cap / 'audit_summary.json', {**result, 'reviewer': 'Claude (AI review, not independent human annotation)'})
    return result
