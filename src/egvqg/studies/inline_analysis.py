"""Inline analysis: does a generated question repeat a number or rule from the material text?

Synthetic runs use programmatic question labels as a text-redundancy proxy; MMQA runs use the existing
consensus sufficiency labels. No judge is called here.

Redundancy follows `docs/necessity_formal_argument.md`: the **text is redundant** when the image alone
suffices (`image_suff` / `both_suff` verdicts), and the **image is redundant** when the text alone suffices
(`text_suff` / `both_suff`). Earlier versions of this module read `text_suff` as text redundancy, which
inverted the reported quantity; both sides are now computed and reported explicitly.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from ..core.records import read_jsonl, stable_id, utc_now, write_json, write_jsonl

STOPWORDS = {
    'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'from', 'by', 'with', 'and', 'or', 'but',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'as', 'that', 'this', 'these', 'those', 'it',
    'its', 'his', 'her', 'their', 'he', 'she', 'they', 'which', 'what', 'who', 'whom', 'whose', 'when',
    'where', 'why', 'how', 'not', 'no', 'do', 'does', 'did', 'has', 'have', 'had', 'can', 'could',
    'would', 'should', 'may', 'might', 'will', 'shall', 'than', 'then', 'there', 'here', 'also',
}

RULE_CUES = (
    'at least', 'minimum value', 'threshold', 'only items', 'that count', 'from', 'between',
    'more', 'less', 'earlier', 'later', 'larger', 'smaller', 'greater', 'higher', 'lower',
    'first', 'last', 'longer', 'shorter', 'older', 'younger', 'won', 'award', 'before', 'after',
)


_ONES = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
         'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen')
_TENS = ('', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety')


def int_to_words(value):
    if value < 20:
        return _ONES[value]
    if value < 100:
        tens, ones = divmod(value, 10)
        return _TENS[tens] + ('' if ones == 0 else '-' + _ONES[ones])
    return None


def number_tokens(text):
    return [match.group(0) for match in re.finditer(r'(?<![\d.])\d+(?:\.\d+)?(?![\d.])', text)]


def contains_number(question, text, min_digits=1):
    matches = []
    for token in number_tokens(text):
        digits = token.replace('.', '')
        if len(digits) < min_digits:
            continue
        pattern = r'(?<![\d.])' + re.escape(token) + r'(?![\d.])'
        if re.search(pattern, question):
            matches.append(token)
            continue
        if '.' not in token:
            word = int_to_words(int(token))
            if word and re.search(r'(?<![a-z])' + re.escape(word) + r'(?![a-z])', question.lower()):
                matches.append(token)
    return matches


def _words(text):
    return [w.lower() for w in re.findall(r"[A-Za-z0-9']+", text)]


def _ngrams(words, n):
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def lexical_inline(question, text, n=5):
    q_tokens = _words(question)
    t_tokens = _words(text)
    if len(q_tokens) < n or len(t_tokens) < n:
        return None
    for gram in _ngrams(t_tokens, n) & _ngrams(q_tokens, n):
        if any(token not in STOPWORDS for token in gram):
            return ' '.join(gram)
    return None


def rule_inline(question, text, category=None):
    q = question.lower()
    if category == 'bridging':
        return 'alpha' if 'alpha' in q else None
    if category == 'interval':
        if ('from' in q and 'to' in q) or 'between' in q:
            return 'interval wording'
        return None
    if any(cue in q for cue in ('at least', 'minimum value', 'threshold', 'only items', 'that count')):
        return 'threshold wording'
    for cue in RULE_CUES:
        if cue in text.lower() and re.search(r'(?<![a-z])' + re.escape(cue) + r'(?![a-z])', q):
            return cue
    return None


def inline_features(question, text, *, category=None, min_digits=1, lexical_n=5, use_lexical=True):
    numbers = contains_number(question, text, min_digits=min_digits)
    rule = rule_inline(question, text, category=category)
    lexical = lexical_inline(question, text, n=lexical_n) if use_lexical else None
    return {'inline_number': bool(numbers), 'numbers': numbers, 'inline_rule': bool(rule),
            'rule': rule, 'inline_lexical': bool(lexical), 'lexical': lexical,
            'inline_any': bool(numbers or rule or lexical)}


def _wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _rate(rows, predicate):
    values = [predicate(row) for row in rows]
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {'n': len(values), 'k': sum(values), 'rate': sum(values) / len(values),
            'wilson_95': _wilson(sum(values), len(values))}


def _load_json(path):
    return json.loads(Path(path).read_text())


def analyze_synthetic_inline(*, run_id, project_root='.', output_root='outputs'):
    root = Path(project_root).resolve()
    out = root / output_root / run_id
    rows = read_jsonl(out / 'generated.jsonl')
    if not rows:
        raise ValueError(f'no generated questions in {out}')
    manifest = _load_json(out / 'manifest_generation.json')
    from .synthetic_composition import load_composition_materials, classify_generated_question
    materials = {m['material_id']: m for m in load_composition_materials(root, manifest['materials_path'])}
    model = manifest['model_identity']['model']
    enriched = []
    for row in rows:
        material = materials[row['material_id']]
        features = inline_features(row['question'], material['text'], category=row['category'],
                                    min_digits=1, use_lexical=False)
        label = classify_generated_question(row['question'], material)
        text_redundant = label == 'single_image'
        enriched.append({**row, **features, 'programmatic_label': label,
                         'text_redundant_proxy': text_redundant if label != 'other' else None})
    by_category = {}
    for category in sorted({row['category'] for row in rows}):
        subset = [row for row in enriched if row['category'] == category]
        inline = [row for row in subset if row['inline_any']]
        non_inline = [row for row in subset if not row['inline_any']]
        by_category[category] = {
            'n': len(subset),
            'inline_number_rate': _rate(subset, lambda r: r['inline_number']),
            'inline_rule_rate': _rate(subset, lambda r: r['inline_rule']),
            'inline_any_rate': _rate(subset, lambda r: r['inline_any']),
            'text_redundancy_proxy': _rate(subset, lambda r: r['text_redundant_proxy']),
            'text_redundancy_inline': _rate(inline, lambda r: r['text_redundant_proxy']),
            'text_redundancy_non_inline': _rate(non_inline, lambda r: r['text_redundant_proxy']),
            'labels': dict(Counter(row['programmatic_label'] for row in subset)),
        }
    report = {'run_id': run_id, 'dataset': 'synthetic', 'model': model, 'n': len(rows),
              'by_category': by_category}
    write_json(out / 'inline_report.json', report)
    write_jsonl(out / 'inline_questions.jsonl', [{
        'material_id': row['material_id'], 'category': row['category'], 'question': row['question'],
        'inline_number': row['inline_number'], 'numbers': row['numbers'], 'inline_rule': row['inline_rule'],
        'rule': row['rule'], 'inline_any': row['inline_any'], 'programmatic_label': row['programmatic_label'],
        'text_redundant_proxy': row['text_redundant_proxy'],
    } for row in enriched])
    return report


def analyze_mmqa_inline(*, run_id, candidates_path, materials_path, project_root='.', output_root='outputs',
                        min_digits=2):
    root = Path(project_root).resolve()
    out = root / output_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(root / candidates_path)
    materials = {m['material_id']: m for m in read_jsonl(root / materials_path)}
    enriched = []
    for row in rows:
        material = materials.get(row['material_id'])
        if material is None:
            continue
        verdict = (row.get('label_consensus') or {}).get('verdict')
        decided = verdict is not None and verdict != 'undetermined'
        text_redundant = None if not decided else verdict in ('image_suff', 'both_suff')
        image_redundant = None if not decided else verdict in ('text_suff', 'both_suff')
        features = inline_features(row['question'], material['text'], min_digits=min_digits, use_lexical=True)
        enriched.append({**row, **features, 'verdict': verdict, 'decided': decided,
                         'text_redundant': text_redundant, 'image_redundant': image_redundant})
    conditions = {}
    for condition in sorted({row.get('condition', 'unknown') for row in enriched}):
        subset = [row for row in enriched if row.get('condition', 'unknown') == condition]
        inline = [row for row in subset if row['inline_any']]
        non_inline = [row for row in subset if not row['inline_any']]
        conditions[condition] = {
            'n': len(subset),
            'decided': sum(1 for row in subset if row['decided']),
            'inline_number_rate': _rate(subset, lambda r: r['inline_number']),
            'inline_lexical_rate': _rate(subset, lambda r: r['inline_lexical']),
            'inline_any_rate': _rate(subset, lambda r: r['inline_any']),
            'text_redundancy': _rate(subset, lambda r: r['text_redundant']),
            'text_redundancy_inline': _rate(inline, lambda r: r['text_redundant']),
            'text_redundancy_non_inline': _rate(non_inline, lambda r: r['text_redundant']),
            'image_redundancy': _rate(subset, lambda r: r['image_redundant']),
            'image_redundancy_inline': _rate(inline, lambda r: r['image_redundant']),
            'image_redundancy_non_inline': _rate(non_inline, lambda r: r['image_redundant']),
            'verdicts': dict(Counter(row['verdict'] for row in subset)),
        }
    report = {'run_id': run_id, 'dataset': 'mmqa', 'candidates_path': candidates_path,
              'materials_path': materials_path, 'min_digits': min_digits, 'conditions': conditions}
    write_json(out / 'inline_report.json', report)
    write_jsonl(out / 'inline_questions.jsonl', [{
        'cand_id': row['cand_id'], 'material_id': row['material_id'], 'condition': row.get('condition'),
        'question': row['question'], 'inline_number': row['inline_number'], 'numbers': row['numbers'],
        'inline_lexical': row['inline_lexical'], 'lexical': row['lexical'], 'inline_any': row['inline_any'],
        'verdict': row['verdict'], 'text_redundant': row['text_redundant'],
        'image_redundant': row['image_redundant'],
    } for row in enriched])
    return report


def combine_inline_reports(*, run_id, synthetic_reports, mmqa_reports, project_root='.', output_root='outputs'):
    root = Path(project_root).resolve()
    out = root / output_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    report = {'created_at': utc_now(), 'synthetic': synthetic_reports, 'mmqa': mmqa_reports}
    write_json(out / 'inline_combined.json', report)
    lines = ['# Inline rate and text redundancy', '',
             'Inline = the question repeats a material number, a rule phrase, or (MMQA) a 5-word lexical span '
             'from the text. Synthetic text redundancy is the programmatic `single_image` proxy. MMQA '
             'redundancy follows the formal definition: text redundant = `image_suff`/`both_suff` (image alone '
             'suffices), image redundant = `text_suff`/`both_suff` (text alone suffices). No judge is called.', '',
             '## Synthetic', '',
             '| run | model | category | n | inline number | inline rule | inline any | text redundant | redundant \\| inline | redundant \\| non-inline |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for label, item in synthetic_reports.items():
        for category, values in item['by_category'].items():
            def fmt(key):
                value = values.get(key)
                if not value:
                    return 'n/a'
                return f'{value["rate"]:.3f} ({value["k"]}/{value["n"]})'
            lines.append(f'| {label} | {item["model"]} | {category} | {values["n"]} | {fmt("inline_number_rate")} | '
                         f'{fmt("inline_rule_rate")} | {fmt("inline_any_rate")} | {fmt("text_redundancy_proxy")} | '
                         f'{fmt("text_redundancy_inline")} | {fmt("text_redundancy_non_inline")} |')
    lines += ['', '## MMQA (existing labels)', '',
              '| condition | n | decided | inline number | inline lexical | inline any | text redundant | redundant \\| inline | redundant \\| non-inline | image redundant | redundant \\| inline | redundant \\| non-inline |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label, item in mmqa_reports.items():
        for condition, values in item['conditions'].items():
            def fmt(key):
                value = values.get(key)
                if not value:
                    return 'n/a'
                return f'{value["rate"]:.3f} ({value["k"]}/{value["n"]})'
            lines.append(f'| {label}/{condition} | {values["n"]} | {values["decided"]} | '
                         f'{fmt("inline_number_rate")} | {fmt("inline_lexical_rate")} | {fmt("inline_any_rate")} | '
                         f'{fmt("text_redundancy")} | {fmt("text_redundancy_inline")} | {fmt("text_redundancy_non_inline")} | '
                         f'{fmt("image_redundancy")} | {fmt("image_redundancy_inline")} | '
                         f'{fmt("image_redundancy_non_inline")} |')
    lines.append('')
    (out / 'inline_report.md').write_text('\n'.join(lines))
    return report
