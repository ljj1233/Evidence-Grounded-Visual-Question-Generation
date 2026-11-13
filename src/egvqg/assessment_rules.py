"""Shared knowledge contract and conservative deterministic proof checks."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from .records import sha256_file


def load_assessment_config(root, path):
    if path is None:
        return None
    config = json.loads((Path(root) / path).read_text())
    required = {'protocol_version', 'knowledge', 'analysis_template', 'verification_template',
                'analysis_max_tokens', 'verification_max_tokens'}
    if set(config) != required or any(not isinstance(config[k], str) or not config[k].strip()
                                    for k in required - {'analysis_max_tokens', 'verification_max_tokens'}):
        raise ValueError('invalid assessment config')
    if any(type(config[k]) is not int or not 256 <= config[k] <= 16384
           for k in ('analysis_max_tokens', 'verification_max_tokens')):
        raise ValueError('invalid assessment token budget')
    return config


def check_frozen_sources(sources, out, *, completed):
    """Only completed historical code can use its exact archived bytes, read-only."""
    for path, expected in sources.items():
        path = Path(path)
        if path.exists() and sha256_file(path) == expected:
            continue
        snapshot = Path(out) / 'code_snapshots' / path.name
        if (completed and path.suffix == '.py' and path.parent.name == 'egvqg'
                and snapshot.exists() and sha256_file(snapshot) == expected):
            continue
        raise ValueError('frozen source changed: ' + str(path))


def _normalized(value):
    return re.sub(r'\W+', '', unicodedata.normalize('NFKC', value).casefold())


def choice_conflicts(proof, choices):
    """Reject explicit letter/literal option contradictions; never guess translations."""
    if not choices or not isinstance(proof, dict):
        return []
    answers = [proof[k] for k in ('answer', 'answer_a', 'answer_b') if isinstance(proof.get(k), str)]
    answers += [a for a in proof.get('recomputed_answers', []) if isinstance(a, str)]
    conflicts = []
    for answer in answers:
        match = re.match(r'^\s*(?:Option\s+)?([A-Z])\s*[.、:：)\-]\s*(.+?)\s*$', answer, re.I)
        if not match:
            continue
        index = ord(match[1].upper()) - ord('A')
        text = _normalized(match[2])
        matches = [i for i, choice in enumerate(choices) if _normalized(choice) == text]
        if index >= len(choices) or (matches and index not in matches):
            conflicts.append({'answer': answer, 'letter_index': index, 'literal_matches': matches})
    return conflicts
