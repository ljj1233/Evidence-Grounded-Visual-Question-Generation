"""Shared knowledge contract and conservative deterministic proof checks."""
from __future__ import annotations

import json
from pathlib import Path



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

