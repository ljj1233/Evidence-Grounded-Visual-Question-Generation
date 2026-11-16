"""Helpers of the baseline and evaluation workflow (the remaining part of an older module; everything else was removed)."""
from __future__ import annotations



CHECKS = ('presupposition_supported', 'answer_supported', 'scope_clear')

def _decision(item):
    if not isinstance(item, dict) or 'value' not in item or item.get('value') not in (0, 1, None) or (
            item.get('value') is not None and type(item['value']) is not int):
        raise ValueError('review decision must be 0/1/null')
    if not isinstance(item.get('reason'), str) or not item['reason'].strip():
        raise ValueError('review decision needs a reason')
    return item['value']
