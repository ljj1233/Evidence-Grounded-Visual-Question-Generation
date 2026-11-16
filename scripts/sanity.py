"""Single CPU sanity check for the current method (synthetic data, seconds).

Checks: (1) the paired bootstrap covers 0 on null paired data; (2) the gate's precision expression rises above the
base rate exactly when sensitivity exceeds luck; (3) answer-equivalence does not merge two different numbers and does
merge a paraphrase; (4) Wilson lower bounds stay below the point estimate and tighten with n.
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

from egvqg.studies.self_verification import _answers_match  # noqa: E402
from egvqg.core.stats import paired_bootstrap, wilson, wilson_lower  # noqa: E402


def check(name, ok, detail):
    print(f'[{"PASS" if ok else "FAIL"}] {name}: {detail}')
    return bool(ok)


def precision(pi, luck, sens):
    keep = pi * (1 - luck)
    return keep / (keep + (1 - pi) * (1 - sens))


def main():
    ok = True

    rng = np.random.default_rng(11)
    groups = np.repeat(np.arange(40), 5)
    null = rng.normal(size=len(groups)) * 0.5
    boot = paired_bootstrap(lambda idx: float(null[idx].mean()), groups, n=2000)
    ok &= check('paired bootstrap covers 0 on null data', boot['ci'][0] <= 0 <= boot['ci'][1],
                f'point={boot["point"]:+.3f} CI=[{boot["ci"][0]:+.3f}, {boot["ci"][1]:+.3f}]')

    pi = 0.15
    below = precision(pi, luck=0.30, sens=0.20)
    above = precision(pi, luck=0.10, sens=0.40)
    ok &= check('gate beats the base rate iff sensitivity exceeds luck',
                below < pi < above and precision(pi, 0.2, 0.2) == pi,
                f's<l: {below:.3f} < {pi} < s>l: {above:.3f}')

    ok &= check('different numbers never agree', not _answers_match('35', '53') and not _answers_match('at least 6', 'at least 7'),
                'numeric conflicts rejected')
    ok &= check('a paraphrase with the same content agrees', _answers_match('melzer', 'she was formerly melzer'),
                'containment matched')

    loose = wilson_lower(30, 60)
    tight = wilson_lower(3000, 6000)
    point = wilson(3000, 6000)['p']
    ok &= check('Wilson lower bound tightens with n and stays below the point', 0 < loose < tight < point,
                f'n=60 {loose:.3f} < n=6000 {tight:.3f} < p {point:.3f}')

    print('\nSANITY', 'PASSED' if ok else 'FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
