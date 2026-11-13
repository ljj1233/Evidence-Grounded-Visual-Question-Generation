"""Verifier failures remain unresolved/rejected and never rewrite source results."""

import pytest

from egvqg.evidence_verification import CHECKS, verification_gate


@pytest.mark.parametrize('change,expected',[(None,'accept'),('well_posed','reject'),('premises_supported','unresolved'),('missing','unresolved')])
def test_verifier_is_fail_closed(change,expected):
    output={**dict.fromkeys(CHECKS,'pass'),'reason':'fixture','recomputed_answers':['A','B']}
    if change=='well_posed':output[change]='fail'
    elif change=='premises_supported':output[change]='uncertain'
    elif change=='missing':output.pop('conclusion_checked')
    assert verification_gate(output)==expected


def test_explicit_failure_with_empty_recomputation_is_rejected_in_v2_only():
    output={**dict.fromkeys(CHECKS,'pass'),'premises_supported':'fail','reason':'unsupported premise','recomputed_answers':[]}
    assert verification_gate(output)=='unresolved'  # unchanged historical semantics
    assert verification_gate(output,strict=True)=='reject'

