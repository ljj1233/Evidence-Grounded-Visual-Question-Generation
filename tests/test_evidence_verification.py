"""Verifier failures remain unresolved/rejected and never rewrite source results."""

import pytest

from egvqg.evidence_verification import CHECKS, verification_gate
from egvqg.assessment_rules import choice_conflicts, check_frozen_sources
from egvqg.records import sha256_file


@pytest.mark.parametrize('change,expected',[(None,'accept'),('well_posed','reject'),('premises_supported','unresolved'),('missing','unresolved')])
def test_verifier_is_fail_closed(change,expected):
    output={**dict.fromkeys(CHECKS,'pass'),'reason':'fixture','recomputed_answers':['A','B']}
    if change=='well_posed':output[change]='fail'
    elif change=='premises_supported':output[change]='uncertain'
    elif change=='missing':output.pop('conclusion_checked')
    assert verification_gate(output)==expected


@pytest.mark.parametrize('answer,bad', [('B. beaver',True),('C. beaver',False),('B. silver maple',False),
                                      ('海狸',False),('C',False),('D. beaver',True)])
def test_choice_letter_text_binding(answer,bad):
    assert bool(choice_conflicts({'answer':answer},['gray fox','silver maple','beaver']))==bad


def test_explicit_failure_with_empty_recomputation_is_rejected_in_v2_only():
    output={**dict.fromkeys(CHECKS,'pass'),'premises_supported':'fail','reason':'unsupported premise','recomputed_answers':[]}
    assert verification_gate(output)=='unresolved'  # unchanged historical semantics
    assert verification_gate(output,strict=True)=='reject'


def test_archived_code_only_allows_completed_readonly_history(tmp_path):
    code=tmp_path/'src/egvqg/runtime.py';code.parent.mkdir(parents=True);code.write_text('old')
    expected=sha256_file(code);out=tmp_path/'run';(out/'code_snapshots').mkdir(parents=True)
    (out/'code_snapshots/runtime.py').write_text('old');code.write_text('new')
    check_frozen_sources({str(code):expected},out,completed=True)
    with pytest.raises(ValueError,match='source changed'):
        check_frozen_sources({str(code):expected},out,completed=False)
    data=tmp_path/'image.png';data.write_bytes(b'old');h=sha256_file(data);data.write_bytes(b'new')
    with pytest.raises(ValueError,match='source changed'):
        check_frozen_sources({str(data):h},out,completed=True)

