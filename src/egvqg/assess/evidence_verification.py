"""A frozen sidecar verifier; it never changes source necessity labels or proofs."""
from __future__ import annotations



TEMPLATE='verify-necessity-evidence-v1'
CHECKS=('question_preserved','retained_preserved','well_posed','premises_supported','conclusion_checked')


def verification_gate(output, *, strict=False):
    if not isinstance(output,dict) or any(output.get(k) not in ('pass','fail','uncertain') for k in CHECKS):
        return 'unresolved'
    if not isinstance(output.get('reason'),str) or not output['reason'].strip():return 'unresolved'
    if strict and any(output[k]=='fail' for k in CHECKS):return 'reject'
    if not isinstance(output.get('recomputed_answers'),list) or not output['recomputed_answers']:return 'unresolved'
    if any(output[k]=='fail' for k in CHECKS):return 'reject'
    return 'accept' if all(output[k]=='pass' for k in CHECKS) else 'unresolved'

