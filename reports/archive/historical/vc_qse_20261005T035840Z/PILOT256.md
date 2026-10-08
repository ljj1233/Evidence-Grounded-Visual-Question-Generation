# PILOT 256 — C0_DIRECT4 vs VC-QSE_PILOT (official metric)

Run: `reports/vc_qse_20261005T035840Z/pilot256/` · seed 42 · GPU 0
Images: development1000 sorted by integer image_id, first 256. No final_holdout.

Evaluator: **single pass** SeeQ single-question rubric (`visual_evidence_coverage`,
model `gpt-5.6-luna`, variant documented — same model as the archived v2 run) and
**single pass** GPT-4o visual support. No prompt/threshold change, no repeated
query, no checkpoint re-selection. SeeQ 256/256 ok per system; support 256/256
valid per system (0 parser failures, 0 API failures).

## Coverage and support

| metric | C0_DIRECT4 | VC-QSE_PILOT | Δ (VC−C0) |
|---|---:|---:|---:|
| Coverage (mean `visual_evidence_coverage`, 0–5) | **1.6797** | **1.5430** | **−0.1367** |
| Supported (GPT-4o) | 0.7578 | **0.8164** | +0.0586 |
| Unsupported | 0.0820 | **0.0664** | −0.0156 |
| Uncertain | 0.1602 | 0.1172 | −0.0430 |
| duplicate rate (final) | 0.0898 | 0.1016 | +0.0117 |
| mean words | 5.621 | 5.645 | +0.023 |
| fallback rate | 0.000 | 0.0820 | +0.082 |

Coverage paired per-image: mean **−0.1367**, bootstrap 95% CI **[-0.293, +0.012]**
(10000 resamples, seed 42), **47 up / 63 down / 146 tied**.

## Stratification

Fallback vs non-fallback:

| stratum | n | mean coverage Δ |
|---|---:|---:|
| VC candidate (no fallback) | 235 | −0.1362 |
| Direct-4 fallback | 21 | −0.1429 |

Concept type of the selected final question:

| type | n | mean coverage Δ |
|---|---:|---:|
| object | 89 | −0.2022 |
| attribute | 41 | −0.3415 |
| relation | 43 | −0.1628 |
| **scene** | 62 | **+0.1129** |

Length dependence: corr(Δwords, Δcoverage) = 0.079.

## Gate

| # | criterion | value | result |
|---|---|---|---|
| 1 | Coverage point estimate > C0 | 1.543 < 1.680 | **FAIL** |
| 2 | Supported drop ≤ 0.01 | +0.0586 | PASS |
| 3 | Unsupported rise ≤ 0.01 | −0.0156 | PASS |
| 4 | fallback rate ≤ 0.10 | 0.082 | PASS |
| 5 | gain not only from <5% images | negative gain | **FAIL** |
| 6 | concept candidates not mass-duplicated | candidate dup 0.210 | PASS |
| 7 | ≥2 concept types with valid candidates | 4 types | PASS |
| 8 | not length-driven | |corr| 0.079 | PASS |

**GATE = FAIL.** VC-QSE does **not** improve Coverage on the official metric
(point estimate lower, CI includes 0). Local Flat support improves, but that is
not the Coverage gate.

## Decision

- Do **not** generate a concept-conditioned train pool.
- Do **not** train VC-QSE.
- Do **not** run RL.
- Record as a first, honest pilot result; only the `scene` concept stratum shows a
  positive (non-significant) Coverage delta and is the only signal worth a
  follow-up.

JSON: `gate.json`; per-image cases: `selected/failure_cases_merged.jsonl` (156).
