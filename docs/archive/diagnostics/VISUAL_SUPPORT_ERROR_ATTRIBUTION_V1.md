# Phase 2.1E — Flat Verifier Error Attribution and Production Readiness

## Scope and freeze boundary

This analysis uses only the Phase 2.1 `dev` (600) and `val` (200) splits.
The Phase 2.1 test set, controlled v1/v2 benchmarks, final holdout, GQA,
SFT, and GRPO were not used for analysis or tuning. The Flat prompt,
requirement schema, aggregation, thresholds, and label mapping were unchanged.
The finalized attribution summary is hashed in
`outputs/visual_support_error_attribution_v1/error_attribution_report.sha256`:

`a99a45d8badbad9d31317e5b2576ae89406a5106f01f28ae52d4d3f028fb7c0d`

## Decision traces and error groups

One trace was reconstructed for each of the 800 dev/val questions. Each trace
contains the image/question, GPT pseudo-label, extracted requirement type/text,
Flat local prediction and raw output, Flat dimensions, final decision, and the
Holistic decision. A malformed Flat response is a parser failure, never an
unsupported label.

| split | false rejection | false acceptance | Flat uncertain | correct supported | correct unsupported | parse failure |
|---|---:|---:|---:|---:|---:|---:|
| dev (600) | 108 | 31 | 30 | 277 | 60 | 12 |
| val (200) | 45 | 9 | 6 | 79 | 30 | 6 |

The audit subset contains 176 cases: 100 false rejections, 40 false
acceptances, and 36 uncertain cases. It is deterministic and stratified across
dev/val. GPT-4o audited each extracted requirement on necessity/extraction
correctness and, for necessary requirements, local visual-evidence correctness;
it also classified the final aggregation outcome. All 176 responses were
returned successfully.

## Root-cause attribution

The 100 audited false rejections were attributed as follows:

| cause | count | percentage |
|---|---:|---:|
| local visual verification error | 74 | 74% |
| extra extracted requirement | 26 | 26% |
| aggregation too strict | 0 | 0% |
| parser failure | 0 | 0% |
| other | 0 | 0% |

The local-verification errors were entity 39, target 16, attribute 12,
relation 4, and reference 3. False acceptances were also dominated by local
verification errors (27/40), with 8 uncertain and 4 other audit outcomes and
one extra-requirement attribution. The evidence does not identify a single
dominant extraction or aggregation defect.

## Unsupported-question detection

Flat recall was generally higher than Holistic recall for the visible failure
types, especially ambiguous reference and unobservable target. Dev/val details
are in `unsupported_failure_analysis.json`; examples include:

| failure type | dev Holistic / Flat recall | val Holistic / Flat recall |
|---|---:|---:|
| missing entity | 0.333 / 0.667 | 0.333 / 0.667 |
| wrong attribute | 0.857 / 0.857 | 1.000 / 1.000 |
| ambiguous reference | 0.118 / 0.471 | 0.167 / 0.667 |
| unobservable target | 0.450 / 0.550 | 0.409 / 0.818 |
| external knowledge | 0.444 / 0.556 | 0.600 / 0.400 |

Small per-type counts, especially wrong relation in val, make these descriptive
rather than tuning evidence.

## Production readiness

Across 800 questions, requirement extraction succeeded on 97.75%; malformed
outputs were 2.25%. Flat decisions were fully scored (non-uncertain) on
93.25%, partial/uncertain on 4.50%, and failed on 2.25%. Valid questions had
3.744 extracted requirements on average and used one local VLM call per
question. The Phase 2.1 harness emits extraction and local verification in one
call, so their individual times are not identifiable; measured joint Flat
throughput was 1.604 s/question at batch size 8. A 10,000-image workload with
four rollouts/image is therefore estimated at 40,000 calls and 17.82 GPU-hours
on the same setup (excluding queueing and model startup).

The GPT audit used direct HTTP with proxy variables disabled, max concurrency
32, `max_tokens=900`, resume/incremental writes, and zero failed requests.
Full fingerprints and timing are in `prompt_manifest.json` and
`runtime_report.json`.

## Interpretation and correction status

Flat should currently be treated as an explicit visual-violation detector, not
as a reliable full support classifier. Its controlled-v2 behavior preserved
only 0.875 of supported questions while detecting 0.980 of unsupported ones;
the attribution shows that the supported-question loss is mainly local visual
verification error rather than deterministic aggregation. This interpretation
does not select a reward formula.

The correction proposal is `NEED_MORE_EVIDENCE`; no verifier code, prompt,
aggregation, threshold, or label mapping was changed. A narrowly scoped fix
should be considered only after a larger, independently audited error sample
isolates one systematic cause.
