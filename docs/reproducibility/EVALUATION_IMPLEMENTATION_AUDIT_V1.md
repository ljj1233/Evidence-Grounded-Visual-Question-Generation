# EVALUATION IMPLEMENTATION AUDIT v1

Status: complete. (Post DPO-removal note: DPO/preference data and all DPO-system
predictions/results were deleted outright; the `archive/evaluation_legacy/` and
`outputs/archive/evaluation_legacy/` paths referenced below no longer exist.
Aggregated audit metadata remains in `outputs/evaluation/evaluation_audit_v1/`.)

This audit traces every active reported metric through
`definition -> implementation -> raw inputs -> aggregation -> report field`, and
consolidates the evaluation subsystem. No API or GPU re-evaluation was run.

- Machine-readable findings: `outputs/evaluation/evaluation_audit_v1/audit_findings.json`
- Metric registry: `outputs/evaluation/evaluation_audit_v1/metric_registry.json`
- Differential tests: `outputs/evaluation/evaluation_audit_v1/differential_tests.json`
- Golden fixtures: `outputs/evaluation/evaluation_audit_v1/golden_fixture_results.json`
- Data integrity: `outputs/evaluation/evaluation_audit_v1/data_integrity_report.json`
- Old vs recomputed: `outputs/evaluation/evaluation_audit_v1/old_vs_recomputed.json`
- Lineage: `outputs/evaluation/evaluation_audit_v1/metric_lineage.json`
- Provenance/hashes: `outputs/evaluation/evaluation_audit_v1/environment.json`
- Exact judge prompts + hashes: `outputs/evaluation/evaluation_audit_v1/judge_prompts.json`
- Cleanup manifest: `outputs/evaluation/evaluation_audit_v1/cleanup_manifest.json`

## 1. Conformance summary

| metric | declared vs implemented | differential | status |
|---|---|---|---|
| BLEU-4 corpus | corpus-level confirmed | matches coco-caption when all orders present; diverges on zero-count orders (method-1 vs additive) | VERIFIED (smoothing variant documented) |
| CIDEr | official `pycocoevalcap`, single call | repeated-call diff 0.0 | VERIFIED |
| Distinct-2 | corpus-level confirmed | unit test | VERIFIED |
| Normalized excess duplicate rate | variant A confirmed | unit test `[A,A,B] -> 1/3` | VERIFIED |
| Q-BLEU4-variant | reimplementation | official package absent -> BLOCKED | VARIANT_DOCUMENTED |
| CLIPScore | raw-question adaptation | code inspection | VARIANT_DOCUMENTED |
| CLIP retrieval | 1-based rank, ties favour source | structural checks | VERIFIED |

The active canonical CLI (`scripts/vvqg/evaluate.py`) reproduces the historical
vanilla numbers exactly:

| value | historical (`METRIC_BAKEOFF_V1` Table A) | recomputed (active CLI) |
|---|---:|---:|
| corpus BLEU-4 | 0.21871301674208266 | 0.21871301674208266 |
| CIDEr | 0.3561583660243859 | 0.3561583660243859 |
| Q-BLEU4 (variant) | 0.21303091383114028 | 0.21303091383114028 |
| Distinct-2 | 0.209721 | 0.209721 |
| Dup rate | 0.291 | 0.291 |

## 2. Golden fixtures (K=4, hand-computed)

Fixture records `(source, applicable_negatives)`:
`(1,0) (1,1) (1,4) (0,2) (0,0)`.

| quantity | hand value | module value | pass |
|---|---:|---:|---|
| source_applicable_rate | 3/5 | 0.6 | yes |
| any_negative_applicable_rate | 3/5 | 0.6 | yes |
| mean_negative_fraction | 7/20 | 0.35 | yes |
| conditional_any_negative_rate | 2/3 | 0.666667 | yes (tol 1e-6) |
| conditional_mean_negative_fraction | 5/12 | not exposed | n/a |
| source_unique_rate | 1/5 | not exposed | n/a |

The independent reference computation (`audit/reference_metrics.py:k4_rates`,
written separately from the module under test) matches the hand values exactly.
The module rounds public outputs to 6 decimals, hence tolerance 1e-6.
`conditional_mean_negative_fraction` and `source_unique_rate` were **not
reported by the 1+4 protocol**; they are computed here independently and
recorded, but the historical reports are not overwritten.

Edge cases checked: all-applicable, none-applicable, zero source-applicable
denominator (`null`), malformed/unknown judgments excluded, candidate index
validity, duplicate candidate ids (structurally absent), out-of-range indices
rejected. See `candidate_identity_audit.json` and
`golden_fixture_results.json`.

## 3. Lineage of historical 1+4 fields

| old field | actual formula | verified name |
|---|---|---|
| Source Applicable | `sum(a_i)/N` | `source_applicable_rate` |
| Hard-Neg Applicable | `sum(m_i>0)/N` | `any_negative_applicable_rate` |
| Pairwise Portability | `sum(a_i AND m_i>0)/sum(a_i)` | `conditional_any_negative_rate` |
| Mean neg fraction | `sum(m_i)/(K*N)` | `mean_negative_fraction` |
| (not exposed) | `sum(a_i*m_i)/(K*sum(a_i))` | `conditional_mean_negative_fraction` |
| Source Unique (2.2B) | `sum(a_i AND m_i==0)/N` | `source_unique_rate` |

Important: "Pairwise Portability" is the **conditional any-negative** rate
(`m_i > 0`), not the conditional **mean** fraction. This distinction was not
explicit in the historical report and is now named.

## 4. Old vs recomputed

All reported historical numbers were recomputed from preserved raw artifacts and
match exactly:

- metric bake-off (`metric_bakeoff_v1`): BLEU-4 corpus, Q-BLEU4, Distinct-2,
  duplicate rate, CIDEr mean — all pass.
- 1+4 specificity: source/any-negative/mean-fraction/conditional-any — all pass.
- 2.2B setwise: source-applicable, source-unique, source-best, unique-source-best,
  portability — all pass.

Details: `outputs/evaluation/evaluation_audit_v1/old_vs_recomputed.json`.

## 5. Joins, caches, statistics

- Join keys are explicit and stable (`system`, `image_id`, and question
  identity); no line-order joins. `data_integrity_report.json` shows
  0 duplicate `(system, image_id)` keys and 0 malformed records.
- Cache identity for judges is content-addressed: `question | ordered candidate
  image ids | judge model | prompt version`. System name is provenance, not
  identity (questions differ per system).
- Validation and reorder/swap calls are stored separately and excluded from the
  main denominators.
- Conditional statistics recompute inside the bootstrap; corpus BLEU is never
  approximated by mean sentence BLEU; CIDEr IDF policy is documented.

## 6. Historical GPT dependence inconsistency

The hypothesized artifact `mean=4.943, P(dependence>=4)=0.948` does **not**
exist anywhere in the repository. Recomputed from `gpt4o_judge.jsonl` joined to
the prediction files:

- `mdpo`: mean `4.943`, `P(>=4)=1.0` (histogram 4:57, 5:943) — internally consistent.
- all 5000 records: mean `4.9432`, `P(>=4)=0.9996`.

If `P(>=4)=0.948` were true for integer 1–5 values, the maximum possible mean is
`0.948*5 + 0.052*3 = 4.896 < 4.943`, so the pair is arithmetically impossible.
Conclusion: reporting error / not reproducible from current evidence; the
current artifacts are consistent. No historical report was edited.

## 7. Judge rubric

Findings classified (see `audit_findings.json`):

- **IMPLEMENTATION_BUG**: none in the judge JSON parsing/validation.
- **SPECIFICATION_MISMATCH**: the 2.2A `visual_dependence` rubric measures
  answer-image reliance, not question–image specificity.
- **RUBRIC_VALIDITY_RISK**: saturation (mode 5 ≈ 95%); the `generic`
  failure category penalises portable-but-valid questions; `unique_best_match`
  wording encourages uniqueness. A rubric risk cannot be fixed by downstream
  arithmetic. Any replacement is recorded in `rerun_plan.json` (PROPOSED_NOT_RUN)
  and was **not** executed.

## 8. Cleanup summary

- Active package rebuilt as `src/vvqg/evaluation/{metrics,common,audit}` with one
  canonical implementation per metric.
- Obsolete packages/scripts/tests/docs/outputs moved under
  `archive/evaluation_legacy/`, `docs/archive/evaluation/`, and
  `outputs/archive/evaluation_legacy/` with hashes recorded in
  `cleanup_manifest.json`.
- Stale bytecode for removed modules deleted.
- No active module is named after an obsolete experiment; no active runner
  offers deprecated protocols; the active CLI exposes only supported options.
- BERTScore/PAC-S/VQAScore are not imported, configured or offered anywhere in
  the active package; they are mentioned only in exclusion statements.

Before/after trees: `pre_cleanup_tree.txt`, `post_cleanup_tree.txt`.

## 9. Verification

- Active evaluation tests: `pytest tests/test_evaluation_metrics.py` -> 9 passed.
- Import smoke: `vvqg.evaluation.{metrics,common,audit}` import cleanly; the
  archived modules are no longer importable from the public active package.
- CLI smoke: `scripts/vvqg/evaluate.py --help` lists only supported options; a
  full run on the historical vanilla predictions reproduces the archived
  numbers exactly (Section 1).
- Repository stale-reference search: deprecated names appear only in exclusion
  statements, the exclusion unit test, and the unrelated reward-schema test
  (`tests/test_reward_framework.py` uses `visual_dependence` as a human-label
  field, not the GPT rubric).
- `pytest tests/` (active collection): the 12 failures are pre-existing and
  confined to `tests/premise_dpo/` (missing `artifacts/preference_baseline_pools`
  data), unrelated to the evaluation subsystem; active evaluation tests pass
  without depending on archived modules.
