# Visual Support Dependency Ablation v1

## 1. Goal and hypothesis

This is a single dependency-aware requirement ablation following Phase 2.1. The hypothesis was that downstream reference and target checks should not receive positive credit when a prerequisite relation or entity check fails. No model, GPT label, split, SFT output, or reward model was changed.

## 2. Previous flat result

The frozen test set had Holistic overall macro-F1 0.412458 and Flat Requirement-aware macro-F1 0.501979. The paired 5,000-resample Flat minus Holistic bootstrap gave delta 0.089521, 95% CI [-0.001108, 0.176689], and P(delta>0)=0.9728. Balanced accuracy delta was 0.091435 with CI [0.020533, 0.169231].

## 3. Dependency representation

The existing flat requirement records were extended only with `id`, `type`, and `depends_on`. Relations depend on their entity prerequisites, references depend on preceding requirements, and the query target depends on references (or preceding prerequisites when no reference exists). Yes/no and count targets receive no existence presupposition. No graph embeddings, learned module, ontology, scene graph, or external parser was added.

## 4. Structural diagnostics

The existing local extraction output contained 397/1000 parseable structures; all parseable dependency structures were valid DAGs, while the remaining 603 are explicitly marked `parse_invalid`. Mean requirements/question among parseable structures was 3.758, mean edges/question 2.970, maximum depth 6, mean depth 1.852, and the deterministic compositional subset contained 246 examples. Full diagnostics are in `dependency_extraction_diagnostics.json`; 100 inspection examples are in `dependency_examples.jsonl`.

## 5. Dependency gating

Prerequisite negative states become `blocked_negative`; prerequisite uncertain states become `blocked_uncertain`; only all-positive prerequisites allow a local state through. The final three dimensions and overall label use the unchanged Phase 2.1 categorical contract. All local evidence was reused; the dependency run made zero VLM calls and had a 1.0 cache-hit rate.

## 6. Three-way test comparison

| Verifier | Overall Macro-F1 | Balanced Acc | Premise F1 | Reference F1 | Answerability F1 |
|---|---:|---:|---:|---:|---:|
| Holistic | 0.412458 | 0.435084 | 0.431253 | 0.231183 | 0.407407 |
| Flat | 0.501979 | 0.526519 | 0.405513 | 0.229227 | 0.552310 |
| Dependency | 0.502341 | 0.510932 | 0.405513 | 0.229227 | 0.552310 |

The dependency representation changed the overall output only through prerequisite propagation; reference performance did not improve.

## 7. Dependency versus Flat bootstrap

For overall macro-F1, Dependency minus Flat was 0.000362, with 95% CI [-0.054860, 0.062805] and P(delta>0)=0.5028. Balanced accuracy delta was -0.015587 with CI [-0.070799, 0.038360]. Premise, reference, and answerability deltas also had intervals crossing zero. Results are in `dependency_vs_flat_bootstrap.json`.

## 8. Error transitions

On the 200 test examples: both correct 124, both wrong 67, Flat wrong → Dependency correct 3, and Flat correct → Dependency wrong 6. Dependency therefore introduced twice as many observed transitions in the wrong direction as in the right direction. Detailed examples are in `error_transition_examples.jsonl`.

## 9. Reference grounding

Reference macro-F1 remained 0.229227, equal to Flat at the reported precision. Ambiguous-reference evidence is sparse and dependency propagation did not improve it. The observed overall difference is negligible and does not demonstrate better target binding.

## 10. Runtime and practicality

The ablation used cached per-requirement evidence, made zero new local VLM calls, and ran as a deterministic CPU aggregation pass. Runtime and cache accounting are in `runtime_report.json`.

## 11. Decision

**NO-GO for retaining dependency structure.** The dependency-aware representation is statistically indistinguishable from Flat and has unfavorable error transitions. Keep the Flat Requirement-aware verifier as the structured candidate. Do not expand this into a larger dependency system and do not start GRPO.

## 12. Limitations

The reference labels remain GPT-4o pseudo-labels rather than human ground truth. Dependency extraction coverage is limited by the existing Phase 2.1 local extraction artifacts, so the structural result is an ablation of the parseable subset while evaluation uses the common 200-example test intersection. This experiment does not establish generalization beyond the frozen calibration benchmark.
