# Controlled Visual-Support Minimal-Pair Benchmark v1

## Goal

This separate stress test asks whether the frozen Holistic and Flat Requirement-aware visual-support verifiers react to one controlled visual-support failure while preserving the surrounding question wording. It uses only Phase 2.1 VQAv2/COCO calibration images and supported anchors. It does not modify the frozen Phase 2.1 test result.

## Construction and validation

GPT-4o generated candidate perturbations and a separate GPT-4o prompt independently validated both questions. The validator randomized question order and required: supported q+, unsupported q−, grammaticality for both, exactly one visual factor changed, category agreement, and non-identical normalized text. Proxy variables were disabled for API calls. Rejected candidates and reasons are retained in `pair_rejections.jsonl`.

The pilot retained 1/50 candidates. The frozen construction retained only 6 pairs: wrong attribute 1, wrong relation 4, and unobservable target 1. Missing entity and ambiguous reference retained zero. Every category is below the required minimum of 50, so this benchmark is underpowered.

## Frozen verifier evaluation

Both verifiers used the same Qwen2.5-VL-7B-Instruct checkpoint and existing Phase 2.1 prompt/configuration family. No dependency-aware output or prompt tuning was used. All 6 pairs were scored by both systems.

| Verifier | Positive Preservation | Negative Detection | Strict Pair Accuracy | False Acceptance | Coverage |
|---|---:|---:|---:|---:|---:|
| Holistic | 0.667 | 0.833 | 0.500 | 0.167 | 1.000 |
| Flat Requirement-aware | 0.667 | 0.833 | 0.667 | 0.000 | 1.000 |

Holistic ordered pairs were win/tie/loss = 3/3/0; Flat was 5/1/0. These are descriptive pilot statistics only.

## Category metrics and paired tests

Per-category metrics are in `category_metrics.json`. The paired bootstrap and McNemar contingency table are in `paired_bootstrap.json` and `mcnemar_results.json`. With only six valid pairs, uncertainty intervals are not suitable for a superiority claim.

## Error transitions and mechanism audit

Transition counts and deterministic examples are in `error_transitions.json` and `error_transition_examples.jsonl`. Flat trigger counts are in `flat_trigger_analysis.json`. No meaningful category-level mechanism conclusion is possible because two categories have no accepted pairs.

## Runtime and coverage

Construction used separate generation and validation calls with 32-way bounded concurrency, incremental raw artifacts, and rejection logging. Verifier runtime and parser coverage are in `runtime_report.json`; both verifiers had full coverage on the six retained pairs.

## Decision

**NO SUPPORT (UNDERPOWERED).** The strict validator retained only six pairs and no category reached the required minimum of 50. The apparent Flat strict-pair advantage is not evidence of verifier superiority. Do not tune either verifier on this benchmark or use it to justify downstream changes.

## Limitations

This is an underpowered controlled benchmark, not the Phase 2.1 frozen test set. GPT-4o validation is pseudo-annotation rather than human ground truth. The zero-count categories prevent meaningful category-level inference. No optional COCO instance-annotation sanity subset was run; no GQA, final_holdout, SFT changes, dependency structure, or GRPO was used.
