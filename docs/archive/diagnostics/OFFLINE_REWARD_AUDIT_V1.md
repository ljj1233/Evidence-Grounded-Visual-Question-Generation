# Phase 2.5 — Offline Best-of-4 Reward Audit

## Decision

**`PARTIAL_VIOLATION_ONLY`**. The frozen four-sample SFT pool has substantial visual-centric headroom, and the Flat verifier tracks independent GPT-4o unsupported judgments. `G_text` does not rank Search or Evidence Coverage above random selection, so the VIC component is not ready for GRPO. No GRPO run was started.

## Protocol and coverage

The audit uses only `outputs/reward_calibration/sft_candidates_4x.jsonl`, with seed 42. The canonical pool has 1,000 image IDs and exactly four candidates per image (4,000 records); no questions were regenerated and no B2/B3/B4 or final-holdout records were used. Candidate IDs are `sha256(image_id|generation_id|sha256(whitespace-collapsed-casefold(question)))`, so identity is independent of execution order.

All candidate records remain present in every workstream. VIC answers and scores cover 4,000/4,000 records. After repairing the interrupted Flat pass (retrying non-success rows and using the frozen 256-token generation limit), Flat states cover 4,000/4,000: `NO_CONFIRMED_VIOLATION` 2,387, `CONFIRMED_VIOLATION` 1,386, `UNCERTAIN` 161, and `FAILED` 66. Luna records cover 4,000/4,000 (3,981 successful and 19 cached timeout failures); GPT-4o support records cover 4,000/4,000 (3,998 successful, one parser failure, one API failure). Failures are retained and excluded only from the corresponding rate denominators.

## Signal validity

| Signal | External diagnostic | Result |
|---|---|---:|
| `G_text` | top-vs-bottom Search, paired image bootstrap | −0.444 [−0.519, −0.370] |
| `G_text` | top-vs-bottom Evidence Coverage | −0.325 [−0.412, −0.242] |
| `G_text` | pairwise Search concordance | 0.373 (1,087/2,913) |
| `G_text` | pairwise Coverage concordance | 0.403 (1,038/2,577) |
| Flat confirmed violation | independent Unsupported rate | 0.286 (396/1,386) |
| Flat no confirmed violation | independent Unsupported rate | 0.074 (177/2,385) |

Top-vs-bottom bootstrap uses 999 images with usable Luna scores. Ties in `G_text` are resolved by the stable seed-42 candidate hash; evaluator-score ties are excluded from pairwise denominators.

## Candidate headroom

| Selector | Search | Coverage | Supported | Unsupported |
|---|---:|---:|---:|---:|
| Random-from-4 | 1.587 | 1.651 | 0.659 | 0.171 |
| Oracle Search (diagnostic only) | 2.339 | 2.268 | 0.700 | 0.130 |
| Oracle Coverage (diagnostic only) | 2.065 | 2.503 | 0.694 | 0.121 |

The oracle gap establishes selectable headroom beyond random. It is not a deployable selector and was not used as an input to any method.

## Offline selector results

| Selector | BLEU-4 | CIDEr | Search | Coverage | Context | Spatial | Supported | Unsupported | Distinct-2 | Dup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT Greedy | 0.317 | 0.455 | 1.373 | 1.299 | 1.352 | 0.366 | 0.725 | 0.159 | 0.140 | 0.570 |
| Random-from-4 | 0.216 | 0.320 | 1.587 | 1.651 | 1.392 | 0.344 | 0.659 | 0.171 | 0.305 | 0.183 |
| VIC-only | 0.178 | 0.244 | 1.406 | 1.532 | 1.512 | 0.294 | 0.604 | 0.196 | 0.319 | 0.181 |
| Violation-only | 0.221 | 0.322 | 1.520 | 1.602 | 1.371 | 0.360 | 0.730 | 0.115 | 0.289 | 0.207 |
| Full lexicographic | 0.200 | 0.275 | 1.416 | 1.536 | 1.452 | 0.318 | 0.707 | 0.122 | 0.307 | 0.183 |

`Dup` is normalized excess duplicate rate. One Full-selection support failure leaves 999 valid support labels for that row; all other displayed selector rows have 1,000 valid support labels. SeeQ within-image semantic diversity is intentionally not reported for one-question-per-image selections.

The 100 deterministic random draws (seeds 42–141) have mean Search 1.563 [1.518, 1.611], Coverage 1.644 [1.603, 1.692], Supported 0.673 [0.654, 0.700], and Unsupported 0.161 [0.146, 0.179].

## Paired bootstrap contrasts

Image-level paired bootstrap uses 5,000 resamples and seed 42. VIC minus Random is Search −0.180 [−0.237, −0.123] and Coverage −0.122 [−0.194, −0.051]. Violation-only minus Random is Unsupported −0.056 [−0.076, −0.036] and Supported +0.071 [+0.046, +0.095]. Full minus Random is Search −0.168 [−0.226, −0.109], Coverage −0.116 [−0.185, −0.049], and Unsupported −0.048 [−0.074, −0.022]. Full minus Greedy is Search +0.045 [−0.008, +0.098], Coverage +0.239 [+0.173, +0.306], Supported −0.018 [−0.055, +0.017], and Unsupported −0.037 [−0.066, −0.009]. Complete values are in `outputs/phase25_offline_reward_audit/evaluation/paired_bootstrap.json`.

## Availability and interpretation

970/1,000 images have at least one admissible candidate; 30/1,000 have all four confirmed violations; 1/1,000 has only uncertain/failed states. Violation-only uses a deterministic random fallback for those 30 images; Full uses `argmax G_text` as its constraint fallback for the same 30 images. Per-image counts are in `violation/candidate_availability.jsonl`.

The result supports the violation detector as a useful constraint, but not VIC as a ranking signal for Search/Coverage. The Full selector therefore does not satisfy the practical GO condition for a GRPO pilot. The appropriate next label is `PARTIAL_VIOLATION_ONLY`; redesign or independently validate the information-contribution component before reconsidering GRPO.

Final-holdout image IDs were checked against the candidate pool (zero overlap), and `final_holdout_touched` is false in `final_decision.json`.
