# FINAL REPORT — Support-Delta GRPO development1000 confirmation

No training, no optimizer step, no model/reward change. Original pilot status unchanged.

## Headline

- **STATUS: CONFIRMATION_COMPLETE**
- **ORIGINAL_PILOT_STATUS: FAIL_PILOT** (unchanged).
- **Coverage did NOT replicate**: remaining744 delta −0.0030 (CI [−0.0096, +0.0033]); full1000 +0.0011 (CI [−0.0050, +0.0070]).
- **Supported improved on full1000**: +0.0120 (CI [+0.0020, +0.0224]), driven by remaining744 (+0.0175).
- Unsupported is flat (full1000 −0.0010, CI crosses 0); the Supported gain is largely Uncertain→Supported.

## FACT

1. Splits: existing256 = 256, remaining744 = 744, full1000 = 1000 (`IMAGE_SPLIT_AUDIT.json`).
   existing256 equals the fixed pilot 256 manifest; disjoint from remaining744; all ⊂ development1000.
2. Base/step300 outputs are fixed: 1000 questions each, identical image_id sets, no duplicates, no empty
   questions; existing256 question hashes match the prior pilot exactly
   (`base 22cd1299…`, `step300 74839e1f…`). (`generation/merge_audit.json`).
3. remaining744 generation complete (744 Base + 744 step300, 0 failures, 0 empty) with the frozen
   config (greedy, T=0, mn=32, seed 42, same checkpoints; adapter hashes `52ee692c…`, `bdd810a4…`).
4. Evaluator repeats 3/3 complete on original only; 0 parser failures; 7000 API calls
   (support 844×3, seeq 1488×3 + 4 bounded retries); existing256 repeats reused unchanged.
5. Paired results (image-level bootstrap 5000, seed 42):

| split | metric | n | Base | step300 | delta | 95% CI |
|---|---|---:|---:|---:|---:|---|
| existing256 | coverage | 256 | 0.3826 | 0.3958 | +0.0133 | [+0.0013, +0.0268] |
| existing256 | supported | 256 | 0.8411 | 0.8372 | −0.0039 | [−0.0260, +0.0182] |
| existing256 | unsupported | 256 | 0.0495 | 0.0560 | +0.0065 | [−0.0104, +0.0234] |
| remaining744 | coverage | 744 | 0.3926 | 0.3895 | −0.0030 | [−0.0096, +0.0033] |
| remaining744 | supported | 742 | 0.8607 | 0.8783 | +0.0175 | [+0.0063, +0.0292] |
| remaining744 | unsupported | 742 | 0.0395 | 0.0359 | −0.0036 | [−0.0099, +0.0022] |
| full1000 | coverage | 1000 | 0.3900 | 0.3911 | +0.0011 | [−0.0050, +0.0070] |
| full1000 | supported | 998 | 0.8557 | 0.8677 | +0.0120 | [+0.0020, +0.0224] |
| full1000 | unsupported | 998 | 0.0421 | 0.0411 | −0.0010 | [−0.0073, +0.0050] |
| full1000 | uncertain | 998 | 0.1022 | 0.0912 | −0.0110 | [−0.0197, −0.0030] |

6. Per-repeat full1000 deltas: coverage +0.0012 / +0.0030 / −0.0008 (directions +,+,−);
   supported +0.012 / +0.013 / +0.011 (+,+,+); unsupported −0.001 × 3; uncertain −0.011 × 3.
   remaining744 coverage deltas are negative in all three repeats.
7. Label flip (same system, same fixed question, r1 vs r2/r3): Base 35/1000 and 32/1000;
   step300 30/1000 and 30/1000.
8. Gain image rates (full1000): coverage gain 4.5%, supported gain 2.3%, unsupported increase 0.7%.

## INTERPRETATION

- The Coverage gain reported on the pilot 256 images **does not generalize** to the 744 images that
  were not part of the pilot; on full development1000 the Coverage delta is indistinguishable from 0.
  Development1000 is not an independent test set.
- Supported **does** improve on the larger set and is consistent across all three repeats; the gain
  is a reclassification of Uncertain→Supported (Uncertain falls by 0.0110) rather than an
  Unsupported reduction (Unsupported is flat). This is a **Coverage/support trade-off candidate**:
  better support-label rate without a Coverage gain.
- The two splits move in opposite directions for Coverage, which is why the full set is null.

## UNKNOWN

- Whether the Supported gain is causally due to the reward or to judge behavior on the changed
  question distribution (no retraining ablation; step300 rollouts unsaved).
- Whether either effect survives on a truly independent set (development1000 is not one).
- Whether the Coverage reversal on remaining744 reflects a genuine distribution difference or
  residual evaluator variance (3 repeats bound, do not remove, systematic judge bias).

## Preset descriptive status

- COVERAGE_REPLICATION_STATUS: `COVERAGE_NOT_REPLICATED`
- SUPPORTED_STATUS: `SUPPORT_IMPROVEMENT_SUPPORTED`
- UNSUPPORTED_STATUS: `UNSUPPORTED_UNCERTAIN`

## Answers

1. 256 / 744 / 1000. 2. Base and step300 questions are completely fixed and identical sets.
3. remaining744 generation complete. 4. 3/3 repeats complete. 5-7. Coverage/Supported/Unsupported per split above.
8-9. Deltas and CI above; Coverage direction is not consistent (full +,+,−; remaining −,−,−), Supported is consistent (+,+,+).
10. Coverage not replicated. 11. Supported improved (not merely preserved). 12. Unsupported flat.
13. Base 35/32, step300 30/30 flips (1000 images). 14. 7000 API calls. 15. 0 parser failures.
16. final_holdout NOT accessed. 17. FAIL_PILOT unchanged. 18. No new training authorized.

`TRAINING_AUTHORIZED_BY_THIS_TASK: NO`.
