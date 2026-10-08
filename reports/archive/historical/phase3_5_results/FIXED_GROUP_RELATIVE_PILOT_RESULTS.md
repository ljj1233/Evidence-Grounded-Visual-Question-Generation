# Fixed-Rollout Group-Relative Policy Optimization Pilot — Results

Phase 3A. Objective and data contract: `docs/FIXED_GROUP_RELATIVE_PILOT_METHOD.md`.
Runtime: `docs/FIXED_GROUP_RELATIVE_PILOT_RUNTIME.md`. No master score.

Both pilots start from the exact same Final SFT adapter and use the fixed
seed-42 rollouts. `grpo_started = false` (no on-policy GRPO); `final_holdout`
was not touched.

## Primary table

All rows use the 1,000-image development manifest (`development1000`), greedy
decoding (1 question/image), and Evaluation Protocol v1. SeeQ dims are the
released single-question rubric with the documented `gpt-5.6-luna` judge
substitution. "Greedy Dup" is the normalized excess duplicate rate.

| System | BLEU-4 | CIDEr | Search | Coverage | Context | Spatial | Supported | Unsupported | Greedy Dup | 4x SemDiv |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Plain SFT | 0.3172 | 0.4550 | 0.2746 | 0.2598 | 0.2704 | 0.0732 | 0.725 | 0.159 | 0.570 | 0.5360 |
| P1 UVFC-only | 0.2998 | 0.4533 | **0.2908** | **0.2804** | 0.2748 | 0.0698 | 0.713 | 0.132 | 0.516 | 0.5334 |
| P2 UVFC+Violation | 0.3222 | 0.4723 | **0.2840** | **0.2710** | 0.2636 | 0.0706 | **0.741** | 0.131 | 0.539 | 0.5272 |

Sampled 4× lexical diversity (on the same 1,000 images):

| System | Distinct-2 (4×) | Normalized dup (4×) |
|---|---:|---:|
| P1 UVFC-only | 0.1973 | 0.317 |
| P2 UVFC+Violation | 0.1886 | 0.339 |

Language-quality preservation: P1 BLEU-4 = 0.946× SFT, P2 BLEU-4 = 1.016× SFT;
P1 CIDEr = 0.996× SFT, P2 CIDEr = 1.038× SFT. Neither crosses the 0.80×
degeneration flag. Malformed rate 0.0 and question-mark rate 1.0 for both.

## Paired bootstrap (5,000 resamples, seed 42, image-level)

Delta direction is left minus right. 95% percentile CIs.

| Metric | P1 − SFT | P2 − SFT | P2 − P1 |
|---|---:|---:|---:|
| Search | +0.0162 [0.0074, 0.0246] | +0.0094 [0.0022, 0.0162] | −0.0068 [−0.0146, 0.0010] |
| Coverage | +0.0206 [0.0102, 0.0312] | +0.0112 [0.0022, 0.0202] | −0.0094 [−0.0196, 0.0008] |
| Context | +0.0044 [−0.0020, 0.0108] | −0.0068 [−0.0118, −0.0018] | −0.0112 [−0.0174, −0.0050] |
| Spatial | −0.0034 [−0.0118, 0.0048] | −0.0026 [−0.0104, 0.0052] | +0.0008 [−0.0072, 0.0088] |
| Supported | −0.012 [−0.037, 0.012] | +0.016 [−0.004, 0.037] | +0.028 [0.007, 0.050] |
| Unsupported | −0.027 [−0.046, −0.008] | −0.028 [−0.046, −0.011] | −0.001 [−0.019, 0.017] |

Corpus BLEU-4 / CIDEr deltas are in `comparison/paired_bootstrap.json`
(resampled with the frozen CIDEr IDF).

## Interpretation

- Both P1 and P2 raise **Visual Search** and **Evidence Coverage** over SFT with
  CIs excluding zero, so the offline UVFC ranking transfers into the policy.
- The violation constraint (P2) trades a small amount of Search/Coverage against
  P1 for a **higher Supported rate** (0.741, +0.016) and an equally large
  Unsupported reduction (−0.028); P2 is the most conservative system.
- P1 reduces greedy duplication most (0.570 → 0.516); P2 reduces Unsupported as
  much as P1 without needing a separate fast rule.
- Context/Spatial do not improve; P2's Context is modestly below SFT
  (−0.0068). This is expected: neither reward targets those axes.
- No reward-hacking signature: length, malformed rate, question-mark rate and
  duplication all stay within guard bands on the train-only monitor set.

## Decision

`outputs/phase3a_fixed_group_relative/final_decision.json`:

```json
{"decision": "FULL_METHOD_GO", "grpo_started": false, "final_holdout_touched": false}
```

FULL_METHOD_GO applies to P2 because: Search and Coverage point estimates exceed
SFT; at least one paired CI excludes zero positively (both do); the other is not
significantly negative; Unsupported ≤ SFT with CI upper bound ≤ +0.02; and no
degeneration guard triggered.

This is evidence that fixed-rollout group-relative learning can transfer the
ranking signal into the SFT policy. It is **not** evidence of on-policy GRPO
success; no larger or on-policy run is launched automatically.
