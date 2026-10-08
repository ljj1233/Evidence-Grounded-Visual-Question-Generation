# P1strong Evaluation Results (Phase 3A.2)

Evaluation-only phase. Checkpoint: UVFC-only fixed-rollout group-relative policy
trained for 2 epochs at lr 3e-6 (`p1_uvfc_stronger/checkpoint_final`). No
retraining, reward change, or online GRPO was performed. `final_holdout` untouched.

Generated on the frozen 1,000-image `development1000` set: greedy (1/img) and
sampled (4/img, T=0.8, top_p=0.9, 32 tokens). Generation reproduced the
in-training outputs exactly.

## Baseline table

| System | BLEU | CIDEr | Search | Coverage | Context | Spatial | Supported | Unsupported | Dup | SemDiv |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Plain SFT | 0.3172 | 0.4550 | 0.2746 | 0.2598 | 0.2704 | 0.0732 | 0.725 | 0.159 | 0.570 | 0.5360 |
| P1 | 0.2998 | 0.4533 | 0.2908 | 0.2804 | 0.2748 | 0.0698 | 0.713 | 0.132 | 0.516 | 0.5334 |
| P2 | 0.3222 | 0.4723 | 0.2840 | 0.2710 | 0.2636 | 0.0706 | 0.741 | 0.131 | 0.539 | 0.5272 |
| **P1strong** | 0.2915 | 0.4703 | **0.3200** | **0.3256** | **0.2854** | 0.0622 | 0.734 | **0.114** | **0.461** | **0.5423** |

## Paired bootstrap (5,000 resamples, seed 42, image-level)

| Metric | P1strong − SFT | P1strong − P1 | P1strong − P2 |
|---|---:|---:|---:|
| Search | +0.0454 [0.0348, 0.0562] | +0.0292 [0.0198, 0.0390] | +0.0360 [0.0258, 0.0464] |
| Coverage | +0.0658 [0.0528, 0.0794] | +0.0452 [0.0336, 0.0574] | +0.0546 [0.0416, 0.0674] |
| Context | +0.0150 [0.0066, 0.0232] | +0.0106 [0.0040, 0.0170] | +0.0218 [0.0138, 0.0300] |
| Spatial | −0.0110 [−0.0206, −0.0016] | −0.0076 [−0.0156, 0.0002] | −0.0084 [−0.0178, 0.0008] |
| Supported | +0.009 [−0.020, 0.038] | +0.021 [−0.004, 0.046] | −0.007 [−0.034, 0.020] |
| Unsupported | −0.045 [−0.068, −0.022] | −0.018 [−0.037, 0.001] | −0.017 [−0.039, 0.004] |

## Distribution shift (Plain SFT vs P1strong, generated questions only)

- Length: mean token length 6.554 → 6.661 (+1.6%); p95 9 → 9. No length blow-up.
- Diversity: unique-question rate 0.430 → 0.539; normalized duplicate rate
  0.570 → 0.461; Distinct-2 0.1395 → 0.1856; sampled rollout unique rate 0.963;
  within-image semantic diversity 0.5360 → 0.5423 (highest of all systems).
- Question type (GPT-4o, `phase25b` taxonomy; SFT → P1strong): action 0.299 →
  0.131, object 0.231 → 0.185, color 0.145 → 0.091, attribute 0.073 → 0.044;
  yes_no 0.047 → 0.232, count 0.015 → 0.110, scene/context 0.045 → 0.082. The
  policy shifts from open action/object questions toward grounded yes/no and
  counting questions, consistent with higher evidence coverage and unchanged
  support/quality rather than formatting degeneration.

## UVFC behaviour check

Mean UVFC of the development1000 greedy questions under the frozen SFT+SAE
scorer (`diagnostics/uvfc_shift.json`):

| System | mean UVFC | Δ vs SFT |
|---|---:|---:|
| Plain SFT | 0.2180 | — |
| P2 | 0.2448 | +0.0268 |
| P1 | 0.3333 | +0.1153 |
| **P1strong** | **0.4545** | **+0.2365** |

Stronger optimization moved the output distribution substantially toward the
UVFC-optimized feature distribution (≈2× P1's shift), and this is accompanied by
the largest Search/Coverage gains. The violation constraint (P2) suppresses most
of that shift (+0.027), consistent with its smaller Search/Coverage movement.

## Interpretation

The stronger update (final reference-KL 0.0198 vs P1's 0.0037, ≈5.3×; clip
fraction ~0.18–0.24 vs 0.031) roughly doubles-to-triples the realized gain:
Search +0.045 and Coverage +0.066 over SFT, both far larger than P1 (+0.016 /
+0.021) and P2 (+0.009 / +0.011), with paired CIs excluding zero. Context also
turns significantly positive (+0.015). Supported is flat-to-slightly-up and
Unsupported drops most (−0.045). Diversity/duplication improve rather than
degrade; length is stable. Only Spatial regresses slightly (−0.011).

This is the expected signature of an **under-updated** pilot: the fixed-rollout
group-relative objective was learning the ranking (surrogate improving, UVFC
rising) but the original lr=1e-6 update moved the distribution too little for the
gain to materialize. Increasing the update magnitude transfers markedly more of
the offline ranking signal.

## Decision

`outputs/phase3a_p1strong_eval/decision.json`:

```json
{"decision": "UPDATE_LIMITED", "degeneration_flag": false,
 "grpo_started": false, "final_holdout_touched": false}
```

UPDATE_LIMITED: P1strong improves Search/Coverage over SFT; the improvement is
larger than P1 and P2; and no degeneration guard triggered. The previous pilot
was under-updated. As before, this remains a fixed-rollout result and is not
evidence of on-policy GRPO success; no larger or online run is launched.

## Caveats

- Single training seed; the magnitude is large and CI-separated, but cross-seed
  variance is still unmeasured.
- Spatial degrades significantly (−0.011); the reward does not target it.
- SeeQ dims use the documented `gpt-5.6-luna` substitution, not paper GPT-5.4.
- Support rates use valid parsed labels (1,000/1,000 after retry).
