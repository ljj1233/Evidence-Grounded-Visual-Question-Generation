# P1strong Multi-Seed Confirmation (Phase 3A.4)

Training-only seed change (43, 44) vs the seed-42 P1strong run. Everything else
frozen: UVFC reward, ties, advantages, optimizer (AdamW lr 3e-6, cosine, 5%
warmup, clip 1.0, beta_KL 0.02), 2 epochs / 500 steps, microbatch 1 group,
grad-accum 8, LoRA r16/a32/d0.05, the same P1 rollout file and frozen old-logprob
cache, and the same Final SFT starting checkpoint. No checkpoint averaging, no
best-seed selection, no online GRPO, `final_holdout` untouched.

## Per-seed results (development1000 greedy)

| System | Search | Coverage | Context | Spatial | Supported | Unsupported | BLEU | CIDEr | Dup | Unique | mean UVFC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Plain SFT | 0.2746 | 0.2598 | 0.2704 | 0.0732 | 0.725 | 0.159 | 0.3172 | 0.4550 | 0.570 | 0.430 | 0.218 |
| seed42 | 0.3200 | 0.3256 | 0.2854 | 0.0622 | 0.734 | 0.114 | 0.2915 | 0.4703 | 0.461 | 0.539 | 0.4545 |
| seed43 | 0.3100 | 0.3206 | 0.2888 | 0.0610 | 0.760 | 0.105 | 0.2918 | 0.4623 | 0.456 | 0.544 | 0.4554 |
| seed44 | 0.3136 | 0.3186 | 0.2840 | 0.0630 | 0.758 | 0.112 | 0.2885 | 0.4582 | 0.462 | 0.538 | 0.4542 |

All three seeds completed 500 steps with no early stop, no stop guard, no
malformed output (malformed rate 0.0), final reference-KL 0.0135–0.0198 and clip
fraction 0.18–0.23 (no KL explosion).

## Paired bootstrap vs SFT (5,000 resamples, seed 42)

| Metric | seed42 Δ [95% CI] | seed43 Δ [95% CI] | seed44 Δ [95% CI] |
|---|---:|---:|---:|
| Search | +0.0454 [0.0348, 0.0562] | +0.0354 [0.0250, 0.0454] | +0.0390 [0.0284, 0.0496] |
| Coverage | +0.0658 [0.0528, 0.0794] | +0.0608 [0.0476, 0.0738] | +0.0588 [0.0458, 0.0720] |
| Context | +0.0150 [0.0066, 0.0232] | +0.0184 [0.0098, 0.0270] | +0.0136 [0.0056, 0.0216] |
| Spatial | −0.0110 [−0.0206, −0.0016] | −0.0122 [−0.0212, −0.0034] | −0.0102 [−0.0196, −0.0010] |
| Supported | +0.009 [−0.020, 0.038] | +0.035 [0.005, 0.065] | +0.033 [0.004, 0.062] |
| Unsupported | −0.045 [−0.068, −0.022] | −0.054 [−0.078, −0.031] | −0.047 [−0.071, −0.023] |

Mean ± std of per-seed Δ: Search **+0.0399 ± 0.0041**, Coverage **+0.0618 ± 0.0029**,
Context +0.0157 ± 0.0020, Spatial −0.0111 ± 0.0008, Supported +0.0257 ± 0.0118,
Unsupported −0.0487 ± 0.0039. Positive-seed count: Search 3/3, Coverage 3/3,
Context 3/3, Supported 3/3; Spatial 0/3, Unsupported 0/3 (all reduce Unsupported).

## Decision

`outputs/phase3a_multiseed_p1strong/decision.json`:

```json
{"decision": "ROBUST_UVFC_UPDATE", "final_holdout_touched": false,
 "grpo_started": false, "checkpoints_averaged": false, "best_seed_selected": false}
```

## Answers to the required questions

1. **Is the P1strong improvement seed-stable?** Yes. Across seeds 42/43/44 the
   Search and Coverage gains are large and tightly clustered (std ≤0.4 and ≤0.3
   points), and every seed beats SFT with CIs excluding zero. This is not a
   single-seed artifact.

2. **Does UVFC consistently improve Search/Coverage?** Yes, 3/3 seeds, both
   metrics, all CIs positive. The mean UVFC of generated questions is nearly
   identical across seeds (0.4545 / 0.4554 / 0.4542), so the optimization lands
   in the same feature regime regardless of shuffle order.

3. **Is Unsupported controlled?** Yes. All three seeds reduce Unsupported
   (−0.045, −0.054, −0.047) and non-negatively raise Supported; none breaches the
   +0.02 degeneration band.

4. **Is Spatial degradation consistent or noisy?** Consistent but small
   (−0.0111 ± 0.0008; 0/3 seeds positive). Phase 3A.3 attributed it to an
   optimization-side distribution shift (fewer action/object/color questions,
   more yes/no and count), because the offline UVFC ranking actually favours
   higher-Spatial questions. It is a stable side-effect, not judge noise and not
   a reward-construct defect.

5. **Is the fixed-rollout optimization ready for a small online GRPO pilot?**
   Qualified yes. The update is stable, seed-robust, and consistently improves
   the primary visual-centric axes (Search/Coverage) and Unsupported, with
   diversity and length improving rather than degrading. The known costs are a
   small consistent Spatial decline and unchanged Context magnitude. A **small,
   guarded** fresh-rollout on-policy pilot is justified as the next step; it
   should keep the same KL/clip stop guards, not use `final_holdout`, and still
   requires solving UVFC's online cost (proxy/distillation), which this phase did
   not attempt. No such online run is launched here.

## Caveats

- Seed here changes only the group shuffle order (rollouts frozen, dropout off,
  deterministic adapter init); it measures optimization-order sensitivity, not
  full end-to-end variance.
- SeeQ dims use the documented `gpt-5.6-luna` substitution; support uses GPT-4o.
- Mean UVFC is computed with the frozen SFT+SAE scorer; it is a diagnostic, not a
  final metric.
