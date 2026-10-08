# FAILURE ANALYSIS — G3 pilot gate FAIL

## Result

All 12 gate conditions were evaluated; 9 passed, 3 failed:

1. Supported point estimate +0.01042 < required +0.02.
2. Supported 95% CI lower −0.01432 < 0.
5. Unsupported 95% CI upper +0.00654 > 0.

The directions are favorable (Supported ↑, Unsupported ↓, Coverage ↑) but the
effects are within evaluator/rollout noise at 256 images.

## Root causes

1. **The policy barely moved.** `ratio_mean` stayed in [0.9926, 1.0098] and
   `reference_kl ≤ 0.0039` for all 300 steps. With `lr=5e-7`, `beta_kl=0.02` and
   the KL anchor at the init, the update is a very small conservative nudge. Over
   300 steps this yields ~+0.01 Supported, below the gate.
2. **Reward contrast is low on this policy.** Base Full Ours already has high
   external Supported (0.850) and low Unsupported (0.066). Most 4-rollout groups
   are all-supported → near-zero group-relative advantage → little gradient. The
   support term (0/0.5/1) saturates and cannot separate already-supported groups.
3. **Local Flat vs external GPT-4o mismatch.** The local frozen Flat Supported
   proxy *fell* (0.770→0.734) while external GPT-4o Supported *rose* (+0.010).
   Optimising the Flat-derived reward is only weakly aligned with the external
   gate metric.
4. **UVFC anchor rose** (0.724→0.769) — the reward's 0.20 anchor pulled toward
   higher-UVFC questions, which is not the same as higher Supported.

## Minimal modifications to try (not run)

1. Increase effective movement: `lr=2e-6` (or 3 inner epochs) with the same KL
   anchor, so ratio/clip actually bite (target clip_fraction ~0.05–0.15).
2. Rebalance the reward toward the gate: raise the anti-unsupported contrast
   (e.g., larger negative term for CONFIRMED_VIOLATION) and down-weight the UVFC
   anchor, which does not track Supported.
3. Mine harder groups: sample from a pool enriched in unsupported/uncertain base
   questions so group-relative advantage is nonzero.
4. Keep the 256-image pilot gate but report a larger pilot (e.g., 512) if the
   expected effect is ~+0.01, since 256 images cannot resolve it.

## Not done (per gate rules)

- No 1000-step formal run.
- No new RL variant.
- No checkpoint re-selection based on intervention results.
- `final_holdout` not accessed; current Full Ours checkpoint unmodified.
