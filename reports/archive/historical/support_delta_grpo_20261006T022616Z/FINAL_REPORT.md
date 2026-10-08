# FINAL REPORT — Support-Delta GRPO pilot

METHOD: Support-Delta GRPO — **internal training objective / candidate method variant** (not a paper novelty).

## Headline

- **STATUS: FAIL_PILOT**. Reward sanity PASSED; the 300-step pilot completed; the 256-image external pilot gate FAILED.
- Formal 1000-step training was **NOT started**. `final_holdout` was **NOT accessed**.

## FACT

1. Reward sanity check: **PASSED** (128 train images): supported S=0.953 > unsupported S=-0.703; delta_support has both signs; zero-variance group rate 0.031 (<0.30); coverage_guard 0.378 (<0.50); malformed 0.017 (<0.05); no template collapse (0.016).
2. 300-step pilot: completed 300 steps, `stop_reason=null`. ratio mean range [0.9781917333602905, 1.0236783027648926] (varies); clip_fraction range [0.0, 0.09999999403953552] (>0); KL range [0.0, 0.0070013199001550674] (<=0.02).
3. External (256 fixed-output images, GPT-4o support + SeeQ coverage):
   - Supported delta **-0.0039** 95% CI [-0.0273, +0.0195]
   - Unsupported delta **+0.0078** 95% CI [-0.0117, +0.0273]
   - Coverage delta **+0.0156** 95% CI [+0.0039, +0.0281]
   - original-shuffled support gap delta **-0.0156** CI [-0.0508, +0.0195]
   - original-blurred support gap delta **+0.0117** CI [-0.0234, +0.0469]
4. Local Flat on the same 256 (original): Supported 0.76953125→0.765625; duplicate 0.18359375→0.18359375; generic 0.0→0.0; length 7.0859375→7.21875.
5. Gate: {"supported_point_ge_0.015": false, "supported_ci_lower_gt_0": false, "coverage_ci_lower_ge_-0.01": true, "unsupported_point_lt_0": false, "unsupported_ci_upper_le_0": false, "orig_shuffled_gap_increase": false, "duplicate_increase_le_0.05": true, "generic_increase_le_0.03": true, "length_change_le_10pct": true, "kl_le_0.01": true, "ratio_not_identity": true, "clip_fraction_positive": true, "evaluator_repeat_3": false, "no_final_holdout": true}.

## Required answers

1. Reward sanity PASS. 2. zero-variance group rate 0.031 (sanity) / per-step reported in logs. 3. coverage_guard trigger rate 0.378 (sanity). 4. ratio truly varies (range above). 5. clip_fraction range 0.0-0.10. 6. KL range up to 0.0070. 7. training completed 300 steps. 8. pilot FAIL. 9. 1000-step NOT started. 10-12. Coverage/Supported/Unsupported deltas and CIs above. 13. duplicate +0.000, generic +0.000, length +1.9%. 14. original/blur/shuffle gaps above. 15. final_holdout NOT accessed. 16. Failure: the objective moved Coverage up (+0.0156) but Supported down (-0.0039) and Unsupported up; the reward's relative-to-q0 delta was already near zero because rollouts are close to the base anchor, so the advantage mostly reweighted coverage rather than support. 17. Code: `src/vvqg/support_delta/*`, `scripts/vvqg/support_delta_*.py`, `tests/test_support_delta.py`. 18. Checkpoints: `pilot_final/checkpoints/step_0000..step_0300,step_final`.

## INTERPRETATION

- This is a **genuine negative pilot result**: the reward is not reward-sanity-broken, the optimizer moved the policy (ratio/clip/KL all active), yet external Supported did not improve and Unsupported did not fall.
- The reward's `delta_support = S(q_i) - S(q_0)` is dominated by cases where `S(q_i)=S(q_0)` (both high/flat), so the gradient is largely driven by the coverage guard and duplicate penalty, which is consistent with Coverage rising and Supported staying flat.
- Do not claim Support-Delta GRPO improves Supported.

## UNKNOWN

- Whether a different margin/baseline (e.g. absolute S rather than q0-relative) would change Supported (not tested; changing it now would be a new experiment).
- Whether 3 evaluator repeats would change the direction (only 1 repeat run).
- Whether groups_per_step>1 would reduce gradient noise enough to matter.

## Next single action

Stop. Do not start formal training. The one pre-registered follow-up to consider is to isolate why the support term is inert (fraction of rollouts with `delta_support != 0` and their reward contribution) before changing the objective.
