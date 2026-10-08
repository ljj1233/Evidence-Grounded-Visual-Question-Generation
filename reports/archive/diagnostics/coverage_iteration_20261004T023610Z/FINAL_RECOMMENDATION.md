# FINAL RECOMMENDATION — coverage iteration (Phase 1)

Decision: **STOP. Phase-1 gate FAILED. Do not enter D1, do not run RL.**
The observed difference is, at best, a **coverage-quality trade-off**, not a
validated Coverage improvement. No new method should be called a Coverage
improvement from this experiment.

## Direct answers

**1. Is focus-conditioned candidate expansion effective?**
**No, not under the frozen criterion.** A2 (4 focus prompts) raises mean Coverage
by +0.045 raw / +0.0091 normalized over A1 (identical model, seed, decoding,
K=4), but the paired image-bootstrap 95% CI includes 0:
- single run: mean +0.0557 raw, CI [−0.0061, +0.1204]
- 3-run per-image mean: mean +0.0455 raw, CI [−0.0115, +0.1012]

So the Coverage gain is **not statistically distinguishable from zero**. At the
same time Supported falls and Unsupported rises:
- Supported: A1 0.7441 → A2 0.7292 (Δ −0.0149, exceeds the 0.01 allowance)
- Unsupported: A1 0.0986 → A2 0.1054 (Δ +0.0068, within 0.01 but wrong direction)

Per the task's own rule this pattern must be reported as a
**coverage-quality trade-off**, not as an improvement.

**2. Is soft-positive distillation effective?**
**Not tested.** Phase 1 failed, so per protocol D1-hard-A2 and D1-soft-A2 were
not built or trained. No claim is made either way. The target builder and the
optional per-record 0.7/0.3 weighting are implemented and staged
(`coverage_iteration_build_targets.py`, `phase5c_distill.py`) but unrun.

**3. Where would a Coverage change come from?**
Only the **candidate pool / selection** was exercised. A2 changes pool
composition (higher `support_floor` on object/attribute, higher spatial UVFC;
lower object/attribute UVFC and scene support) and yields a small positive mean
selection Coverage, but no significant one. Distillation and RL ran zero steps,
so they contribute nothing here. Compared with the existing A0 baseline
(Direct-4 DIRECT4-prompt → S3 → D1 → G2), A1 is itself a distributional change
(SFT instruction instead of DIRECT4), so the A2−A1 delta isolates the focus
prompts, not the baseline.

**4. Are Supported and Unsupported stable?**
**No.** Across 3 independent evaluator runs on byte-identical questions:
- support-label flips vs run 1: A1 53 / 72 per 1000; A2 96 / 84 per 1000
- SeeQ score-vector flips vs run 1: A1 452 / 470; A2 461 / 482
- Supported swings by up to ~0.03 run-to-run (larger than the 0.01 gate band)

The Supported drop of −0.015 is of the same order as the judge noise, but it
points the wrong way and the Coverage CI still excludes no positive effect; the
gate therefore fails on both the Coverage CI and the Supported condition.

**5. Is it worth entering the formal main experiment?**
**No.** There is no robust evidence that focus-conditioned expansion improves
Coverage, and the Supported/Unsupported signals do not justify spending the
D1 + RL budget. Re-visit only with a pool-level signal that clears the CI gate
or a materially lower-variance evaluator, and re-run Phase 1 before any
distillation.

**6. Which experiments did not pass and why?**
- **A2 candidate pool (Phase 1) — FAILED.** Paired Coverage 95% CI lower bound
  −0.0023 (< 0) over 3 evaluator runs; Supported drop −0.0149 (> 0.01).
- **D1-hard-A2 — NOT RUN** (blocked by Phase-1 gate).
- **D1-soft-A2 — NOT RUN** (blocked by Phase-1 gate).
- **G2 from soft A2 — NOT RUN** (blocked by Phase-1 gate).
- **A0 G2 1000-step rerun — NOT LAUNCHED.** The patience guard is implemented and
  unit-tested (`DEGEN_STREAK=3`; `tests/test_guard_patience.py`), but the
  correction was not needed for Phase 1 and was not run. The existing A0 G2
  remains step 256.
- **Soft-positive weighting** cannot be judged because it never ran; it must not
  be reported as successful.

## Boundaries respected

- No existing method or result overwritten; all new artifacts are under
  `reports/coverage_iteration_20261004T023610Z/`.
- `final_holdout` was never read, calibrated, trained or tuned.
- No new backbone, dataset, external model, GPT judge reward, human label, or
  primary metric. Coverage/Supported/Unsupported remain the only primary metrics.
- Existing UVFC, Flat verifier, S3 selector and τ=0.5 were used unchanged.

## Evidence files

`candidate_pool_stats.json`, `selector_outputs.jsonl`, `paired_metrics.csv`,
`evaluator_variance.csv`, `failure_cases.jsonl`, `ablation_table.md`,
`phase1_gate_variance.json`, `CONFIGS.md`, `PLAN.md`.
