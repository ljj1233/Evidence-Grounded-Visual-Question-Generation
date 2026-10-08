# FINAL REPORT — Evidence-Constrained VQG Pilot (CER)

METHOD: Counterfactual Evidence Regularization (CER) — **candidate method under evaluation**.
Not a published method; not causal grounding; no novelty claim.

## Headline

- **STATUS: FAIL_LOCAL**
- Phase A (64-image local code/loss validation) was reached. The 8-image smoke and the
  full-eligible delta_cf diagnostic both show that the counterfactual term is **inactive**:
  `L_cf ≡ 0`.
- Phase B (256-image external pilot) and Phase C (1000-step formal) were **not started**.
- `final_holdout` was **NOT accessed**.

## 1. FACT — Base checkpoint (Phase 0)

- `outputs/phase5c_qse_relativeguard/checkpoints/step_final`, `global_step = 256` (is_256 = True).
- file sha256 parity with CHECKPOINT.json: True.
- adapter sha256 `52ee692ca08737058594fe408a7817419f53532395f9249b2f227b43b4d9a58d`.
- canonical greedy reproduction: 1000/1000 (pass = True).
- train ∩ development1000 = 0; train ∩ fixed 256 pilot = 0 (audited on image stems).

## 2. FACT — Phase A local data and anchors

- Local counterfactual subset: 96 train images, 0 derangement fixed points, blur radius 4-19.
- Anchor generation: 96 greedy Base questions; 84 eligible; 12 skipped ({'confirmed_violation': 11, 'parser_failed': 1}).
- Skipped anchors are retained in `local64/anchors.jsonl` with `skipped_for_evidence_loss=true`.

## 3. FACT — 8-image smoke

- steps = 2; loss finite = True;
  pairing_ok = True; LoRA updated = True;
  base checkpoint unchanged = True.
- `L_cf` value = 0.0 at every step (`l_cf_nonzero = False`).
- anchor term is wired and trains (grad norm > 0; LoRA parameters move, see `KL_DIAGNOSTICS.csv`).
- KL = 0.0 at step 1 (policy == reference, KL minimum), 0.00246 at step 2 (wired; grows after the update).

## 4. FACT — delta_cf distribution over all 84 eligible train anchors (margin 0.5)

| stat | value |
|---|---:|
| delta_cf mean | 4.0067135487638765 |
| delta_cf std | 1.5030658391758287 |
| p10 | 2.1775399148464203 |
| p50 | 3.9131470024585724 |
| p90 | 5.9681086987257 |
| min | 0.3384759724140167 |
| max | 7.509949862957001 |
| fraction with L_cf active | 0.011904761904761904 |
| mean margin loss | 0.0019229050903093246 |

## 5. Required questions

1. How far did CER run: **Phase A only**.
2. Passed local smoke: **NO** (L_cf non-zero gate failed).
3. Passed 256-image external pilot: **NOT RUN** (blocked).
4. Ran 1000-step formal training: **NO**.
5. Passed final acceptance: **NO**.
6-8. Coverage / Supported / Unsupported deltas and 95% CI: **UNKNOWN** (external evaluator not run).
9. original-shuffled Support gap: **UNKNOWN** (external). Local log-prob gap at init = 4.0067135487638765 nats (already large).
10. original-blurred Support gap: **UNKNOWN**.
11. duplicate / generic / length / malformed changes: **UNKNOWN for a trained CER** (no training signal produced).
12. Did delta_cf increase: **NO** (counterfactual loss never contributes gradient).
13. KL over budget: **NO** (KL at step 2 = 0.00246; but see §6).
14. Evaluator stable: **NOT RUN**.
15. final_holdout accessed: **NO**.
16. Source files changed: `src/vvqg/evidence_constrained/*`, `scripts/vvqg/evidence_constrained_*.py`, `tests/test_evidence_constrained.py`.
17. Checkpoints generated: `local64/smoke/checkpoints/step_0000`, `step_final` only (8-image smoke).
18. Failure class: **proxy failure / local optimization failure** (implementation verified correct).

## 6. INTERPRETATION

- The Base policy already assigns the anchor question ~4.0 nats/token more probability on the
  original image than on its shuffled counterfactual. The mandated `margin=0.5` is therefore
  satisfied for 83/84 anchors, so `L_cf = mean(relu(0.5 - delta_cf)) ≈ 0.0019` and its
  gradient is identically zero. CER reduces to `L_anchor + lambda_kl * KL`, which only
  reinforces the Base behaviour and cannot increase visual dependence.
- The smoke `cf_has_gradient=false` is not an autograd bug: `delta_cf` is differentiable and in
  the graph, but the ReLU is saturated. The `kl_has_gradient=false` at step 1 is expected
  (KL is at its minimum, gradient 0); KL is wired and becomes non-zero after the first update.
- Therefore the counterfactual evidence constraint, as specified (fixed margin 0.5), has no
  room to act on this Base checkpoint.

## 7. UNKNOWN

- Whether CER with an active margin changes external Supported/Unsupported/Coverage.
- Whether the external evaluator agrees with the local log-prob gap.
- Whether blur is a better counterfactual than shuffle for this policy.

## 8. Paired examples (Base anchors, 20)

See `PAIRED_EXAMPLES.jsonl`. Each row is a Base anchor with `lp_plus`, `lp_minus`, `delta_cf`;
the CER columns are `null` because no trained CER checkpoint exists.

## 9. Next single action (do not auto-retune)

Re-register one new experiment that changes **only the margin**: set `margin` above the Base
`delta_cf` distribution (e.g. the p75-p90 band ≈ 3.0-5.0) so `L_cf` is active on a meaningful
fraction of anchors, then re-run Phase A. Do not start Phase B/C before that passes.
