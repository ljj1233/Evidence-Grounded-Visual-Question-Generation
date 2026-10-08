# FINAL REPORT — Risk-aware DPO-v2 (relaxed count guard), fresh-256 pilot

New data contract: pair-count guard relaxed to S>U≥25, U>X≥25, S>X≥100; the UVFC hard filter was
removed. S/U/X definitions, support_floor threshold and loss weights unchanged. Training was allowed
this time. No `final_holdout` access; no dev1000/DPO-v3/IPO/GRPO/DAPO started.

## Headline

- **STATUS: FAIL_KL_GUARD**
- Training ran 100 steps. **KL(step100) = 0.495 > 0.10** (step50 = 0.025); the pre-registered KL guard fires.
- Supported(Base→v2) = +0.0039 (CI [−0.0156, +0.0234]) — the DPO-v1 gain collapsed.
- Coverage(Base→v2) = −0.0008 (CI [−0.0102, +0.0078]) — CI lower crosses the −0.01 margin.
- Unsupported(Base→v2) = −0.0117 (CI [−0.0313, +0.0039]); Uncertain +0.0078.
- Original DPO-v1 on the same fresh256: Supported +0.0391 (CI [+0.0039, +0.0742]), Unsupported −0.0156 (CI [−0.0313, −0.0039]), Coverage +0.0344.

## FACT

- Data: train 578 images, fresh pilot 256 (overlap 0 with train, 0 with old pilot, 0 with final_holdout).
  Train pairs (no UVFC filter): S>U 67, U>X 67, S>X 578; class-image counts S 578 / U 67 / X 578.
  Guards pass (S>U≥25, U>X≥25, S>X≥100) (`PAIR_COUNTS.json`).
- Training (`dpo_v2/training_summary.json`, `TRAIN_LOG.jsonl`): 100 steps, β=0.1, lr=2.5e-6, weighted
  loss `1.0·L(S>U)+1.5·L(U>X)+0.25·L(S>X)`, reference = Full Ours `step_final`.
  KL step50 0.0246 → step100 0.4952 (`kl_guard_fail = true`). Mean per-type preference accuracy:
  S>U 0.65, U>X 0.55, S>X 0.75. Duplicate rate 0; malformed 0; peak GPU 49.6 GB.
- Generation on fresh256 for Base / DPO-v1 / DPO-v2: 256 each, 0 empty, 0 failures.
- Single evaluator pass (SeeQ + GPT-4o): 0 parser failures, valid_n 256/256/256 for both evaluators;
  1143 new API calls.

Paired (image-level bootstrap 5000, seed 42):

| comparison | metric | delta | 95% CI | up/down/tied |
|---|---|---:|---|---|
| Base→DPO-v1 | coverage | +0.0344 | [+0.0117, +0.0563] | 33/13/210 |
| Base→DPO-v1 | supported | +0.0391 | [+0.0039, +0.0742] | 16/6/234 |
| Base→DPO-v1 | unsupported | −0.0156 | [−0.0313, −0.0039] | 0/4/252 |
| Base→DPO-v2 | coverage | −0.0008 | [−0.0102, +0.0078] | 6/6/244 |
| Base→DPO-v2 | supported | +0.0039 | [−0.0156, +0.0234] | 4/3/249 |
| Base→DPO-v2 | unsupported | −0.0117 | [−0.0313, +0.0039] | 1/4/251 |
| Base→DPO-v2 | uncertain | +0.0078 | [−0.0078, +0.0234] | 3/1/252 |
| DPO-v1→DPO-v2 | supported | −0.0352 | [−0.0703, +0.0000] | — |

Transition matrix (Base→DPO-v2): supported→supported 219, supported→uncertain 2, supported→unsupported 1,
uncertain→supported 1, uncertain→uncertain 22, uncertain→unsupported 0, unsupported→supported 3,
unsupported→uncertain 1, unsupported→unsupported 7. Coverage-vs-Unsupported delta Pearson −0.153.

## Gate (Base→v2)

| condition | value | result |
|---|---|---|
| Supported delta > 0 | +0.0039 | pass (weak) |
| Unsupported delta ≤ +0.01 | −0.0117 | pass |
| Coverage CI lower ≥ −0.01 | −0.0102 | **fail (marginal)** |
| Uncertain delta ≤ 0 | +0.0078 | **fail** |
| KL ≤ 0.10 | 0.495 | **fail** |
| generation failures = 0 | 0 | pass |
| parser ≤ max(Base,v1) | 0 ≤ 0 | pass |

→ **FAIL_KL_GUARD**.

## INTERPRETATION

- The risk-aware reweighting did **not** preserve the DPO-v1 Supported gain; it erased it (Base→v2
  +0.0039 vs Base→v1 +0.0391). The model diverged from the reference (KL 0.495), which likely explains
  the regression.
- DPO-v1 itself looks favorable on this fresh 256 (Supported +0.039, Unsupported −0.016, Coverage +0.034),
  unlike its earlier 128-image behavior; this is a new-sample result, not a re-interpretation of the old pilot.
- The v2 Unsupported mean (−0.0117) is within +0.01, but Coverage and Uncertain gates fail and KL is far
  over budget, so v2 is not usable.

## UNKNOWN

- Whether the DPO-v1 fresh256 gain generalizes (single evaluator pass; 256 images from the aligned pool).
- Whether a step50 v2 checkpoint (KL 0.025) would behave differently — not evaluated (pre-registered
  checkpoint is step100; no result-based checkpoint selection).

## Answers

1. v2 did **not** preserve the DPO-v1 Supported gain (Base→v2 +0.0039, CI crosses 0; Base→v1 +0.0391).
2. Unsupported Base→v2 = −0.0117, inside +0.01.
3. Coverage Base→v2 = −0.0008 with CI lower −0.0102, just outside the −0.01 margin.
4. Main transitions: mostly unchanged; supported→uncertain 2, supported→unsupported 1, unsupported→supported 3.
5. Expand to dev1000: **NO**.
6. No follow-up training started.
