# FINAL HOLDOUT REPORT — Full Ours and Full Ours + Offline DPO Refinement

Frozen before any holdout read/generation: `FINAL_EVAL_FREEZE.json` / `.md`.
Systems (internal, frozen):
- Full Ours — `outputs/phase5c_qse_relativeguard/checkpoints/step_final`
- Full Ours + DPO — `reports/offline_dpo_unified_refinement_20261008/dpo/checkpoints/step_00207`
  (selected on DEV256, confirmed on DEV744)

Integrity (`FINAL_HOLDOUT_INTEGRITY.json`): final_holdout manifest sha
`eedd8b3ae1164729e166429cd6d467a22c67ca658e5ce2fc62194c24a414f54c`, **2000 unique IDs**,
**zero overlap** with SFT/Base train, formal method-train, 6633 DPO pairs, historical
DPO train/pilot, UVFC discovery/calibration, DEV256, DEV744, development1000, val_debug.

Generation: identical greedy protocol (batch 16, max_new_tokens 32, seed 42) for both
systems; 2000/2000 records each, 0 empty/malformed, no deterministic runtime failure.
SeeQ (`gpt-5.6-luna`) and GPT-4o support (`visual-support-phase23-v1.0`) run once;
one bounded retry of 8 transient SeeQ HTTP failures.

## 1. Full Ours (final_holdout)

Supported **0.8580**, Unsupported **0.0445**, Uncertain 0.0975; Coverage **0.3850**,
Search **0.3684**, Context **0.2739**, Spatial **0.0637**; BLEU-4 0.2645, CIDEr 0.4215,
Q-BLEU4-v 0.2558.

## 2. Full Ours + DPO (final_holdout)

Supported **0.9365**, Unsupported **0.0195**, Uncertain 0.0440; Coverage **0.4244**,
Search **0.4166**, Context **0.2296**, Spatial **0.0586**; BLEU-4 0.2595, CIDEr 0.4262,
Q-BLEU4-v 0.2508.

## 3. Does the DPO refinement reproduce the development gains? (paired, n=10000/seed42)

| metric | delta (DPO − Full Ours) | 95% CI | reproduced |
|---|---:|---|---|
| Supported | **+0.0785** | [+0.0640, +0.0930] | **yes** |
| Unsupported | **−0.0250** | [−0.0340, −0.0160] | **yes** |
| Coverage | **+0.0394** | [+0.0310, +0.0479] | **yes** |
| Search | **+0.0482** | [+0.0406, +0.0561] | **yes** |

All four primary directions reproduce on the sealed holdout with CIs excluding zero.

## 4. Context on final_holdout

Context **−0.0443** (CI [−0.0489, −0.0397]) — the degradation is real but **smaller than
bs8-original DPO (−0.064)** and consistent with the development sets (−0.051 / −0.047).
Spatial −0.0050 (CI crosses 0).

## 5. Are BLEU/CIDEr/Q-BLEU4-v preserved?

Largely yes: BLEU-4 0.2645→0.2595 (−0.005), CIDEr 0.4215→0.4262 (+0.005),
Q-BLEU4-v 0.2558→0.2508 (−0.005). Question length 5.85→6.25; exact duplicate
0.4665→0.5045; broad type JS(Base→DPO) 0.0497.

## 6. Generation / evaluator failures

Generation: 0 empty/malformed of 2000 per system. Support: 0 failures (2000 each).
SeeQ: 8 error records over **5 unique images** (`13525, 223764, 500477, 566596, 95516`),
persistent HTTPStatusError after one bounded retry; kept missing.
Per-system SeeQ valid n = 1995; common paired valid n = 1995. No imputation.

## 7. Directional consistency (never pool holdout with development)

| delta (DPO − Full Ours) | DEV256 | DEV744 | FINAL_HOLDOUT |
|---|---:|---:|---:|
| Supported | +0.082 | +0.085 | +0.0785 |
| Unsupported | −0.027 | −0.013 | −0.0250 |
| Coverage | +0.053 | +0.051 | +0.0394 |
| Search | +0.049 | +0.063 | +0.0482 |
| Context | −0.051 | −0.047 | −0.0443 |
| Spatial | +0.017 ns | 0.000 ns | −0.0050 ns |
| broad type JS(Base→DPO) | 0.046 | 0.055 | 0.0497 |

DEV256, DEV744 and final_holdout are directionally consistent. The final_holdout
result is treated as final.

## Tables

`FINAL_INTERNAL_TABLE.csv` / `.md`, `FINAL_DPO_DELTA_TABLE.md`,
`FINAL_PUBLISHED_BASELINES_TABLE.md` (intentionally empty: no published-baseline
holdout results under the frozen protocol). `LOCAL_METRICS.json`,
`LOCAL_COMPARISON.csv`, `GENERATION_ACCOUNTING.json`, `FINAL_STATS.json`.

## Hard stop

No checkpoint change, retraining, step/pair/LR/beta/batch/scheduler/decoding/evaluator
change, seed search, or holdout rerun. The final_holdout result is final.
