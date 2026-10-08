# FINAL REPORT — Evidence-Calibrated Question Space Expansion Revision

Timestamp: 2026-10-04T08:54:51Z · seed 42 · git `9dea27f0` · GPU 0 (A800-80GB)

## 1. Scope executed

Phase 0 (precheck) → Phase 1 (prompt-aligned Direct-4 dev pool) → Phase 2
(selectors A/B/C) → Phase 3 (selector-only gate). The gate **failed**, so Phases
4–6 (D1 targets, D1 training, preference refinement) were **not run**, per the
stop rule.

## 2. Inputs

| item | value |
|---|---|
| Final-SFT adapter | `outputs/checkpoints/qwen3vl8b_lora_r16_vqav2_50k/checkpoint-final` |
| A0 pool | `outputs/phase5b_direct4_uvfc/candidates_scored.jsonl` — 3811 cand / 1000 img |
| aligned dev pool | `outputs/ec_qse_aligned_direct4/dev/` — 3828 cand / 1000 img |
| dev manifest | `outputs/same_backbone_baselines/manifests/development1000_images.jsonl` |
| A0 external labels | `reports/a0_candidate_oracle_20261004T080307Z/candidate_external_scores.jsonl` |
| calibration split | 700 images · validation 297 images (image-disjoint, seed 42) |

Aligned dev generation: instruction `Generate a question about this image.`,
K=4/T=0.8/top_p=0.9/mn=64/seed=42/sdpa/batch=16, wall 190 s; output sha256
`84806637…`. Aligned dev scoring: UVFC + frozen Flat completed on 3828/3828.
Aligned dev external: SeeQ 3828/3828 ok; GPT-4o 3825/3828 ok (3 parser
failures). Flat states on the aligned pool:
`NO_CONFIRMED_VIOLATION 2312, CONFIRMED_VIOLATION 1279, UNCERTAIN 157, FAILED 80`.

## 3. SELECTOR RESULTS

### Aligned pool, held-out validation (297 images)

| selector | Coverage | Supported | Unsupported | uncertain | regret | oracle= | Pareto-dom | fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | **1.892** | 0.734 | 0.114 | 0.152 | 0.382 | 0.662 | 0.384 | 0.00 |
| S1 | 1.838 | 0.774 | 0.091 | 0.135 | 0.437 | 0.669 | 0.367 | 0.00 |
| S3-original | 1.852 | 0.727 | 0.114 | 0.158 | 0.423 | 0.648 | 0.397 | 0.03 |
| S3-conservative | 1.838 | 0.774 | 0.091 | 0.135 | 0.437 | 0.669 | 0.367 | 0.03 |
| ECS | 1.845 | **0.798** | **0.088** | **0.114** | 0.430 | **0.679** | **0.357** | 0.00 |

### A0 pool, held-out validation (297 images)

| selector | Coverage | Supported | Unsupported | uncertain | regret | oracle= | Pareto-dom | fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 1.832 | 0.768 | 0.108 | 0.125 | 0.452 | 0.680 | 0.380 | 0.00 |
| S1 | 1.801 | 0.795 | 0.084 | 0.121 | 0.483 | 0.680 | 0.364 | 0.00 |
| S3-original | 1.825 | 0.758 | 0.111 | 0.131 | 0.459 | 0.677 | 0.387 | 0.04 |
| S3-conservative | 1.801 | 0.795 | 0.084 | 0.121 | 0.483 | 0.680 | 0.364 | 0.04 |
| ECS | 1.798 | 0.811 | 0.081 | 0.108 | 0.486 | 0.684 | 0.360 | 0.00 |

### Paired bootstrap, aligned pool, vs S3-original (resample images, n=297)

| selector | Coverage delta | 95% CI |
|---|---:|---|
| S0 | +0.0404 | [+0.0067, +0.0808] |
| S1 | −0.0135 | [−0.0572, +0.0303] |
| S3-conservative | −0.0135 | [−0.0572, +0.0303] |
| ECS | −0.0067 | [−0.0673, +0.0505] |

## 4. SELECTOR GATE = FAIL

Checked on the aligned pool against S3-original over 297 held-out validation
images. Conditions: (1) Coverage paired-bootstrap CI lower > 0; (2) Supported ≥
S3 − 0.01; (3) Unsupported ≤ S3 + 0.01; (4) uncertain rate down; (5) Pareto-dom
rate down; (6) improvement not driven by few images.

| selector | c1 | c2 | c3 | c4 | c5 | c6 | all |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| S0 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ (8/293 images changed; 7 positive) | FAIL |
| S1 | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ (11/293) | FAIL |
| S3-conservative | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ (11/293) | FAIL |
| ECS | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ (20/293, +10/−10) | FAIL |

- **ECS fails the main target**: Coverage −0.007 vs S3-original (CI includes 0),
  though it improves Supported (+0.071), Unsupported (−0.026), uncertain
  (−0.044) and Pareto-dom (−0.040). The calibrated selector is a better
  support/uncertainty selector but not a higher-Coverage selector.
- **S3-conservative fails**: removing UNCERTAIN does not raise Coverage.
- **S0 passes conditions 1–5 but fails condition 6**: the +0.040 Coverage over
  S3-original comes entirely from 8/293 images (7 positive; top image = 25% of
  the total). This is a small-subset artefact, not a broad improvement.
- Therefore **no selector passes**. `SELECTOR_REVISION_FAILED`; no D1 training.

## 5. Secondary finding — prompt alignment

Same 297 validation images, S3-original: A0 Coverage 1.825 → aligned 1.852
(+0.027); Supported 0.758 → 0.727 (−0.031); Unsupported 0.111 → 0.114 (+0.003).
Prompt alignment raises Coverage but lowers Supported by ~3 pp, beyond the 0.01
budget, so alignment alone does not meet the constraint either. FACT/EXPLANATION.

## 6. D1 / preference

| phase | status | reason |
|---|---|---|
| D1-original / D1-aligned / D1-ECS | NOT_RUN | selector gate failed |
| preference pairs / DPO | NOT_RUN | gate failed; DPO entrypoint not reached |

## 7. Limitations

- SeeQ judge is the documented `gpt-5.6-luna` variant (prompt hash `57a357e0…`),
  not official GPT-5.4.
- 3 GPT-4o support parser failures on the aligned pool; SeeQ 0 failures.
- Threshold and ECS were fit on the 700-image calibration split; validation was
  used only for gating.
- `development1000` is method-selection data.

## 8. Artifacts

Report: `reports/ec_qse_revision_20261004T085451Z/` — `PRECHECK.md`,
`PROTOCOL_FIX.md`, `selector_calibration_split.json`, `selector_results.csv`,
`selection_regret.csv`, `selector_gate.json`, `_bootstrap.json`,
`ecs_model.joblib`, `FINAL_REPORT.md`, `DECISION.md`,
`SELECTOR_REVISION_FAILED.md`, and NOT_RUN stubs for the D1/preference files.
Aligned data: `outputs/ec_qse_aligned_direct4/`.
