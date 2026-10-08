# Current project

Current state: 2026-10-08. This is the entry point for the active paper and comparisons. Checkpoint paths refer to the original research workspace; the Git-ready backup keeps code and lightweight records, with large assets registered separately.

Fresh-clone setup and test policy: [REPRODUCIBILITY.md](docs/reproducibility/REPRODUCIBILITY.md). Current repository verification: [GIT_READINESS_FINAL.md](docs/reproducibility/GIT_READINESS_FINAL.md). These portability changes do not redefine the frozen method or model lineage.

## Final VQAv2 lineage

```text
Public Qwen3-VL-8B-Instruct
  → Final SFT (6250)
  → D1 (1250)
  → Full Ours / G2 step256
  → optional DPO step207
```

| Logical name | Checkpoint path | Status | Paper role | Training / inference role |
|---|---|---|---|---|
| Public Qwen3-VL-8B-Instruct | `~/models/Qwen3-VL-8B-Instruct` | Public frozen backbone | Shared backbone | Base parameters under the adapters |
| Final SFT | `outputs/checkpoints/qwen3vl8b_lora_r16_vqav2_50k/checkpoint-final` | FINAL_MAIN | Frozen mainline ancestor / teacher | 50k VQAv2 image→question SFT; UVFC/teacher source; D1 initialization |
| D1 | `outputs/phase5c_distill_single/checkpoint-final` | FINAL_MAIN | Frozen mainline ancestor / distillation | Single-positive QSE distillation; G2 initialization/reference |
| Full Ours | `outputs/phase5c_qse_relativeguard/checkpoints/step_final` | FINAL_MAIN | FINAL PAPER MAIN MODEL | Primary VQAv2 method; frozen final-holdout evaluation |
| Full Ours + DPO | `reports/offline_dpo_unified_refinement_20261008/dpo/checkpoints/step_00207` | FINAL_OPTIONAL_REFINEMENT | FINAL OPTIONAL DPO MODEL | Optional post-training refinement; selected/frozen step207 |

**FINAL PAPER MAIN MODEL:** `outputs/phase5c_qse_relativeguard/checkpoints/step_final`

**FINAL OPTIONAL DPO MODEL:** `reports/offline_dpo_unified_refinement_20261008/dpo/checkpoints/step_00207`

Final SFT and D1 are retained ancestors in the FINAL_MAIN lineage; only Full Ours is the final paper main model. DPO is an optional post-training refinement, not a component of the Full Ours training objective.

### Current method and entry points

- Full Ours G2: `scripts/vvqg/train_online_uvfc_relative.py` + `configs/phase5c/qse_relativeguard.yaml`; initializes from D1.
- D1 targets/training: `scripts/vvqg/phase5c_build_targets.py` and `scripts/vvqg/phase5c_distill.py`; Direct-4 → UVFC/Flat scoring → S3 QSE selection → single-positive distillation.
- UVFC assets: `outputs/phase26/vfc/sae.pt` and `weights.pt`; required training/scoring assets, retained outside Git.
- Optional DPO: `scripts/vvqg/dpo_v1_formal_train.py`; frozen unified-pair run selects step207.
- Inference: `scripts/vvqg/generate.py`; select the frozen checkpoint explicitly. UVFC/QSE/RelativeGuard are training/selection components; final image→question inference uses the backbone and selected adapter.
- Read [current method code map](docs/method/CURRENT_METHOD_CODEMAP.md), [technical specification](docs/method/CURRENT_METHOD_TECHNICAL_SPEC.md), and [model registry](docs/reproducibility/MODEL_REGISTRY.md) for evidence and exact hashes.

## ACTIVE COMPARISON / CROSS-DATASET

These are independent comparison directions, **not ancestors of VQAv2 Full Ours**. Active means retained for the research direction; it does not assert a currently running training job.

| Direction | Current artifacts / entry point | Paper role |
|---|---|---|
| GQA-native | `outputs/gqa_native/full_ours_base/{distill_d1,g2}/`; `scripts/vvqg/gqa_full_ours_final_report.py` | Current base-init GQA G2 completes 1000 steps; cross-dataset comparison. Separate GQA SFT artifacts remain under `outputs/gqa_native/full_ours/`. |
| Published baselines | `outputs/published_vqg_baselines/`, `outputs/same_backbone_baselines/`, GQA `published_baselines/` and `baselines_10k/` | LOVA3, Genixer, faithful SeeQ, B2 ConVQG and B3 Guiding-VQG; distinct from mainline ancestry. GQA final comparison uses 10k B2/B3. |
| Final evaluation infrastructure | `scripts/vvqg/evaluate.py`, visual-support/visual-centric/diversity evaluators; `reports/final_evaluation_20261008/` and `reports/offline_dpo_freeze_20261008/` | Frozen VQAv2 Full Ours versus optional DPO holdout evaluation and provenance. |

Published-baseline results currently cover development comparisons. The [final published-baseline table](reports/baselines/FINAL_PUBLISHED_BASELINES_TABLE.md) explicitly has no frozen-protocol final-holdout baseline predictions; development scores must not be presented as holdout scores.

## Historical material

Historical/diagnostic scripts, configs, docs and small reports are organized under their `archive/` directories. Removed checkpoint directories contain `CHECKPOINT_REMOVED.md`; those paths are records, not loadable models. Current helper dependencies remain in the live code tree.

See [archive index](ARCHIVE_INDEX.md), [historical phase map](docs/archive/HISTORICAL_PHASE_MAP.md), and [checkpoint cleanup report](docs/reproducibility/audits/checkpoint_cleanup_20261008/FINAL_REPORT.md). This page is the current entry point; older status narratives and contracts are time-stamped evidence and may describe plans rather than completed runs.
