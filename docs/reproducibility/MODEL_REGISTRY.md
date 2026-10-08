# Model registry

Verified paths and recorded run metadata, 2026-10-08. Paths are relative to the original research workspace unless explicitly absolute. No weights are included in the Git-ready copy.

Status vocabulary: `FINAL_MAIN`, `FINAL_OPTIONAL_REFINEMENT`, `ACTIVE_BASELINE`, `ACTIVE_CROSS_DATASET`, `HISTORICAL_REMOVED`. `FINAL_MAIN` includes frozen lineage ancestors; the row named **Full Ours** is the sole final paper main model.

SHA256 below names the exact weight file(s). B2/B3 require both adapter and method head. Released LOVA3/Genixer weight hashes are not recorded; their known config hashes/revisions below are **not weight hashes**. Mainline hashes were verified during checkpoint cleanup; remaining adapter/head hashes were computed read-only for this registry.

## VQAv2 main lineage

| model_name | role | checkpoint_path | global_step | status | sha256 | paper_usage |
|---|---|---|---:|---|---|---|
| Final SFT | Frozen mainline ancestor / teacher | `outputs/checkpoints/qwen3vl8b_lora_r16_vqav2_50k/checkpoint-final` | 6250 | FINAL_MAIN | `adapter_model.safetensors`: `5fb66c36a2255fb3a9011db198fd55a8be8b9cbb26eef611727a816bc7934e0e` | 50k VQAv2 image→question SFT; UVFC/teacher source; D1 initialization |
| D1 | Frozen mainline ancestor / distillation | `outputs/phase5c_distill_single/checkpoint-final` | 1250 | FINAL_MAIN | `adapter_model.safetensors`: `0c8e51148b9f857d26d5bf811e7217461274b5de37fb49bd0b7265c174a280a7` | Single-positive QSE distillation; G2 initialization/reference |
| Full Ours | FINAL PAPER MAIN MODEL | `outputs/phase5c_qse_relativeguard/checkpoints/step_final` | 256 | FINAL_MAIN | `adapter_model.safetensors`: `52ee692ca08737058594fe408a7817419f53532395f9249b2f227b43b4d9a58d` | Primary VQAv2 method; frozen final-holdout evaluation |
| Full Ours + DPO | FINAL OPTIONAL DPO MODEL | `reports/offline_dpo_unified_refinement_20261008/dpo/checkpoints/step_00207` | 207 | FINAL_OPTIONAL_REFINEMENT | `adapter_model.safetensors`: `32384eea130274b29be1f1e8a9882faae56c0db274ce219882f07a8bd9c7f07e` | Optional post-training refinement; selected/frozen step207 |

## VQAv2 comparison baselines

| model_name | role | checkpoint_path | global_step | status | sha256 | paper_usage |
|---|---|---|---:|---|---|---|
| B2 ConVQG | Same-backbone published-method comparison | `outputs/same_backbone_baselines/b2_contrastive/full_fast` | 6250 | ACTIVE_BASELINE | `backbone/adapter_model.safetensors`: `4c4f3ac2437a13723f2a9904093bdbf56bf62746a834ea6536dab97c5c2be443`<br>`contrastive_projection.pt`: `85c4a550427fbb130cc211d0d36870fe55156ec5197a9d344de7f852871b098a` | VQAv2 development comparison; not a Full Ours ancestor |
| B3 Guiding-VQG | Same-backbone published-method comparison | `outputs/same_backbone_baselines/b3_implicit/full_fast` | 6250 | ACTIVE_BASELINE | `backbone/adapter_model.safetensors`: `4b9f4fbabf346270443a461daa2c6785c12eee64517c6466a401cf9a8582e04b`<br>`guided_head.pt`: `797579d61281eaa7d3859bf8b657ff08d10e85cf30a4763ddba069ff81c54707` | VQAv2 development comparison; not a Full Ours ancestor |
| Faithful SeeQ M1 | Same-backbone baseline ancestor | `outputs/same_backbone_baselines/seeq_faithful/round1_m1/checkpoint-final` | 10000 | ACTIVE_BASELINE | `adapter_model.safetensors`: `d557d3f90d16bc3d2712c8bd63269dd6a77e30b68ffe3c2135eee435ed299aed` | Development1000 comparison; no frozen-protocol final-holdout predictions |
| Faithful SeeQ M2 | Same-backbone baseline final | `outputs/same_backbone_baselines/seeq_faithful/round2_m2/checkpoint-final` | 6961 | ACTIVE_BASELINE | `adapter_model.safetensors`: `63fe2a54d3b59a9edd32b4bd1d6a2c0bfde32cb8ea53e55c9227a0ffabf26321` | Development1000 comparison; no frozen-protocol final-holdout predictions |

## Released published baselines

| model_name | role | checkpoint_path | global_step | status | sha256 | paper_usage |
|---|---|---|---:|---|---|---|
| LOVA3 GenQA | Released published checkpoint | `/hpc2hdd/home/fye374/models/LOVA3-llava-v1.5-7b` | NOT_RECORDED (released pretrained model) | ACTIVE_BASELINE | WEIGHT_SHA256_NOT_RECORDED | Inference-only VQAv2/GQA development comparison; not VQAv2 ancestry |
| Genixer-L Common-VQA | Released published checkpoint | `/hpc2hdd/home/fye374/models/Genixer-llava-v1.5-7b` | NOT_RECORDED (released pretrained model) | ACTIVE_BASELINE | WEIGHT_SHA256_NOT_RECORDED | Inference-only VQAv2/GQA development comparison; not VQAv2 ancestry |

## GQA-native models

| model_name | role | checkpoint_path | global_step | status | sha256 | paper_usage |
|---|---|---|---:|---|---|---|
| GQA SFT | ACTIVE COMPARISON / CROSS-DATASET | `outputs/gqa_native/full_ours/sft/checkpoint-final` | 6250 | ACTIVE_CROSS_DATASET | `adapter_model.safetensors`: `8c3cf2d2359efda3723ff041c87d089c7c672548de891fa3e15b2bb236c1d3f4` | Separate GQA SFT direction; not the base-init GQA G2 ancestor |
| GQA base-init D1 | ACTIVE COMPARISON / CROSS-DATASET | `outputs/gqa_native/full_ours_base/distill_d1/checkpoint-final` | 1250 | ACTIVE_CROSS_DATASET | `adapter_model.safetensors`: `17c66b429aed33655e76e280d79bb564dac7b79d607e8e14c73f8897f4a14b63` | Current GQA base-init G2 initialization/reference |
| GQA Full Ours (base-init) | ACTIVE COMPARISON / CROSS-DATASET | `outputs/gqa_native/full_ours_base/g2/checkpoints/step_final` | 1000 | ACTIVE_CROSS_DATASET | `adapter_model.safetensors`: `5a0c720f07c3eb33ea668add63c5e39423ebf2bffeae12d15af7a1ff9c2777a7` | Current GQA final comparison row; no VQAv2 learned weights reused |

## GQA comparison baselines

| model_name | role | checkpoint_path | global_step | status | sha256 | paper_usage |
|---|---|---|---:|---|---|---|
| GQA B2 ConVQG (10k) | Current GQA final comparison baseline | `outputs/gqa_native/baselines_10k/b2_contrastive/full_fast` | 1250 | ACTIVE_BASELINE | `backbone/adapter_model.safetensors`: `fe8f984217b98b856f752bd7188dc3e4a1de8ee303de257b2d1343392b786b44`<br>`contrastive_projection.pt`: `3c30c51e6ed3482ac65caaceb9fbebac69e3d07b2057f5f6324c9b3de8e3d0ee` | Chosen by gqa_full_ours_final_report.py; 10k method-training comparison |
| GQA B3 Guiding-VQG (10k) | Current GQA final comparison baseline | `outputs/gqa_native/baselines_10k/b3_implicit/full_fast` | 1250 | ACTIVE_BASELINE | `backbone/adapter_model.safetensors`: `26163110bd2505c99f030510c69184fe1912aab3c8e4164cbb87d0331036b3e9`<br>`guided_head.pt`: `255ff18ffa8b6b212c60705fd51df11e03b2626216dd4f5201ea629b9cd567c1` | Chosen by gqa_full_ours_final_report.py; 10k method-training comparison |
| GQA SeeQ M1 | GQA baseline ancestor | `outputs/gqa_native/published_baselines/seeq/round1_m1/checkpoint-final` | 1392 | ACTIVE_BASELINE | `adapter_model.safetensors`: `d458a8d3b9466d89e9907153cd9a59b814e41f241a4ff976db229172ed91e869` | GQA final comparison uses M2; M1 retained as its ancestor |
| GQA SeeQ M2 | GQA baseline final | `outputs/gqa_native/published_baselines/seeq/round2_m2/checkpoint-final` | 1427 | ACTIVE_BASELINE | `adapter_model.safetensors`: `4ab15ba8306330b3b859ceb24096fe6c3d2f38da91328716d925342e782bc079` | GQA final comparison uses M2; M1 retained as its ancestor |

## Provenance and retained comparison variants

- VQAv2 steps/hashes: training summaries, checkpoint `CHECKPOINT.json`, and `reports/offline_dpo_freeze_20261008/FREEZE_RECORD.md`.
- GQA selection: `scripts/vvqg/gqa_full_ours_final_report.py` references base-init G2, 10k B2/B3, GQA SeeQ M2, LOVA3 and Genixer. The separate GQA-SFT path is retained but is not the base-init model ancestor.
- B2/B3 final-root step provenance: corresponding terminal `checkpoint-6250/trainer_state.json` (VQAv2) or `checkpoint-1250/trainer_state.json` (GQA 10k), plus their root final-save training path. Other protected checkpoints, including GQA broader-data B2/B3, remain retained; this registry lists the models selected by current comparison entry points, not every intermediate save.
- LOVA3 GenQA: release revision `0a7c184c5d37112f038d3ebb3c612036fa6d7991`; **config.json only** SHA256 `94a7b0cb34e74fff78fb2a25a3290fe29e73b34d0fe62ab065e3d18feca816f1`; source `outputs/published_vqg_baselines/download_manifest.json`.
- Genixer-L Common-VQA: release revision `9401b250a704a0806618d508664f24d69057dda4`; **config.json only** SHA256 `d0e6a66c6425ccef52e85b2bb7ecc243eea35db6115476ad8f6ec215e1fb3323`; source `outputs/published_vqg_baselines/download_manifest.json`.

## HISTORICAL_REMOVED

153 historical checkpoint payloads were removed in the completed checkpoint-only cleanup. They are not active models. Use [DELETE_LOG.tsv](audits/checkpoint_cleanup_20261008/DELETE_LOG.tsv) and per-experiment `CHECKPOINT_REMOVED.md` for exact paths, steps, hashes and replacements. No additional checkpoint deletion is part of this snapshot stage.
