# FREEZE RECORD — Full Ours + Offline DPO Refinement

Frozen 2026-10-08. No further DPO optimization or checkpoint/hyperparameter selection
is permitted after this freeze. `final_holdout` remains sealed.

## Candidate

| field | value |
|---|---|
| name | Full Ours + Offline DPO Refinement |
| checkpoint | `reports/offline_dpo_unified_refinement_20261008/dpo/checkpoints/step_00207` |
| adapter_model.safetensors sha256 | `32384eea130274b29be1f1e8a9882faae56c0db274ce219882f07a8bd9c7f07e` |
| global_step | 207 |
| init / reference | `outputs/phase5c_qse_relativeguard/checkpoints/step_final` (frozen) |

## Training recipe

| field | value |
|---|---|
| dataset | `reports/offline_dpo_unified_refinement_20261008/UNIFIED_PAIRS.jsonl` |
| dataset sha256 | `ae27e2ee32a0314378d1b4c334e07a91afb089aec15cc7d771b1a188dd59c94e` |
| pairs / images | 6633 / 6633 (one pair per image; chosen unchanged) |
| matched rejected | 2733 |
| original-rejected fallback | 3900 |
| beta / lr | 0.1 / 5e-6 |
| scheduler | constant |
| effective batch size | 8 |
| optimizer | AdamW (0.9,0.999), wd 0, grad-clip 1.0 |
| LoRA | r16, alpha32, dropout0.05 (q/k/v/o/gate/up/down) |
| precision / attn | bf16 / sdpa |
| seed | 42 |
| optimizer steps | 207 (of the 830-step schedule; step_00207 = 25%) |
| training API | 0 |

## Generation config (evaluation)

greedy (`do_sample=False`), `max_new_tokens` 32, seed 42, batch 16,
instruction `Generate a question about this image.`, one question per image.

## Evaluators (frozen)

| eval | model | version / prompt |
|---|---|---|
| SeeQ Coverage/Search/Context/Spatial | `gpt-5.6-luna` | `seeq-single-question-v1`, status `VARIANT_DOCUMENTED`, prompt_hash `57a357e077e599248bc2212ca1e3d83b42c4ae5ab6448d6fbd26e824773fe133` |
| GPT support | `gpt-4o` | `visual-support-phase23-v1.0`, rubric `configs/new_vqg/visual_support_prompts.json` |
| statistics | paired image-level bootstrap | n=10000, seed 42, unit=image_id |

## Dev sets

| set | manifest | sha256 |
|---|---|---|
| DEV256 (selection) | `reports/offline_dpo_checkpoint_trajectory_20261008/DEV256_MANIFEST.jsonl` | `afc3825c1f19d9acda459b27a3ef6534a409a6e87aff9d58c727dc378957a590` |
| DEV744 (secondary confirmation) | `reports/offline_dpo_freeze_20261008/DEV744_MANIFEST.jsonl` | `a6db294a3d9dc4887de8eead7fb911441dd2aa783d5764ced36a7275112ff1ae` |

## Selection

`step_00207` was selected on **DEV256** (see `PHASE1/PHASE1_REPORT.md` and
`PHASE1/STAGE1EXT.json`). It was the only checkpoint in the unified trajectory that
simultaneously reduced type JS, improved K=4 diversity, avoided a BLEU/CIDEr
regression, and retained preference learning.

## Statements

1. **DEV256 is a model-selection development subset.**
2. **No further checkpoint or DPO hyperparameter selection is permitted after this freeze.**
3. **DEV744_SECONDARY_CONFIRMATION** = formal dev1000 minus DEV256. It is **not** a new
   untouched test set (the broader formal dev1000 participated in prior development),
   but `step_00207` was not selected on it.
4. **`final_holdout` remains sealed** and may only be evaluated after the Full Ours
   configuration, DPO configuration, baseline list, metric list, evaluator prompts and
   statistical protocol are all frozen.
5. git HEAD at freeze: `9dea27f09aa5c21f4fa53f730923ac1df64169eb`.
