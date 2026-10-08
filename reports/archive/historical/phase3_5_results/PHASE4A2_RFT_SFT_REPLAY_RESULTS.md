# Phase 4A.2 — UVFC+Violation-RFT with 1:1 Original-Human SFT Replay

One pre-specified seed-42 pilot testing whether original-SFT replay preserves the
visual-exploration gains of UVFC+Violation-RFT while recovering the capabilities
that degraded consistently in Phase 4A.1 (BLEU-4, Spatial, within-image semantic
diversity). Only the dataset composition changed: the corrected pure-RFT run
(`outputs/phase4a_uvfc_rft/uvfc_violation`, seed 42, 250 optimizer steps) is reused
as the reference, and the replay run matches its optimizer-step budget exactly.
No GRPO, no UVFC/Flat change, no tuning, `final_holdout` untouched.

> **Post-audit correction (Phase 4A.4):** Replay42 SeeQ had 999/1000 valid
> (1 transient HTTP 502). After retrying that single ID, the seed42 replay values
> are Search 0.3214, Coverage 0.3234, Context 0.2562, Spatial 0.0684
> (Supported/Unsupported/BLEU/CIDEr/Dup/SemDiv unchanged); Replay−SFT CIs shift by
> ≤ 0.0004. Decision unchanged (**REPLAY_PARETO_IMPROVES**). See
> `docs/PHASE4A4_FULL_INTEGRITY_AUDIT.md`.

## A. Data construction

- Input UVFC+Violation selection reused unchanged:
  `outputs/phase4a_uvfc_rft/data/uvfc_violation_train.jsonl`
  (2,000 image-question pairs, 2,000 unique train2014 images, SHA256
  `dd559d3d686e1265c7848af3a4b5c102457b6588dafb32858ff7be82168e7be9`).
- Replay source: the original 50k VQAv2 SFT training corpus used to train Final
  SFT, `src/vvqg/data/sft/train.jsonl`
  (50,000 records / 9,184 images, SHA256
  `17b7892dc74fec76dccdc0a278db986d5ceb91ae7c5ce671247f247df1ecf1f1`).
- One deterministic original-human question was attached to each of the same
  2,000 image IDs. Candidate questions were the image's original human questions;
  candidates with normalized text identical to the selected UVFC question were
  deprioritized, then the first by `SHA256(image_id + "\t" + normalized_question)`
  was taken. No random sampling, no reranking, no regeneration.
- All 2,000 replay images exist in the original corpus (min 3, max 82 human
  questions each). `replay_same_as_uvfc_count = 0`.
- Outputs (`outputs/phase4a2_rft_replay/data/`):
  - `human_replay_2000.jsonl` — 2,000 records, SHA256 `e632cdc9…`
  - `mixed_uvfc_violation_replay_1to1.jsonl` — 4,000 records (2,000 UVFC+Violation
    + 2,000 human replay), SHA256 `df537d10…`
  - `replay_audit.json`

Integrity (from `replay_audit.json`): UVFC 2,000 / replay 2,000 / mixed 4,000;
unique images 2,000 = 2,000, image-id sets identical; zero overlap with
`development1000`, reward calibration, and `final_holdout`. The `source` metadata
token is stored but never fed to the model (the trainer uses only `image` and
`question`).

## B. Fairness control

- Initialization: `PeftModel.from_pretrained(base, ...checkpoint-final,
  is_trainable=True)` from Final SFT
  `outputs/checkpoints/qwen3vl8b_lora_r16_vqav2_50k/checkpoint-final`. Asserted in
  `train/manifest.json`: `init_adapter == final_sft_checkpoint`,
  `base_params_trainable = 0`, `lora_trainable = true`, trainable parameters
  `43,646,976` (252 modules), `smoke_loss` finite, `image_affects_loss = true`
  (blanking `pixel_values` changes the loss by 0.879). Assertions recorded in
  `train/smoke_report.json` and `train/manifest.json`.
- Optimizer-step budget (`train/manifest.json`):
  `reference_pure_rft_steps = 250`, `replay_run_max_steps = 250`,
  `budget_match = true`. Batch 2 × accum 8 (effective 16) over 4,000 mixed records
  ⇒ exactly one pass / 250 optimizer steps, identical record exposure to the pure
  run (2,000 × 2 epochs).
- Everything else matches the corrected pure-RFT run: LoRA r16/a32/d0.05, AdamW
  (HF default `adamw_torch_fused`), cosine schedule, 3% warmup (7 steps), lr 2e-4,
  max-grad-norm 1.0, bf16, FlashAttention-2, gradient checkpointing, seed 42.
  Per-step log: `train/train_log.jsonl` (250 records; loss finite, grad_norm
  finite). Training loss moved from ~0.91 (first 10 steps) to ~0.88 (last 10) — a
  much smaller drop than pure RFT (0.68→0.34), as expected when half the updates
  are placed on the diverse human distribution.

## C. Main results (development1000, greedy)

| Metric | SFT | Pure RFT seed42 | RFT + 1:1 Replay | Replay vs SFT | Replay vs Pure RFT |
|---|---:|---:|---:|---:|---:|
| BLEU-4 | 0.3172 | 0.2781 | 0.3164 | −0.0008 | +0.0383 |
| CIDEr | 0.4550 | 0.4729 | 0.4847 | +0.0297 | +0.0118 |
| Q-BLEU4-v | 0.3107 | 0.2705 | 0.3103 | −0.0005 | +0.0398 |
| Search | 0.2746 | 0.3620 | 0.3213 | +0.0466 | −0.0408 |
| Coverage | 0.2598 | 0.3804 | 0.3235 | +0.0637 | −0.0571 |
| Context | 0.2704 | 0.2648 | 0.2563 | −0.0142 | −0.0086 |
| Spatial | 0.0732 | 0.0564 | 0.0685 | −0.0044 | +0.0122 |
| Supported | 0.725 | 0.821 | 0.786 | +0.0610 | −0.0350 |
| Unsupported | 0.159 | 0.056 | 0.095 | −0.0640 | +0.0390 |
| Dup | 0.570 | 0.397 | 0.487 | −0.083 | +0.090 |
| SemDiv | 0.5360 | 0.4427 | 0.5085 | −0.0275 | +0.0659 |

Comparison uses only the corrected pure-RFT seed42 artifact (the pre-fix Phase 4A
draft is ignored). BLEU/CIDEr/Q-BLEU/Dup/SemDiv columns are point estimates;
Search/Coverage/Context/Spatial/Supported/Unsupported carry the paired bootstrap
below.

Visual support failure categories (rate over valid labels):

| Failure type | SFT | Pure RFT | Replay |
|---|---:|---:|---:|
| missing_entity | 0.003 | 0.000 | 0.004 |
| wrong_attribute | 0.001 | 0.002 | 0.003 |
| wrong_relation | 0.000 | 0.000 | 0.000 |
| ambiguous_reference | 0.064 | 0.007 | 0.021 |
| unobservable_target | 0.122 | 0.101 | 0.099 |
| external_knowledge_required | 0.004 | 0.012 | 0.003 |

Generation quality: 1,000 greedy / 4,000 sampled; 0 empty, 0 malformed,
question-mark rate 1.0.

Rollout unique rate (sampled 4× within-image, from the diversity summaries):
SFT 0.9555, pure RFT 0.804, Replay 0.9248 — replay restores most of the sampled
lexical uniqueness that pure RFT lost.

## D. Paired bootstrap (5,000 resamples, seed 42)

Replay − SFT:

| Metric | Δ | 95% CI |
|---|---:|---:|
| Search | +0.0466 | [+0.0356, +0.0581] |
| Coverage | +0.0637 | [+0.0507, +0.0771] |
| Context | −0.0142 | [−0.0214, −0.0070] |
| Spatial | −0.0044 | [−0.0136, +0.0048] |
| Supported | +0.0610 | [+0.0310, +0.0900] |
| Unsupported | −0.0640 | [−0.0870, −0.0400] |

Replay − Pure RFT:

| Metric | Δ | 95% CI |
|---|---:|---:|
| Search | −0.0408 | [−0.0527, −0.0292] |
| Coverage | −0.0571 | [−0.0709, −0.0430] |
| Context | −0.0086 | [−0.0160, −0.0012] |
| Spatial | +0.0122 | [+0.0030, +0.0214] |
| Supported | −0.0350 | [−0.0640, −0.0070] |
| Unsupported | +0.0390 | [+0.0200, +0.0590] |

Replay remains significantly better than SFT on Search, Coverage, Supported and
significantly lowers Unsupported. Relative to pure RFT it gives back a
significant part of Search/Coverage/Supported and raises Unsupported, while
significantly improving Spatial.

## E. Search / Coverage retained-gain

| Metric | Pure RFT gain vs SFT | Replay gain vs SFT | Retained |
|---|---:|---:|---:|
| Search | +0.0874 | +0.0467 | **0.535** |
| Coverage | +0.1206 | +0.0637 | **0.528** |
| Supported | +0.0960 | +0.0610 | **0.635** |
| Unsupported (reduction) | −0.1030 | −0.0640 | **0.621** |

Replay retains roughly half to two-thirds of the exploration/support gains of
the pure-RFT run.

## F. BLEU / Spatial / SemDiv recovery

`recovery_fraction = (M_replay − M_pure) / (M_sft − M_pure)`:

| Metric | SFT | Pure RFT | Replay | Recovery |
|---|---:|---:|---:|---:|
| BLEU-4 | 0.3172 | 0.2781 | 0.3164 | **0.980** |
| Spatial | 0.0732 | 0.0564 | 0.0685 | **0.718** |
| SemDiv | 0.5360 | 0.4427 | 0.5085 | **0.706** |

BLEU-4 is essentially fully restored (0.9975× SFT); Spatial and within-image
semantic diversity recover ~71% of the distance back to SFT. CIDEr improves over
both SFT and pure RFT (1.065× / 1.025×).

## G. Negative results / tradeoffs

- **Exploration gains are not free**: Search/Coverage/Supported are significantly
  below pure RFT, and Unsupported is significantly higher (0.095 vs 0.056) —
  replay does give back part of the pure-RFT support benefit.
- **Context declines** relative to SFT (−0.0142, CI excluding 0) and is slightly
  below pure RFT; this is similar in direction to the non-robust Context movement
  seen across seeds in Phase 4A.1.
- **Duplication rises** versus pure RFT (0.487 vs 0.397; distinct-2 0.167 vs
  0.218), though still well below SFT (0.570). The replay mixture is less
  lexically aggressive than pure RFT.
- **Uncertain support** rises to 0.119 (SFT 0.116, pure RFT 0.123 regime); the
  residual unsupported mass concentrates on `unobservable_target` (0.099).
- Training loss plateaus near 0.88 instead of the pure run's 0.34 — expected, and
  the mechanism behind the recovered language/diversity behaviour.

## H. Decision

`outputs/phase4a2_rft_replay/comparison/decision.json`:

```json
{"decision": "REPLAY_PARETO_IMPROVES",
 "search_significant_above_sft": true,
 "coverage_significant_above_sft": true,
 "unsupported_significant_below_sft": true,
 "tradeoff_metrics_recovered_vs_pure_rft": ["BLEU", "Spatial", "SemDiv"],
 "no_severe_visual_gain_loss": true,
 "retained_search_gain": 0.5346,
 "retained_coverage_gain": 0.5284,
 "final_holdout_touched": false}
```

**REPLAY_PARETO_IMPROVES.** With 1:1 original-human replay and an identical
optimizer-step budget, all four PARETO conditions hold: Search and Coverage stay
significantly above SFT, Unsupported stays significantly below SFT, and all three
known tradeoff metrics (BLEU-4, Spatial, SemDiv) improve relative to pure RFT
without severe loss of the visual gains (Search/Coverage retain ~53% of the pure
gain). This is the desired outcome of the phase.

One pilot only (`replay ratio = 1:1`, seed 42); no follow-up ratios, LR, epoch, or
reward variants were attempted. `final_holdout` was never generated, scored,
inspected, or used for selection; `uvfc_modified = false`, `flat_modified = false`,
`grpo_started = false`.
