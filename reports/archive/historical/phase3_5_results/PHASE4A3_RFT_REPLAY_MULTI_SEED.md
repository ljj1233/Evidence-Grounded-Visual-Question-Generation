# Phase 4A.3 — UVFC+Violation-RFT + 1:1 SFT Replay Multi-Seed Confirmation

Confirmation-only study: does the Phase 4A.2 seed42 replay Pareto pattern
reproduce for seeds 43 and 44? Method, replay ratio (1:1), dataset, and
optimizer-step budget are frozen; only the training seed changes. No tuning, no
GRPO, no UVFC/Flat change, `final_holdout` untouched.

> **Post-audit correction (Phase 4A.4):** Replay42 SeeQ was 999/1000 and
> PureRFT44 support was 999/1000 (transient 502 / ReadTimeout); both failed IDs
> were retried to 1000/1000 and the summaries regenerated. Corrected Replay42
> Search/Coverage/Context/Spatial = 0.3214 / 0.3234 / 0.2562 / 0.0684; mean
> Replay Δ vs SFT Search +0.0382, Coverage +0.0586, Spatial −0.0029; PureRFT44
> Supported 0.792. Supported+Unsupported+Uncertain = 1.0 for every system.
> Decision unchanged (**REPLAY_ROBUST_WITH_RESIDUAL_TRADEOFF**). See
> `docs/PHASE4A4_FULL_INTEGRITY_AUDIT.md`.

## 1. Experimental control

Three replay runs, each initialized independently from
`outputs/checkpoints/qwen3vl8b_lora_r16_vqav2_50k/checkpoint-final`:

| Run | Start | Dataset | max_steps | Seed |
|---|---|---:|---:|---:|
| Replay42 (Phase 4A.2, reused) | Final SFT | mixed 1:1 | 250 | 42 |
| Replay43 (new) | Final SFT | mixed 1:1 | 250 | 43 |
| Replay44 (new) | Final SFT | mixed 1:1 | 250 | 44 |

Frozen dataset: `outputs/phase4a2_rft_replay/data/mixed_uvfc_violation_replay_1to1.jsonl`
(4,000 records = 2,000 UVFC+Violation + 2,000 original-human replay; 2,000 unique
images; UVFC/replay image-id sets identical), SHA256
`df537d10b0127f35f1454ee2d4f5292a464b6ad25cb6c23199cc47cb4b8018ce` — identical
for all three seeds. No replay questions were rebuilt or resampled, no UVFC
candidate regenerated or reranked.

## 2. Initialization verification

Each of seed43/seed44 (`seed{43,44}/manifest.json`, `train/smoke_report.json`)
asserts and records:

- `init_adapter == outputs/checkpoints/qwen3vl8b_lora_r16_vqav2_50k/checkpoint-final`
  via `PeftModel.from_pretrained(..., is_trainable=True)` (no `get_peft_model`).
- `base_params_trainable = 0`, `base_frozen = true`.
- `lora_trainable = true`, `trainable_parameters = 43,646,976` (252 modules).
- `smoke_loss` finite, `image_affects_loss = true`, backward `gradients_finite = true`.
- `final_holdout_touched = false`.

No run initialized from seed42 replay, pure RFT, P1strong, GRPO, or any
intermediate checkpoint.

## 3. Dataset / hash verification

- Mixed records 4,000; UVFC 2,000; replay 2,000; unique images 2,000.
- Dataset SHA256 identical across seeds: `df537d10…` (== seed42).
- Leakage audit (Phase 4A.2 `replay_audit.json`): overlap with `development1000`
  = 0, reward calibration = 0, `final_holdout` = 0.

## 4. Training-budget equality

Read from `outputs/phase4a2_rft_replay/train/manifest.json`
(`replay_run_max_steps = 250`); not hard-coded.

| Run | seed42 max_steps | run max_steps | equality |
|---|---:|---:|---|
| Replay43 | 250 | 250 | true |
| Replay44 | 250 | 250 | true |

Batch 2 × accum 8 (effective 16) over 4,000 records ⇒ one pass, 250 optimizer
steps. Per-step `train_log.jsonl` = 250 records for each seed.

## 5. Seed42/43/44 results (development1000, greedy)

| Metric | SFT | Replay42 | Replay43 | Replay44 | Mean Replay | Mean Δ vs SFT | Std Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| BLEU-4 | 0.3172 | 0.3164 | 0.2984 | 0.3088 | 0.3079 | −0.0093 | 0.0074 |
| CIDEr | 0.4550 | 0.4847 | 0.4744 | 0.4807 | 0.4799 | +0.0250 | 0.0042 |
| Q-BLEU4-v | 0.3107 | 0.3103 | 0.2918 | 0.3031 | 0.3017 | −0.0090 | 0.0076 |
| Search | 0.2746 | 0.3213 | 0.3114 | 0.3056 | 0.3128 | +0.0382 | 0.0065 |
| Coverage | 0.2598 | 0.3235 | 0.3192 | 0.3126 | 0.3184 | +0.0586 | 0.0045 |
| Context | 0.2704 | 0.2563 | 0.2776 | 0.2698 | 0.2679 | −0.0025 | 0.0088 |
| Spatial | 0.0732 | 0.0685 | 0.0722 | 0.0704 | 0.0704 | −0.0028 | 0.0015 |
| Supported | 0.725 | 0.786 | 0.755 | 0.747 | 0.7627 | +0.0377 | 0.0168 |
| Unsupported | 0.159 | 0.095 | 0.110 | 0.119 | 0.1080 | −0.0510 | 0.0099 |
| Dup | 0.570 | 0.487 | 0.455 | 0.489 | 0.4770 | −0.0930 | 0.0156 |
| SemDiv | 0.5360 | 0.5085 | 0.5136 | 0.5057 | 0.5093 | −0.0268 | 0.0032 |

Greedy generation quality: 0 empty, 0 malformed, question-mark rate 1.0 for both
new seeds. Rollout unique rate (sampled 4×): SFT 0.9555, Replay42/43/44
0.9247 / 0.9263 / 0.9253 (pure RFT 0.804 / 0.809 / 0.813).

Visual-support failure types (rate): seed43 → missing_entity 0.004,
wrong_attribute 0.000, wrong_relation 0.000, ambiguous_reference 0.024,
unobservable_target 0.129, external_knowledge_required 0.013; seed44 → 0.003 /
0.002 / 0.000 / 0.037 / 0.118 / 0.004.

## 6. Paired bootstrap vs SFT (5,000 resamples, seed 42)

| Metric | Replay42 | Replay43 | Replay44 |
|---|---:|---:|---:|
| Search | +0.0466 [0.0356, 0.0581] | +0.0368 [0.0264, 0.0472] | +0.0310 [0.0206, 0.0414] |
| Coverage | +0.0637 [0.0507, 0.0771] | +0.0594 [0.0464, 0.0726] | +0.0528 [0.0406, 0.0656] |
| Context | −0.0142 [−0.0214, −0.0070] | +0.0072 [−0.0004, +0.0148] | −0.0006 [−0.0076, +0.0064] |
| Spatial | −0.0044 [−0.0136, +0.0048] | −0.0010 [−0.0100, +0.0082] | −0.0028 [−0.0116, +0.0062] |
| Supported | +0.0610 [0.0310, 0.0900] | +0.0300 [0.0000, 0.0600] | +0.0220 [−0.0070, 0.0500] |
| Unsupported | −0.0640 [−0.0870, −0.0400] | −0.0490 [−0.0730, −0.0240] | −0.0400 [−0.0630, −0.0160] |

Search, Coverage, Unsupported: significant in all 3 seeds. Supported: significant
for seeds 42/43, directionally positive but CI includes 0 for seed 44. Context:
seed-dependent.

## 7. Comparison against corrected Pure RFT (Replay − PureRFT)

| Metric | Replay42 | Replay43 | Replay44 |
|---|---:|---:|---:|
| Search | −0.0408 [−0.0527, −0.0292] | −0.0466 [−0.0584, −0.0350] | −0.0378 [−0.0492, −0.0262] |
| Coverage | −0.0571 [−0.0709, −0.0430] | −0.0462 [−0.0602, −0.0324] | −0.0480 [−0.0624, −0.0338] |
| Context | −0.0086 [−0.0160, −0.0012] | +0.0144 [+0.0070, +0.0218] | −0.0098 [−0.0180, −0.0016] |
| Spatial | +0.0122 [+0.0030, +0.0214] | +0.0112 [+0.0020, +0.0208] | +0.0122 [+0.0022, +0.0220] |
| Supported | −0.0350 [−0.0640, −0.0070] | −0.0620 [−0.0910, −0.0320] | −0.0440 [−0.0761, −0.0120] |
| Unsupported | +0.0390 [+0.0200, +0.0590] | +0.0450 [+0.0230, +0.0670] | +0.0440 [+0.0210, +0.0671] |

Replay consistently (3/3) gives back significant Search/Coverage/Supported and
raises Unsupported relative to pure RFT, and consistently significantly improves
Spatial.

### Pure RFT vs Replay (three-seed means)

| Metric | Pure RFT Mean | Replay Mean | SFT | Interpretation |
|---|---:|---:|---:|---|
| BLEU-4 | 0.2709 | 0.3079 | 0.3172 | Replay recovers toward SFT |
| CIDEr | 0.4497 | 0.4799 | 0.4550 | Replay preserved/improved (above SFT) |
| Search | 0.3545 | 0.3128 | 0.2746 | Replay gives up part of pure gain; retains positive gain over SFT |
| Coverage | 0.3688 | 0.3184 | 0.2598 | Replay gives up part of pure gain; retains positive gain over SFT |
| Spatial | 0.0585 | 0.0704 | 0.0732 | Replay recovers Spatial vs pure RFT; remains below SFT |
| Supported | 0.8099 | 0.7627 | 0.725 | Replay retains positive gain over SFT |
| Unsupported | 0.0650 | 0.1080 | 0.159 | Replay retains reduction vs SFT |
| SemDiv | 0.4486 | 0.5093 | 0.5360 | Replay recovers SemDiv vs pure RFT; remains below SFT |

## 8. Retained visual-gain analysis

`retained = (Replay − SFT) / (PureRFT − SFT)`:

| Metric | seed42 | seed43 | seed44 | Mean ± Std |
|---|---:|---:|---:|---:|
| Search | 0.535 | 0.441 | 0.451 | 0.475 ± 0.042 |
| Coverage | 0.528 | 0.562 | 0.524 | 0.538 ± 0.017 |
| Supported | 0.635 | 0.326 | 0.329 | 0.430 ± 0.145 |
| Unsupported (reduction) | 0.621 | 0.521 | 0.471 | 0.538 ± 0.062 |

Replay consistently preserves ~44–56% of the pure-RFT Search/Coverage gain and
~47–62% of the Unsupported reduction. Supported retention is more variable
(0.33–0.64).

## 9. BLEU / Spatial / SemDiv recovery analysis

`recovery = (Replay − PureRFT) / (SFT − PureRFT)`:

| Metric | seed42 | seed43 | seed44 | Mean ± Std |
|---|---:|---:|---:|---:|
| BLEU-4 | 0.980 | 0.621 | 0.833 | 0.811 ± 0.147 |
| Spatial | 0.718 | 0.918 | 0.813 | 0.817 ± 0.082 |
| SemDiv | 0.706 | 0.755 | 0.607 | 0.689 ± 0.061 |

All three metrics recover in all 3 seeds (recovery > 0). BLEU varies most
(seed43 only 62%); Spatial is the most consistent and complete recovery
(~0.82 mean); SemDiv recovers ~0.69 on average.

## 10. Robust gains

- **Search > SFT in 3/3 seeds** (+0.031 to +0.047, CIs exclude zero).
- **Coverage > SFT in 3/3 seeds** (+0.053 to +0.064, CIs exclude zero).
- **Supported > SFT in 3/3 seeds** (direction), significant 2/3 (seed44 CI
  includes zero).
- **Unsupported < SFT in 3/3 seeds** (−0.040 to −0.064, CIs exclude zero).
- **CIDEr improved in 3/3 seeds** (above both SFT and pure RFT).
- **Rollout uniqueness restored** (0.925 vs pure RFT 0.80, SFT 0.956).
- **Spatial recovered vs pure RFT in 3/3 seeds** (CIs exclude zero).

## 11. Residual tradeoffs (not hidden)

- **BLEU-4 is not fully restored every seed**: mean −0.0093 vs SFT; seed43 is
  0.2984 (0.941× SFT). Recovery is 0.62–0.98, not uniformly 1.0.
- **Spatial remains below SFT in 3/3 seeds** (mean 0.0704 vs 0.0732) even though
  it recovers strongly vs pure RFT.
- **SemDiv remains below SFT in 3/3 seeds** (mean 0.5093 vs 0.5360).
- **Supported retention is weak for seeds 43/44** (0.33) and seed44's
  Replay−SFT supported CI includes zero.
- **Search/Coverage/Supported are significantly below pure RFT** and Unsupported
  is significantly above pure RFT — replay does trade away part of the pure-RFT
  support/exploration gain.
- **Context is not robustly changed** (seed-dependent, small).

## 12. Robustness questions

- **Q1 Search > SFT all seeds?** Yes, 3/3 (CIs exclude zero).
- **Q2 Coverage > SFT all seeds?** Yes, 3/3 (CIs exclude zero).
- **Q3 Supported > SFT all seeds?** Yes directionally 3/3; significantly 2/3
  (seed44 CI [−0.007, 0.050]).
- **Q4 Unsupported < SFT all seeds?** Yes, 3/3 (CIs exclude zero).
- **Q5 BLEU close to SFT all seeds?** Approximately: 0.997× / 0.941× / 0.973× SFT;
  mean −0.0093, not fully restored.
- **Q6 CIDEr preserved/improved?** Yes, 3/3 above SFT and above pure RFT.
- **Q7 Spatial consistently recovered vs pure RFT?** Yes, 3/3, significant.
- **Q8 SemDiv consistently recovered vs pure RFT?** Yes, 3/3 by point estimate
  (+0.066 / +0.069 / +0.047).
- **Q9 Spatial still below SFT after replay?** Yes, 3/3 (mean 0.0704 vs 0.0732).
- **Q10 SemDiv still below SFT after replay?** Yes, 3/3 (mean 0.5093 vs 0.5360).
- **Q11 Meaningful fraction of pure Search/Coverage gain retained?** Search
  0.535 / 0.441 / 0.451; Coverage 0.528 / 0.562 / 0.524 — all ≥ 0.44, no threshold
  was tuned.

## Decision

`outputs/phase4a3_rft_replay_multiseed/comparison/decision.json`:

```json
{"decision": "REPLAY_ROBUST_WITH_RESIDUAL_TRADEOFF",
 "search_gt_sft_seeds": 3, "coverage_gt_sft_seeds": 3,
 "supported_gt_sft_seeds": 3, "unsupported_lt_sft_seeds": 3,
 "tradeoff_recovered_seed_counts": {"BLEU": 3, "Spatial": 3, "SemDiv": 3},
 "severe_visual_degeneration": false,
 "spatial_mean_below_sft": true, "semdiv_mean_below_sft": true,
 "final_holdout_touched": false}
```

**REPLAY_ROBUST_WITH_RESIDUAL_TRADEOFF.** The replay method's visual/support
gains and its capability recovery relative to pure RFT are robust across all
three seeds, but Spatial and SemDiv remain clearly below SFT. The seed42
(seed42) Pareto decision reproduces qualitatively but not as a strict all-metric
Pareto improvement. No follow-up tuning was performed; `final_holdout` was never
generated, scored, inspected, or used for selection; `uvfc_modified = false`,
`flat_modified = false`, `grpo_started = false`.
