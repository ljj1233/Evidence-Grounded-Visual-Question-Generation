# Phase 4B.2 — Base-initialization Static UVFC GRPO (results)

Strict initialization ablation of Phase 4B Static UVFC Online GRPO: the only
changed variable is the initialization checkpoint. The run starts from the raw
`Qwen3-VL-8B-Instruct` base model (fresh LoRA) instead of the Final SFT adapter.
Data, reward, rollout, objective, optimizer, budget, guards and evaluation are
identical to the Static UVFC run.

## 1. Pre-run dependency audit

`outputs/phase4b_online_uvfc/base_uvfc/initialization_audit.json` (all checks passed).

- **Training manifest** `outputs/phase4b_online_uvfc/data/train_10000.jsonl`
  SHA256 `c884ed36…a67adc2`, 10,000 records, 9,184 unique images; overlap with
  development1000 / reward_calibration / final_holdout = 0. Fields:
  `{image_id, image, question}` → `training_manifest_type =
  SUPERVISED_FIELDS_PRESENT`. The `question` field is **not consumed**: the
  trainer builds rollouts only from `row["image"]`
  (`online_uvfc_grpo.py:309`); every later `question` access is a freshly
  generated rollout question. So `question_in_rollout_generation`,
  `question_in_reward_calculation`, `question_in_grpo_loss` are all `false`.
- **UVFC anchor**: `p_anchor(j)` = SAE feature frequency over discovery
  activations extracted with the **base model + Final SFT adapter** on human
  train questions; `p_policy` from the SFT greedy policy; `v_j`/`u_j` from SFT
  train only. `uvfc_anchor_source = FINAL_SFT_ANCHOR`,
  `sft_anchor_dependency = true`. All reused frozen and unchanged.
- **Reward identity**: base config differs from `static_uvfc.yaml` only in
  `{run_name, progress_label, sft_checkpoint, output_dir, greedy_out, sampled_out}`;
  `reward_config_match = true`. Reward file hashes (sae.pt, weights.pt,
  uvfc.py, online_flat.py, frozen_verifier.py, fixed_group_relative.py) are the
  frozen ones. The only code change vs Static is progress logging (tqdm +
  `elapsed_seconds`), which does not affect rollout/reward/objective.

## 2. Initialization

- Base model `/hpc2hdd/home/fye374/models/Qwen3-VL-8B-Instruct`,
  `base_model_hash = 70bd1497…` (config + weight index).
- `adapter = null`; a fresh LoRA-on-base adapter
  (`outputs/phase4b_online_uvfc/base_uvfc/init_adapter/`, r=16, alpha=32,
  dropout=0.05, same target modules, peft default init B=0) makes the policy
  mathematically equal to base at step 0.
- `trainable_parameter_count = 43,646,976` (matches Static); total 8,810,770,672.
- `policy_init_hash = reference_model_hash = base_model_hash`; reference is the
  frozen Base Qwen3-VL, **not** Final SFT.

## 3. Training outcome

The run self-terminated at **step 91/1000** under the unchanged Static guard:

| guard | value | threshold |
|---|---:|---:|
| mean question tokens (step 91) | 19.00 | 1.5 × 12.34 = 18.52 |
| reference KL (max) | 0.0123 | 0.2 |
| grad norm (max) | 1.36 | — |
| malformed rate (max) | 0.0 | 0.02 |
| duplicate rate (max) | 0.469 | 0.313 + 0.2 = 0.513 |

`stop_reason = DEGENERATION` (question length inflation: the base policy grows
toward the 32-token generation cap while the guard is calibrated on the
step-1 mean). Wall time 5,939 s, ~58.6 s/step, peak GPU 62.9 GB allocated /
70.9 GB reserved. Checkpoints: `step_0000`, `step_final` (= step 91).

Because the restrictions forbid changing stopping criteria, the guard was not
overridden; the 1000-step completion criterion is therefore **not met**, and
that instability is the central result.

## 4. Evaluation (development1000, greedy)

Same pipeline as Static: SeeQ (`gpt-5.6-luna` variant), GPT-4o support,
BLEU/CIDEr/Q-BLEU, Distinct-2/dup, semantic diversity. SeeQ and support have
full coverage (1000/1000 valid).

| Model | Init | Search | Coverage | Context | Spatial | Supported | Unsupported | BLEU | CIDEr | SemDiv |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT | Final SFT | 0.275 | 0.260 | 0.270 | 0.073 | 0.725 | 0.159 | 0.317 | 0.455 | 0.536 |
| Static UVFC GRPO | Final SFT | 0.332 | 0.332 | 0.255 | 0.071 | 0.814 | 0.094 | 0.326 | 0.519 | 0.537 |
| Base UVFC GRPO (step 91) | Base | 0.399 | 0.440 | 0.388 | 0.145 | 0.587 | 0.268 | 0.075 | 0.097 | 0.431 |

Base UVFC also has Distinct-2 0.296 and duplicate rate 0.065 on greedy, and
sampled4x Distinct-2 0.155 / dup 0.241.

## 5. Paired bootstrap (5000 resamples, seed 42, image-level)

| Base UVFC − | Search | Coverage | Context | Spatial | Supported | Unsupported |
|---|---:|---:|---:|---:|---:|---:|
| SFT | +0.125 [+0.112,+0.137] | +0.180 [+0.164,+0.196] | +0.118 [+0.107,+0.129] | +0.071 [+0.061,+0.082] | −0.138 [−0.174,−0.102] | +0.109 [+0.075,+0.142] |
| Static UVFC GRPO | +0.068 [+0.053,+0.082] | +0.107 [+0.090,+0.124] | +0.133 [+0.122,+0.144] | +0.074 [+0.062,+0.086] | −0.227 [−0.263,−0.191] | +0.174 [+0.143,+0.206] |

All CIs exclude 0.

## 6. Required interpretation

**Q1 — Can Static UVFC optimize from base initialization?**
No. It is not stable: the policy's question length inflates and trips the
frozen `DEGENERATION` guard at step 91/1000. No guard fired in the SFT-init
Static run.

**Q2 — Does base initialization improve Search/Coverage?**
Yes, on the checkpoint it reached: Search +0.125 / Coverage +0.180 vs SFT and
+0.068 / +0.107 vs Static UVFC, all CIs > 0 (Context and Spatial improve too).
This is partly because the base policy never acquired SFT's reference mode
collapse, so its questions are longer and cover more visual evidence.

**Q3 — Does base initialization preserve grounding?**
No. Supported drops −0.138 vs SFT and −0.227 vs Static, and Unsupported rises
+0.109 vs SFT and +0.174 vs Static (all CIs exclude 0). The visual-centric gain
comes at a clear cost in visual support.

**Q4 — Does SFT initialization provide better quality alignment?**
Yes on reference/grounding: Static (SFT init) keeps BLEU 0.326 vs 0.075 and
CIDEr 0.519 vs 0.097, and Supported 0.814 vs 0.587. Base init aligns with the
visual-centric rubric but not with reference-style questions or grounding.

**Q5 — Which initialization gives the best tradeoff?**
Reported only, no ranking beyond the metrics. Base init gives the best
Search/Coverage/Context/Spatial and diversity, but the worst grounding, the
worst reference overlap, and is unstable under the frozen guards; Static
(SFT init) gives the best grounding and reference overlap at lower visual
coverage; SFT is the reference point.

**UVFC reward dependency on SFT-derived anchor statistics.**
Explicitly: `sft_anchor_dependency = true`. `p_anchor`, `v_j`, `u_j`, the SAE
and `p_policy` all come from the SFT-adapted model / SFT train split and are
reused frozen. The base-init policy is therefore optimized against a reward
whose "anchor" is SFT-derived, which the ablation does not remove.

## 7. Decision

`outputs/phase4b_online_uvfc/base_uvfc/decision.json`: **TRAINING_UNSTABLE**
(`stop_reason = DEGENERATION`, steps = 91). This is an initialization effect,
not a reward redesign: the frozen reward/guard configuration is unchanged, and
base initialization is the only variable.

## 8. Limitations

- The Base row is a step-91 checkpoint (guard stop), not a 1000-step model;
  comparisons are against the SFT-init Static run's step-1000 checkpoint.
- SeeQ uses the documented `gpt-5.6-luna` substitution (not paper GPT-5.4 /
  five questions); support uses GPT-4o. Untrained-style baselines are not
  reference-matched, so BLEU/CIDEr and the visual-centric rubric measure
  different objectives.
- One seed (42); no multi-seed confirmation.
- `final_holdout` untouched.

---

# Addendum 4B.2b — base-anchored UVFC reward rebuild

## Motivation

The Phase 4B.2 ablation reused the SFT-anchored reward (SAE, `v_j`, `p_anchor`,
`p_policy`, `u_j` all built on `base + Final SFT adapter`). For a raw-base
policy this is off-distribution: `u_j = log((p_anchor+ε)/(p_policy+ε))` uses the
**SFT** policy's feature frequency as the reference. Rebuilt on the raw base
model, `u_j` flags **260** undercovered features vs **620** for the SFT anchor,
confirming the mismatch.

## Method

`scripts/vvqg/phase4b2_build_base_vfc.sh` rebuilds the reward in the base
model's activation space using the frozen Phase-2.6 protocol unchanged (layer
18, dictionary 4096, top-k 20, token cap 16, same data/seed/pooling; only the
activation model changes: no SFT adapter). Base SAE explained variance 0.906
(SFT 0.883). GRPO config `configs/phase4b/base_uvfc_baseanchor.yaml` is identical
to `base_uvfc.yaml` except `uvfc_dir = outputs/phase4b_online_uvfc/base_vfc`.

CPU fix applied (see `docs/vllm/SAME_BACKBONE_BASELINES_RUNTIME.md`): the
container has an 8-CPU quota but torch/OpenMP defaulted to 64 host threads and
was cgroup-throttled. Pinning `OMP_NUM_THREADS=8` raised base-space extraction
from 3.6 to 12.0 samples/s and training from 58.6 to **21.5 s/step** (GPU 0% →
98%), with throttling stopped.

## Training outcome

Self-terminated at **step 249/1000** under the unchanged Static guard, this time
on duplication (`duplicate_rate 0.53125 − initial 0.3125 > 0.2`), wall 6,084 s.
Reference KL max 0.0034, grad max 1.44, malformed 0. The base policy's inherent
duplication (0.31 at step 1) leaves little headroom under a guard calibrated on
SFT (initial dup ≈ 0.06).

## Four-way table (development1000, greedy)

| Model | Init | Anchor | steps | Search | Coverage | Context | Spatial | Supported | Unsupported | BLEU | CIDEr | SemDiv |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT | Final SFT | SFT | — | 0.275 | 0.260 | 0.270 | 0.073 | 0.725 | 0.159 | 0.317 | 0.455 | 0.536 |
| Static UVFC GRPO | Final SFT | SFT | 1000 | 0.332 | 0.332 | 0.255 | 0.071 | 0.814 | 0.094 | 0.326 | 0.519 | 0.537 |
| Base UVFC GRPO | Base | SFT | 91 | 0.399 | 0.440 | 0.388 | 0.145 | 0.587 | 0.268 | 0.075 | 0.097 | 0.431 |
| Base UVFC GRPO | Base | base | 249 | 0.382 | 0.406 | 0.367 | 0.130 | 0.622 | 0.240 | 0.088 | 0.128 | 0.412 |

Paired bootstrap (5000, seed 42, image-level; all CIs exclude 0):

| Base+base-anchor − | Search | Coverage | Context | Spatial | Supported | Unsupported |
|---|---:|---:|---:|---:|---:|---:|
| Base+SFT-anchor | −0.017 | −0.034 | −0.021 | −0.014 | **+0.035** | **−0.028** |
| SFT | +0.108 | +0.146 | +0.097 | +0.057 | −0.103 | +0.081 |
| Static UVFC GRPO | +0.051 | +0.073 | +0.112 | +0.059 | −0.192 | +0.146 |

## Interpretation

- The base-anchored reward is the methodologically correct construction and it
  changes behaviour: it runs ~2.7× longer before the guard (249 vs 91 steps),
  and **improves grounding** over the SFT-anchored base run (Supported +0.035,
  Unsupported −0.028) with slightly higher BLEU/CIDEr.
- It does **not** rescue base initialization. Both base variants self-stop under
  the frozen SFT-calibrated guards, and both remain far below the SFT-init
  Static run on grounding and reference overlap.
- The visual-centric advantage of base init is essentially unchanged by the
  anchor (a small −0.014…−0.034 shift): it comes from the base policy's
  long, diverse, non-mode-collapsed questions, not from the reward anchor.

## Decision

`outputs/phase4b_online_uvfc/base_uvfc_baseanchor/decision.json`:
**BOTH_BASE_RUNS_UNSTABLE**. Correcting the reward anchor improves grounding and
stability but does not make Static UVFC optimizable from raw base init under the
unchanged Phase-4B protocol.
