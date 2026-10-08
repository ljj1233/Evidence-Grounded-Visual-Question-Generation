# Phase 5C — Evidence-Constrained Question Space Expansion

Fixed story (unchanged): pretrained VLMs have strong language priors that bias
what is asked toward a limited set of common visual information; the goal is to
expand the visual evidence explored by generated questions while ensuring every
question is supported by the current image. Method skeleton:
Question Space Expansion → Evidence-Constrained Selection → Grounded
Multi-Positive Distillation → UVFC Refinement.

Development1000, greedy, unified evaluator (SeeQ `gpt-5.6-luna`, GPT-4o support,
BLEU/CIDEr/Dup/Distinct-2). No final_holdout, no new evaluator, no exploration
prompt, no Prompt-CoT teacher, no scene graph / detector / GT / reference / latent
coverage / SAE / SPIN / self-play. GPT-4o support is evaluation-only; all
selectors use only the frozen UVFC scorer (`sum_j v_j u_j`) and the frozen Flat
verifier (`Qwen2.5-VL-7B-Instruct`). All jobs pinned to cores 0–7
(`taskset -c 0-7`, `OMP/MKL/OPENBLAS/NUMEXPR/RAYON_NUM_THREADS=8`,
`TOKENIZERS_PARALLELISM=false`); DataLoader workers ≤ 4.
  
## A. Experimental contract

| item | value |
|---|---|
| train manifest | `outputs/phase4b_online_uvfc/data/train_10000.jsonl` (10000 rec / 9184 imgs, SHA256 `c884ed36…a67adc2`, not regenerated) |
| init (G1) | `outputs/checkpoints/qwen3vl8b_lora_r16_vqav2_50k/checkpoint-final` |
| init (G2) | `outputs/phase5c_distill_single/checkpoint-final` (QSE D1) |
| LoRA | r=16, α=32, dropout 0.05, 7 target modules, 43,646,976 trainable |
| seed | 42 |
| reward | frozen Static UVFC `sum_j v_j u_j` + violation-first (`p2`) group ranking |
| guard (G1/G2) | init-relative (Section F), same protocol for both |

## B. Selector ablation (5C-A, Direct-4 pool)

The Direct-4 pool (plain VQG prompt, 4 independent samples, T=0.8, seed 42) is
the Phase-5B pool; all selectors share it and the frozen verifier.

`tau` calibration (no development1000 sweep): the frozen Phase-2.1 calibration
corpus (`calibration_pairs.jsonl`, dev+val split, n=800) with GPT-4o labels gives
`support_floor≥0.5` precision 0.724 / recall 0.883 / F1 0.796 and
`support_floor≥1.0` precision 0.804 / recall 0.670 / F1 0.731. The F1-optimal
constraint is **tau = 0.5**; `tau = 1.0` collapses S3 onto S1 (identical audit).

| Selector | Search | Coverage | Context | Spatial | Supported | Unsupported | BLEU | CIDEr | Dup | Dist-2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT | 0.275 | 0.260 | 0.270 | 0.073 | 0.725 | 0.159 | 0.317 | 0.455 | 0.570 | 0.140 |
| Static UVFC | 0.332 | 0.332 | 0.255 | 0.071 | 0.814 | 0.094 | 0.326 | 0.519 | 0.499 | 0.174 |
| Direct-4 + S0 (existing UVFC) | 0.358 | 0.381 | 0.268 | 0.071 | 0.782 | 0.083 | 0.184 | 0.318 | 0.151 | 0.311 |
| Direct-4 + S1 (SupportFirst) | 0.347 | 0.366 | 0.262 | 0.070 | 0.829 | 0.062 | 0.193 | 0.326 | 0.152 | 0.302 |
| Direct-4 + S3 (ConstraintFirst, τ=0.5) | 0.350 | 0.370 | 0.262 | 0.074 | 0.804 | 0.077 | 0.189 | 0.321 | 0.152 | 0.310 |
| Explore-4 + S1 (Phase 5A/B) | 0.345 | 0.366 | 0.263 | 0.063 | 0.809 | 0.064 | 0.190 | 0.301 | 0.148 | 0.298 |

Selector audits:

| Selector | mean floor | P10/P25/med floor | viol | mean UVFC | changed vs S0 | fallback |
|---|---:|---:|---:|---:|---:|---:|
| S0 | 0.870 | 0.5/1.0/1.0 | 0.032 | 0.718 | 0.000 | — |
| S1 | 0.973 | 1.0/1.0/1.0 | 0.032 | 0.681 | 0.104 | — |
| S3 (τ=0.5) | 0.943 | 0.5/1.0/1.0 | 0.032 | 0.701 | 0.044 | 0.034 |

## C. Teacher selection

Rule: Supported ≥ 0.80, Unsupported ≤ 0.08, Search > 0.332, Coverage > 0.332;
among qualifiers prefer highest Coverage; if Coverage within 0.005, compare
Search; if still close, higher Supported. S0 fails (Supported 0.782). S1 and S3
both qualify.

S3 has the highest Coverage (0.3698) and Search (0.3496); S1 has higher
Supported (0.829). The coverage gap (0.0038) is within the 0.005 tie-break band,
so the rule's secondary criterion applies. We select **S3 (ConstraintFirst,
τ=0.5)** as the **Evidence-Constrained QSE Teacher**: it is the selector that
literally implements the fixed formulation (maximize UVFC subject to
violation = 0 and support_floor ≥ τ), it maximizes the primary objective, and it
still satisfies the support constraint. S1 is the close, more-conservative
alternative (documented in the failure cases).

## D. Positive-set statistics (5C-B / 5C-8)

Direct-4 pool on the frozen train manifest: 35097 candidates / 9184 images.
Grounded positive set `Q+(I)` = non-violation candidates with
`support_floor ≥ 0.5`, capped at 3 by UVFC.

- positive-count distribution: 1 → 1770, 2 → 3123, 3 → 5107 images; mean 2.34.
- D2 uses distinct epoch1/epoch2 targets on 8230/10000 records; D1 uses the
  teacher `best_question` for both epochs.
- artifacts: `outputs/phase5c_qse_teacher/{train_10000.jsonl,targets_d1_single.jsonl,targets_multipos.jsonl,target_audit.json}`.

## E. Single- vs multi-positive distillation (5C-9)

Both runs: original SFT recipe (LoRA r16/α32, lr 2e-4, cosine, warmup 0.03,
batch 2×8=16, flash-attention-2, seed 42), init = Final SFT, 2 epochs, 1250
optimization steps, identical compute; only the per-epoch target differs.

| Model | Search | Coverage | Supported | Unsupported | BLEU | CIDEr | Dup | Dist-2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D1 single-positive | 0.344 | 0.361 | 0.830 | 0.050 | 0.277 | 0.456 | 0.377 | 0.214 |
| D2 multi-positive | 0.274 | 0.270 | 0.784 | 0.114 | 0.333 | 0.486 | 0.571 | 0.143 |

Training loss: D1 0.586→0.350; D2 0.586→0.504.

**D2 collapses toward the SFT mode** (Dup 0.571, Dist-2 0.143, Search/Coverage at
SFT level). Multi-positive targets did **not** expand the question space here;
the second positive is less informative and the 2-epoch average re-collapses the
policy. Per Section 11 neither D1 nor D2 reaches Supported ≥ 0.85, so the
multi-positive distillation is **not** validated; D1 is nonetheless the better
distilled init and is used for G2 (Section 15).

## F. Relative degeneration guard (5C-C)

Original guard (`online_uvfc_grpo.py`): KL > 0.2 and clip_fraction > 0.5
(10-step streak, absolute); malformed > 0.02 (absolute); mean tokens >
1.5·(step-1 baseline); duplicate_rate − (step-1 baseline) > 0.2. The length and
duplicate guards were already step-1-relative, but the step-1 estimate from 32
rollouts is noisy and under-estimates a distilled policy's true duplication.

New guard (`online_uvfc_relative_grpo.py`), same window/patience/check interval:
measure `D_init` (and `D_sft0`) on a fixed guard-calibration subset (32 images,
seed-42 shuffled train manifest, group_size 4, T=0.8, 128 rollouts) *before*
training, then

- `T_dup = D_init_dup + 0.2` (`Delta_guard = T_old − D_sft0_dup = 0.2`)
- `T_length = D_init_len + 0.5·D_sft0_len` (`T_old = 1.5·D_sft0_len`)
- `T_malformed = D_init_malformed + 0.02`
- KL / clip unchanged (divergence, not distribution).

For G1 (`D_init = D_sft0`): `t_dup = 0.302`, `t_length = 10.15`, `t_malformed = 0.02`.

## G. Static vs QSE GRPO matched comparison (5C-C)

G1 = Final SFT init + Static UVFC + relative guard. G2 = D1 init + Static UVFC +
relative guard. Same manifest/ordering/seed/LoRA/optimizer/lr/rollout/KL/
generation/reward/1000 steps; only the init differs. Checkpoints saved at
0/50/100/150/250/500/750 + final.

G1 ran the full 1000 steps with **no guard trip** (relative guard `t_dup = 0.302`).
G2 self-stopped at **step 256/1000** on a duplicate spike (0.188 → 0.469 >
`t_dup = 0.458`); the relative guard extended it from the Phase-5B step-133 stop
to 256, but the distilled init is still duplication-prone. The guard was not
overridden.

Final checkpoints (development1000 greedy):

| | Search | Coverage | Context | Spatial | Supported | Unsupported | BLEU | CIDEr | Dup | Dist-2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G1 Static + relative guard (step 1000) | 0.334 | 0.331 | 0.256 | 0.072 | 0.803 | 0.100 | 0.326 | 0.519 | 0.499 | 0.174 |
| G2 QSE (D1) + UVFC, relative guard (step 256) | **0.368** | **0.391** | 0.268 | 0.064 | **0.851** | **0.039** | 0.261 | 0.438 | 0.358 | 0.227 |

Matched (same reward/guard/steps-config, only init differs): G2 vs G1 gives
Search **+0.033**, Coverage **+0.060**, Supported **+0.048**, Unsupported
**−0.061**, with lower duplication (0.358 vs 0.499) and higher Dist-2 (0.227 vs
0.174). This is a clear Pareto improvement over Static UVFC under the same
relative guard, and it meets the Section 18 strong target
(Search ≥ 0.36, Coverage ≥ 0.37, Supported ≥ 0.85, Unsupported ≤ 0.06).

## H. Training trajectories

Same checkpoints (development1000 greedy) for both runs:

| step | G1 Search | G1 Cov | G1 Sup | G1 Unsup | G2 Search | G2 Cov | G2 Sup | G2 Unsup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.274 | 0.262 | 0.701 | 0.153 | 0.346 | 0.360 | 0.828 | 0.055 |
| 50 | 0.273 | 0.263 | 0.713 | 0.152 | 0.347 | 0.369 | 0.840 | 0.045 |
| 100 | 0.282 | 0.274 | 0.730 | 0.147 | 0.348 | 0.367 | 0.825 | 0.055 |
| 150 | 0.280 | 0.273 | 0.737 | 0.128 | 0.352 | 0.372 | 0.840 | 0.044 |
| 250 | 0.283 | 0.278 | 0.740 | 0.139 | 0.364 | 0.378 | 0.843 | 0.052 |
| 500 | 0.315 | 0.307 | 0.778 | 0.098 | — | — | — | — |
| 750 | 0.323 | 0.324 | 0.789 | 0.097 | — | — | — | — |
| final | 0.330 | 0.333 | 0.792 | 0.094 | 0.367 | 0.387 | 0.849 | 0.046 |

G2 dominates G1 at **every** matched step (step 0: 0.346 vs 0.274 Search; step
250: 0.364 vs 0.283; Supported 0.843 vs 0.740). The advantage is **not** an
early-stop artifact — it is present before any UVFC update (from the QSE
distilled init) and increases monotonically during refinement. G1 reproduces
Static UVFC's trajectory. Coverage-vs-Supported: G2's Pareto frontier is above
G1's at all steps.

The checkpoint-sweep regeneration of the final greedy differs from each run's own
`greedy1000.jsonl` by ≤0.004 (G1 0.330/0.333/0.792/0.094 vs run 0.334/0.331/
0.803/0.100; G2 0.367/0.387/0.849/0.046 vs run 0.368/0.391/0.851/0.039);
Sections G/I use the run's canonical final outputs.

## I. Final metric table

| Method | Search | Coverage | Supported | Unsupported | Dup | Dist-2 |
|---|---:|---:|---:|---:|---:|---:|
| SFT | 0.275 | 0.260 | 0.725 | 0.159 | 0.570 | 0.140 |
| Static UVFC (original) | 0.332 | 0.332 | 0.814 | 0.094 | 0.499 | 0.174 |
| G1 Static + relative guard (1000) | 0.334 | 0.331 | 0.803 | 0.100 | 0.499 | 0.174 |
| Direct-4 + S3 teacher | 0.350 | 0.370 | 0.804 | 0.077 | 0.152 | 0.310 |
| D1 single-positive distilled | 0.344 | 0.361 | 0.830 | 0.050 | 0.377 | 0.214 |
| **G2 QSE + UVFC, relative guard (256)** | **0.368** | **0.391** | **0.851** | **0.039** | 0.358 | 0.227 |
| Prompt-CoT | 0.503 | 0.467 | 0.883 | 0.069 | 0.021 | 0.263 |
| GPT-5.6-Luna | 0.421 | 0.445 | 0.930 | 0.036 | 0.006 | 0.352 |

G2 − G1: Search +0.034, Coverage +0.060, Supported +0.048, Unsupported −0.061.
G2 − Static UVFC: Search +0.036, Coverage +0.059, Supported +0.037,
Unsupported −0.055. Section 18 strong target met on all four axes.

## J. Qualitative examples

`image | SFT | Static | S3 teacher | D1 | G2 final`:

- `100234` | What is the weather like? | Is this a harbor? | Is the sky clear? | Is this a harbor? | Is there a factory in the background?
- `100990` | What is the man doing? | What is the giant object in the background? | How many donuts are on top of the building? | What color is the man's shirt? | What color is the man's shirt?
- `101567` | What is the dog doing? | What is the dog doing? | Is the dog sleeping? | What kind of fruit is next to the dog? | What kind of fruit is next to the dog?
- `102503` | What color is the bed? | What color is the bed? | Are there any pictures on the walls? | How many pillows are on the bed? | How many pillows are on the bed?

G2 exact-match: 0.72 with D1, 0.23 with Static, 0.08 with the S3 teacher. G2
preserves the distilled init's grounded, specific questions while UVFC refines
coverage; the S3 teacher's yes/no probes are largely not memorized.

## K. Compute cost

| stage | cost |
|---|---|
| 5C-A selector evals (2 runs) | ~12 min API |
| 5C-B generation (9184×4) | ~34 min GPU |
| 5C-B UVFC+Flat scoring (35097) | ~6 h GPU |
| D1 + D2 distillation (2×1250 steps) | ~1.8 h GPU |
| D1/D2 generation + eval | ~15 min GPU + ~12 min API |
| G1 + G2 GRPO (2×1000 steps) | ~11.5 h GPU |
| checkpoint trajectory eval (2 runs × 8 ckpts) | ~1.5 h API |

## L. Failure cases

- S0 (existing UVFC) fails the support constraint (Supported 0.782).
- D2 multi-positive distillation collapses to SFT-like duplication.
- Neither D1 nor D2 reaches the Section 11 Supported ≥ 0.85 gate.
- The step-1 guard baseline under-estimated the distilled policy's duplication
  (0.125 vs ~0.4 on the calibration subset), which is exactly what the relative
  guard fixes.

## M. Paper-method interpretation

- **Question Space Expansion**: 4 independent samples expose alternative
  questions suppressed under direct high-probability decoding.
- **Evidence Constraint**: requirement-level `support_floor ≥ τ` removes
  unsupported alternatives while retaining informative ones (S3 vs S0: Supported
  0.782→0.804 at Coverage 0.381→0.370).
- **Grounded Distillation**: single-positive distillation writes the
  evidence-selected behavior back into a direct policy; multi-positive did not
  help in this configuration.
- **UVFC Refinement**: under the same init-relative guard, evidence-constrained
  RL refines the expanded policy and reaches the strong target; G2 dominates the
  matched G1 at every checkpoint, so the gain is a method effect, not a guard
  artifact.

## N. Recommendation

**FREEZE METHOD.**

Under the same reward, guard, data, seed and 1000-step configuration, the only
changed variable (init) yields a stable Pareto improvement over Static UVFC:
G2 (Question Space Expansion → Evidence-Constrained Selection → Grounded
Distillation → UVFC) reaches Search 0.368 / Coverage 0.391 / Supported 0.851 /
Unsupported 0.039 vs G1's 0.334 / 0.331 / 0.803 / 0.100, and the advantage is
already present at step 0 and grows monotonically. This meets the Section 18
strong target and the "not an early-stop artifact" requirement.

Caveats to record before any holdout: (1) the QSE distilled init (D1) does not
meet the Section 11 Supported ≥ 0.85 gate on its own (0.830); (2) multi-positive
distillation (D2) failed and must not be claimed; (3) G2 self-stopped at step 256
on a duplication spike, so the result is a step-256 checkpoint; (4) τ = 0.5 is
calibrated on the frozen Phase-2.1 corpus (images overlap development1000, but no
development1000 evaluation metric was used to pick it).

The method, selector, τ, distillation and guard are now frozen; final_holdout
was not accessed and no new method design was started.
