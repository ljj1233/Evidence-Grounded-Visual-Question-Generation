# FINAL REPORT — G3 conservative support-preserving RL

## Headline

- **Pilot: FAIL.** The 256-image pilot gate did **not** pass.
- **Formal 1000-step run: NOT started.**
- G3 − step0 Full Ours (256 images, 3-repeat evaluator mean, paired image bootstrap
  seed 42, 5000 resamples):
  - Coverage **+0.00945** 95% CI **[−0.00630, +0.02625]**
  - Supported **+0.01042** 95% CI **[−0.01432, +0.03516]**
  - Unsupported **−0.00781** 95% CI **[−0.02344, +0.00654]**
- **ratio truly varies** (ratio_mean range 0.99264–1.00978); **clip_fraction > 0**
  (range 0.0–0.03226) → the old `ratio≡1` bug is fixed.
- **No degeneration** (`stop_reason = null`; duplicate rate 0.184→0.199 ≤ +0.05).
- **`final_holdout` NOT accessed.**

## 1. Base model and parity

- base checkpoint `outputs/phase5c_qse_relativeguard/checkpoints/step_final`
  (global_step 256, early-stopped); adapter sha256
  `52ee692ca08737058594fe408a7817419f53532395f9249b2f227b43b4d9a58d`.
- original parity 1000/1000 (verified in the Full Ours intervention task).
- instruction `Generate a question about this image.`; greedy T=0, mn=32;
  rollout T=0.8, top_p 0.9, mn=32; seed 42.

## 2. Data

- train manifest `outputs/phase4b_online_uvfc/data/train_10000.jsonl` →
  deduplicated by image to **9184 images** (`pilot/train_manifest_resolved.jsonl`).
- train ∩ development1000 = **0**.
- 256-image pilot manifest `reports/vc_qse_20261005T035840Z/dev256_manifest.jsonl`
  (first 256 by integer image_id; disjoint from train; no `final_holdout`).

## 3. Reward sanity check — PASS

`REWARD_SANITY_CHECK.md`. support reward separates external classes
(supported +0.475 / uncertain +0.011 / unsupported −0.400), Spearman 0.417 with
external Supported, UVFC–Coverage Spearman 0.320; generic rate 0.006, 0 generic
in top-100 reward. Training authorized.

## 4. Reward

`REWARD_DEFINITION.md`:
`R = 1.00*support_continuous + 0.20*coverage_anchor − 0.25*duplicate_penalty − 0.15*generic_penalty`.

## 5. Corrected PPO

`old_logprob` fixed at rollout time (detached, reused across both inner epochs);
`new_logprob` from the grad policy; `ratio=exp(new−old)`; PPO clip 0.2;
KL anchor = fixed initial Full Ours; `beta_kl=0.02`, `lr=5e-7`, `max_grad_norm=1.0`.
See `RATIO_CLIP_DIAGNOSTICS.csv` — ratio min/max per step move away from 1 and
clip_fraction is nonzero from step 2 onward.

## 6. Pilot training

- 300 steps, `inner_epochs=2`, `groups_per_step=8`, K=4, wall 9351 s, no stop.
- checkpoints `step_0000/0050/0100/0150/0200/0250/0300(+final)`.
- `TRAINING_SUMMARY.json`, `RATIO_CLIP_DIAGNOSTICS.csv`, `SUPPORT_DIAGNOSTICS.csv`,
  `DEGENERATION_DIAGNOSTICS.csv`.

## 7. Local checkpoint metrics (`CHECKPOINT_METRICS.csv`, 256 images)

| tag | UVFC | support_floor | Supported | Unsupported | dup | generic | words |
|---|---:|---:|---:|---:|---:|---:|---:|
| step0 | 0.7239 | 0.8379 | 0.770 | 0.188 | 0.184 | 0.000 | 5.95 |
| step50 | 0.7349 | 0.8340 | 0.762 | 0.188 | 0.172 | 0.000 | 5.91 |
| step100 | 0.7457 | 0.8418 | 0.758 | 0.199 | 0.184 | 0.000 | 5.95 |
| step150 | 0.7557 | 0.8164 | 0.734 | 0.215 | 0.176 | 0.000 | 5.98 |
| step200 | 0.7579 | 0.8281 | 0.742 | 0.211 | 0.180 | 0.000 | 5.98 |
| step250 | 0.7551 | 0.8262 | 0.742 | 0.219 | 0.188 | 0.000 | 6.01 |
| final | 0.7688 | 0.8145 | 0.734 | 0.215 | 0.199 | 0.000 | 6.02 |

UVFC (reward anchor) rises, but the **local frozen Flat** Supported proxy drifts
**down** (0.770→0.734) — a proxy/evaluator mismatch worth noting.

## 8. External evaluation (256 images, 3 repeats, GPT-4o + SeeQ luna variant)

`PAIRED_BOOTSTRAP_RESULTS.csv`:

| metric | step0 | G3 | Δ | 95% CI | pos/neg/tie |
|---|---:|---:|---:|---|---|
| Coverage | 0.38464 | 0.39396 | +0.00945 | [−0.00630, +0.02625] | 47/50/157 (n=254) |
| Supported | 0.85026 | 0.86068 | +0.01042 | [−0.01432, +0.03516] | 8/7/241 |
| Unsupported | 0.06641 | 0.05859 | −0.00781 | [−0.02344, +0.00654] | 3/6/247 |

Directions are favorable (Supported up, Unsupported down, Coverage up) but all
CIs cross 0 and the Supported gain is +0.010, below the +0.02 gate.

## 9. Pilot gate — FAIL

| # | condition | value | result |
|---|---|---|---|
| 1 | Supported point ≥ +0.02 | +0.01042 | **FAIL** |
| 2 | Supported CI lower > 0 | −0.01432 | **FAIL** |
| 3 | Coverage CI lower ≥ −0.01 | −0.00630 | PASS |
| 4 | Unsupported point decrease | −0.00781 | PASS |
| 5 | Unsupported CI upper ≤ 0 | +0.00654 | **FAIL** |
| 6 | duplicate increase ≤ 0.05 | +0.0148 | PASS |
| 7 | generic increase ≤ 0.03 | 0.000 | PASS |
| 8 | malformed no increase | 0.000 | PASS |
| 9 | ratio not ≡ 1 | varies | PASS |
| 10 | clip_fraction > 0 | up to 0.0323 | PASS |
| 11 | KL ≤ 0.2 | ≤0.00386 | PASS |
| 12 | no degeneration guard | none | PASS |

## 10. Decision

Pilot FAIL → **no 1000-step formal run, no new RL variant**. See
`FAILURE_ANALYSIS.md` for the diagnosis and minimal next-step suggestions.

## 11. Claims

- Do **not** claim G3 improves Supported/Coverage over Full Ours.
- Do **not** claim RL is useless; the pilot was underpowered and the policy barely
  moved (ratio ∈ [0.993, 1.010], KL ≤ 0.004).
- The corrected PPO implementation is valid (ratio/clip behave correctly).
