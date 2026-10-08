# PHASE A DIAGNOSTIC — G3 follow-up (read-only)

Report dir: `reports/g3_followup_20261005T161303Z/`
Read-only. No training in Phase A. seed 42. `final_holdout` not accessed.

## Actual files used

| item | path |
|---|---|
| G3 dir | `reports/g3_support_preserving_rl_20261005T105701Z/` |
| reward sanity | `.../reward_sanity.json` + `REWARD_SANITY_CHECK.md` |
| checkpoint metrics | `.../CHECKPOINT_METRICS.csv` |
| ratio/clip | `.../RATIO_CLIP_DIAGNOSTICS.csv` + `.../pilot/train_log.jsonl` |
| support | `.../SUPPORT_DIAGNOSTICS.csv` |
| degeneration | `.../DEGENERATION_DIAGNOSTICS.csv` |
| training summary | `.../TRAINING_SUMMARY.json` |
| final report | `.../FINAL_REPORT.md` |
| reward classification labels | `reports/a0_candidate_oracle_20261004T080307Z/candidate_external_scores.jsonl` |
| group-variance rollout (fresh, inference-only) | `group_variance.json` |

## 1. Reward component statistics (A0 oracle, n=3808 labeled candidates)

| component | mean | std | p25 | p50 | p75 |
|---|---:|---:|---:|---:|---:|
| support_continuous | 0.2663 | 0.9721 | −0.50 | 1.00 | 1.00 |
| coverage_anchor | 0.0003 | 0.7567 | −1.00 | 0.00 | 1.00 |
| duplicate_penalty | 0.0000 | 0.0000 | 0.00 | 0.00 | 0.00 |
| generic_penalty | 0.0058 | 0.0758 | 0.00 | 0.00 | 0.00 |
| final reward | 0.2655 | 0.9821 | — | — | — |

## 2. Reward classification ability

| external label | n | reward mean | reward std |
|---|---:|---:|---:|
| supported | 2622 | **0.4749** | 0.8740 |
| uncertain | 588 | **0.0092** | 0.9896 |
| unsupported | 598 | **−0.4006** | 1.0628 |

- reward(supported) − reward(unsupported) = **0.8575**, image-level 95% CI **[0.7397, 0.9718]**.
- reward(supported) − reward(uncertain) = **0.4423**, 95% CI **[0.3400, 0.5440]**.
- support_continuous vs external Supported: Spearman **0.417**, Pearson **0.341**.
- Monotonic order supported > uncertain > unsupported: **yes**; both CIs exclude 0.

## 3. Group / advantage statistics

Per-step `groups_skipped` from `SUPPORT_DIAGNOSTICS.csv`: **0 / 2400 groups**
across all 300 steps → **zero-variance group rate 0.000**. Fixed-subset
read-only rollout diagnostic (base checkpoint, 128 train images, K=4, seed 42):

| quantity | value |
|---|---:|
| zero-variance group rate | 0.000 |
| mean group reward std | 0.5176 |
| mean advantage std | 1.0000 |
| parser-failure rate | 0.0137 |
| malformed rate | 0.000 |
| local supported / unsupported | 0.730 / 0.221 |

Group variance is healthy; advantage std is 1.0 by construction.

## 4. Policy update statistics (300 steps)

| quantity | min | max |
|---|---:|---:|
| ratio_mean | 0.99264 | 1.00978 |
| ratio_min | 0.65248 | 0.89996 |
| ratio_max | 1.09656 | 1.52337 |
| clip_fraction | 0.00000 | 0.03226 |
| KL to init | 0.00050 | 0.00386 |
| entropy | 0.41142 | 0.66391 |
| grad norm | 0.66254 | 1.94384 |

`stop_reason = null`. Per-token ratio p01/p50/p99 were not logged (only
mean/min/max per step); min/max are reported as bounds.

Per-checkpoint local metrics (256 images) are in `CHECKPOINT_METRICS.csv`; external
step0/final in the G3 `PAIRED_BOOTSTRAP_RESULTS.csv`.

## 5. Trend analysis

- Reward monotonic across classes (supported > uncertain > unsupported), CI lower
  bound > 0 → reward is **valid**.
- Zero-variance group rate **0.000** (< 0.30) and skipped rate **0.000** (< 0.20)
  → group variance is **not** the problem.
- G3 stayed in a **very small-KL / very small-clip regime**: KL max 0.00386
  (≤ 0.004), clip max 0.03226 (≤ 0.05), ratio_mean within ±1% of 1.
- Local metrics: UVFC rose monotonically (0.724→0.769); local Flat Supported
  drifted **down** (0.770→0.734) after step 50 — no clean up-then-down pattern in
  the reward, but the proxy support metric does not improve.

## Decision

**CASE 3 — reward valid, group variance fine, policy update too conservative.**

| rule | threshold | observed | result |
|---|---|---|---|
| reward order correct | supported>uncertain>unsupported | yes | OK |
| supported−unsupported CI lower > 0 | > 0 | 0.7397 | OK |
| zero-variance rate < 0.30 | < 0.30 | 0.000 | OK |
| skipped rate < 0.20 | < 0.20 | 0.000 | OK |
| KL max ≤ 0.004 | ≤ 0.004 | 0.00386 | **triggers Case 3** |
| clip max ≤ 0.05 | ≤ 0.05 | 0.03226 | **triggers Case 3** |
| ratio ≈ 1 | — | 0.993–1.010 | **triggers Case 3** |

**Authorized follow-up:** raise learning rate `5e-7 → 1e-6`, keep K=4,
`beta_kl=0.02`, `inner_epochs=2`, `clip_epsilon=0.2`, `max_steps=300`. Reward
unchanged. No simultaneous change of K and lr.
