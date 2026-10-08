# FINAL REPORT — Support-Delta GRPO fixed-output re-review + reward attribution

Review-only. No training, no optimizer step, no new generation, no reward change.
The original pilot status is unchanged.

## Headline

- **REVIEW_STATUS: COMPLETE** (evaluation repeats 3/3 complete on original; reward attribution on available data).
- **ORIGINAL_PILOT_STATUS: FAIL_PILOT** (unchanged).
- The Coverage gain **repeats in all three rounds**, but is smaller on aggregate (`+0.0133` CI `[+0.0013, +0.0266]`).
- Supported is **not improved** (`-0.0039` CI `[-0.0273, +0.0182]`); Unsupported is not reduced.
- `-S(0)` is **exactly cancelled** by group standardization (`max |Δadvantage| = 5.8e-15`).
- Support **does** influence advantage where it varies, but in **27% of groups it is constant**.
- `TRAINING_AUTHORIZED_BY_THIS_TASK: NO`.

## 1. FACT — input and cache audit (`INPUT_AUDIT.json`)

- Pilot manifest: `reports/support_delta_grpo_20261006T022616Z/resolved_pilot_manifest.jsonl`
  sha256 `5c1277...ace5c8`, 256 images; equals the fixed G3 256 manifest.
- Base questions `pilot_eval/predictions_step0_original.jsonl`; candidate
  `pilot_eval/predictions_step300_original.jsonl`.
- Base adapter sha256 `52ee692c...d9a58d`; step300 adapter sha256 `bdd810a4...17b84c1` (and
  `step_0000` adapter sha == base, confirming step0 is π₀).
- One question per image and identical image sets/paths for both systems; cache questions match
  the fixed outputs verbatim. Flat labels are never used as external labels.
- Repeat 1 reproduced exactly (Coverage Base 0.37344 / step300 0.38906; Support 0.84375 / 0.83984).
- Trained per-candidate rollout logs are **absent** (`pilot_final/rollouts/` empty).

## 2. FACT — Base parity (`BASE_PARITY_AUDIT.md`)

- G3's step0 predictions on the same manifest are **256/256 identical questions** to this Base.
- G3 reported Base Coverage 0.38464 / Supported 0.85026 vs this review's repeat-1
  0.37344 / 0.84375 → difference is **judge fluctuation / API nondeterminism**, not sample or question.

## 3. FACT — evaluator provenance (`EVALUATOR_PROVENANCE.json`)

- Support: `gpt-4o`, `visual-support-phase23-v1.0`; cache identity = version, model, endpoint,
  system prompt, image_id, image_hash, question; labels via the frozen `overall_support` rule.
- Coverage: SeeQ `gpt-5.6-luna`, `seeq-single-question-v1`, status `VARIANT_DOCUMENTED`;
  coverage = `scores.visual_evidence_coverage / 5`; cache identity includes prompt_hash/image_id/image_hash/question.
- No judge model, prompt, or human correction was added.

## 4. FACT — repeats (`REPEAT_METRICS.csv`)

| repeat | metric | n | Base | step300 | delta | 95% CI |
|---|---|---:|---:|---:|---:|---|
| 1 | coverage | 256 | 0.37344 | 0.38906 | +0.01563 | [+0.00391, +0.02813] |
| 1 | supported | 256 | 0.84375 | 0.83984 | -0.00391 | [-0.02734, +0.01953] |
| 1 | unsupported | 256 | 0.05078 | 0.05859 | +0.00781 | [-0.01172, +0.02734] |
| 2 | coverage | 256 | 0.38750 | 0.40391 | +0.01641 | [+0.00156, +0.03281] |
| 2 | supported | 256 | 0.83203 | 0.83203 | 0.00000 | [-0.02344, +0.02344] |
| 2 | unsupported | 256 | 0.05469 | 0.06250 | +0.00781 | [-0.01172, +0.02734] |
| 3 | coverage | 256 | 0.38672 | 0.39453 | +0.00781 | [-0.00703, +0.02266] |
| 3 | supported | 256 | 0.84766 | 0.83984 | -0.00781 | [-0.03125, +0.01563] |
| 3 | unsupported | 256 | 0.04297 | 0.04688 | +0.00391 | [-0.01172, +0.01953] |

## 5. FACT — three-round aggregate (`PAIRED_AGGREGATE_RESULTS.csv`, image-level bootstrap 5000, seed 42)

| metric | common_n | Base | step300 | delta | 95% CI |
|---|---:|---:|---:|---:|---|
| coverage | 256 | 0.38255 | 0.39583 | +0.01328 | [+0.00130, +0.02656] |
| supported | 256 | 0.84115 | 0.83724 | -0.00391 | [-0.02734, +0.01823] |
| unsupported | 256 | 0.04948 | 0.05599 | +0.00651 | [-0.01042, +0.02474] |

Label flip across the 3 repeats: step0 16/256 (6.25%), step300 17/256 (6.64%).
All three Coverage rounds are positive; Supported rounds are `-0.0039 / 0 / -0.0078`.

## 6. FACT — reward formula and attribution

Actual formula (`src/vvqg/support_delta/reward.py`):
```
R_i = delta_support_i - 0.50*coverage_guard_i - 0.25*duplicate_penalty_i - 0.50*malformed_penalty_i
delta_support_i = S(q_i) - S(q_0)
A_i = (R_i - mean_group(R)) / std_group(R)     (0 if std <= 1e-8)
```
No clipping, rank transform, filtering, cross-image normalization, or per-component
standardization on the reward. Attribution data is the saved frozen-π₀ sanity rollout
(128 groups / 1024 candidates); trained step300 per-candidate rollouts were not saved.

| quantity | value |
|---|---:|
| group support zero-variance rate | 0.2734 |
| group total-reward zero-variance rate | 0.0313 |
| support-constant-but-total-variable rate | 0.2422 |
| delta_support nonzero candidate rate | 0.3096 |
| distinct support values per group | 1:35, 2:62, 3:29, 4:2 |
| **WITH_VS_WITHOUT_S0_MAX_ADVANTAGE_DIFF** | **5.773e-15** |

Centered component std (cross-group mean, p10/p50/p90) and corr with centered reward:

| component | weight | mean | p10 | p50 | p90 | corr |
|---|---:|---:|---:|---:|---:|---:|
| support | +1.00 | 0.499 | 0.000 | 0.496 | 0.893 | 0.929 |
| coverage_guard | -0.50 | 0.066 | 0.000 | 0.067 | 0.131 | 0.140 |
| duplicate_penalty | -0.25 | 0.102 | 0.083 | 0.108 | 0.125 | 0.202 |
| malformed_penalty | -0.50 | 0.015 | 0.000 | 0.000 | 0.000 | 0.063 |

Advantage sensitivity (real `group_advantages`, ties kept, sign tol 1e-8):

| removed | mean abs ΔA | max abs ΔA | sign change | top-set change | zero-var rate | cosine |
|---|---:|---:|---:|---:|---:|---:|
| B without S0 | 1.3e-16 | 5.8e-15 | 0.0000 | 0.0000 | 0.031 | 1.000 |
| C without support | 0.6464 | 5.2915 | 0.3145 | 0.3672 | 0.039 | 0.435 |
| D without guard | 0.1460 | 2.8515 | 0.0654 | 0.5547 | 0.039 | 0.944 |
| E without duplicate | 0.2986 | 5.2915 | 0.1787 | 0.5156 | 0.109 | 0.836 |
| F without malformed | 0.0067 | 0.3260 | 0.0010 | 0.0000 | 0.031 | 0.999 |

## 7. INTERPRETATION

- Coverage gain is **reproducible in direction across 3 rounds** and robust on aggregate
  (CI lower +0.0013 > 0). It is nevertheless small and round 3 alone is not significant.
- Supported is **not improved**; its aggregate CI crosses 0. The result is compatible with,
  but does not establish, a non-inferiority margin of 1%.
- `-S(0)` is an exact per-group constant and is fully removed by standardization; it cannot
  affect advantages (verified numerically, not only theoretically).
- Support is the largest source of reward variation on the sampled π₀ actions and materially
  changes advantages (cosine 0.435 when removed), except in the 27% of groups where its score is
  constant. The guard changes the top-advantage set in 55% of groups despite a small sign-change rate.
- The attribution uses π₀ rollouts, not the trained step300 rollouts, so it explains the reward
  geometry rather than proving which component caused the Coverage gain.

## 8. UNKNOWN

- Causal attribution of the Coverage gain to a specific component (needs retraining ablations).
- Whether the same attribution holds on the trained step300 action distribution.
- Whether the Coverage gain survives a different/independent judge (systematic judge bias cannot
  be removed by 3 repeats).

## 9. Gate (unchanged, per original rules)

| # | condition | value | result |
|---|---|---|---|
| 1 | Supported point ≥ +0.015 | -0.0039 | FAIL |
| 2 | Supported CI lower > 0 | -0.0273 | FAIL |
| 3 | Coverage CI lower ≥ -0.01 | +0.0013 | PASS |
| 4 | Unsupported point < 0 | +0.0065 | FAIL |
| 5 | Unsupported CI upper ≤ 0 | +0.0247 | FAIL |
| 6 | orig-shuffled gap increase | -0.0156 | FAIL |
| 7-9 | duplicate/generic/length | +0.000/+0.000/+1.9% | PASS |
| 10 | KL ≤ 0.01 | 0.0070 | PASS |
| 11 | ratio not ≡1, clip>0 | varies / up to 0.10 | PASS |
| 12 | 3 evaluator repeats | 3/3 done, directions consistent for Coverage | PASS |

**The re-review does not change FAIL_PILOT.** The 1-percentage-point tolerance is not a
pre-registered non-inferiority bound.

## 10. Files

`INPUT_AUDIT.json`, `BASE_PARITY_AUDIT.md`, `EVALUATOR_PROVENANCE.json`, `REPEAT_METRICS.csv`,
`PAIRED_AGGREGATE_RESULTS.csv`, `PER_IMAGE_REPEAT_SCORES.jsonl`, `LABEL_FLIP_SUMMARY.json`,
`REWARD_SOURCE_AUDIT.md`, `REWARD_DIAGNOSTICS.json`, `GROUP_SUPPORT_VARIATION.csv`,
`CENTERED_REWARD_COMPONENTS.csv`, `ADVANTAGE_SENSITIVITY.csv`, `CASES.jsonl`,
`MISSING_INFORMATION.md`, `COMMANDS.md`, `GIT_DIFF_DIAGNOSTIC.patch`.
