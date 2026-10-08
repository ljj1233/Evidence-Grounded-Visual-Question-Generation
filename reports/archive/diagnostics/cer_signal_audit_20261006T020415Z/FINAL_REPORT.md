# FINAL REPORT — CER counterfactual signal audit

Read-only analysis + bounded local model diagnostics. No optimizer step, no training,
no external API call, no `final_holdout` access.

## Headline

- **STATUS: INSUFFICIENT_EVIDENCE** (implementation valid; within-image CIs straddle the null,
  so no direction can be asserted at the pre-registered confidence).
- The `delta_cf` counterfactual gap does **not** separate Supported from Unsupported
  questions on the same image (point estimate slightly negative), while the simple
  original-image confidence `lp_plus` **does** separate them.
- Proceeding to a margin calibration of this `delta_cf` objective is **not supported** by
  this evidence.

## 1. FACT — inputs and provenance

See `INPUT_MANIFEST.json`.
- candidate pool `outputs/phase5b_direct4_uvfc/candidates_scored.jsonl` = 3811 records / 1000
  dev1000 images; its `overall_visual_support`/`support_floor` are **LOCAL Flat**, not external.
- external support `outputs/a0_candidate_oracle_20261004T080307Z/evaluation/support/a0_direct4.jsonl`
  = GPT-4o, `evaluator_version=visual-support-phase23-v1.0`, 3811 records, 3808 `status=ok`,
  labels supported 2622 / uncertain 588 / unsupported 598. Join key `(image_id, question)`,
  3811/3811 candidate keys covered.
- external SeeQ coverage `.../evaluation/seeq/scores.jsonl`, model `gpt-5.6-luna`,
  variant `VARIANT_DOCUMENTED` (descriptive only).
- The two label sources are **not mixed**: external separation uses GPT-4o only; Flat is a
  separate comparison signal.

## 2. FACT — negative-image definition (Q1)

See `TRANSFORM_AUDIT.json`.
- type = **CROSS_IMAGE_MISMATCH** (not within-image spatial shuffle).
- canonical derivation `reports/published_baseline_intervention_20261004T032525Z/scripts/build_transforms.py:31-38`
  (`derangement`: `np.random.default_rng(42).permutation(n)`, fixed points rolled out);
  CER wrapper `src/vvqg/evidence_constrained/counterfactual.py:39,82-98`.
- seed 42; donor = a different real image, resized to the source size, JPEG q95; fixed pairing;
  donor != original guaranteed; 0 fixed points.
- `MATCHES_HISTORICAL_INTERVENTION = True` (same function/parameters); the CER local set used
  train2014 while the historical intervention used dev1000.
- The candidate-level audit **reuses the historical dev1000 shuffled images** verbatim.

## 3. FACT — logprob implementation (Q2)

See `LOGPROB_CHECKS.csv`.
- Independent reference path (standard `processor(...)` call + manual causal shift) vs the
  existing `encode_items`/`gather_token_logprobs`: **max abs error = 0.0** over 8 anchors.
- question tokens contiguous = True for all 8.
- The token following the question is `151645` (`<|im_end|>`); it is **not** included in the
  completion (both sides use question tokens only).
- prompt/user/image tokens excluded (question tokens are verified contiguous right after the
  prompt); no padding in the single-item reference.
- `delta_cf` in the logprob check reproduces the prior diagnostic (unique active sample
  image `...000542.jpg`, delta 0.33).

## 4. FACT — CF gradient path (Q3)

See `GRADIENT_CHECKS.json`. Base checkpoint only, backward only, 0 optimizer steps.
- active sample (delta 0.367, margin 0.5): loss 0.1325, 504 nonzero-gradient LoRA params,
  global L2 = 10.78, max |grad| = 1.07, finite.
- inactive sample (delta 7.476): loss 0.0, **0** nonzero-gradient params (relu saturated).
- connectivity probe (test-only margin = detached delta + 0.1): 504 nonzero params,
  L2 = 59.70, finite → the CF gradient path is connected.
- KL: reference is the frozen Base; student == reference at init → KL ≈ 0 and KL grad ≈ 0 is
  expected and not a failure.

## 5. FACT — smoke batch reconciliation

See `SMOKE_BATCH_RECONCILIATION.json`.
- Reconstructed order (eligible[:8] after `random.Random(42).shuffle`, batches of 4) reproduces
  the recorded per-step delta means (step1 4.471 vs recorded 4.469; step2 3.528 vs 3.565;
  step2 uses the once-updated adapter).
- `active_count_margin_0.5 = 0` for **both** steps; recorded `L_cf = 0.0` for both.
- Resolution of the prior report's wording: `L_cf ≈ 0.0019` was the mean margin loss over all 84
  eligible anchors (the single active anchor contributes the entire 0.0019); the smoke batches
  contained 0 active anchors, so their per-step `L_cf` and CF gradient were exactly 0. Both
  statements are consistent: the hinge value is ~0.002 over the population, and identically 0
  on every full batch because saturation is per-sample.

## 6. FACT — candidate-level support separation (Q4)

256 dev1000 images sampled seed 42 (label-blind), 972 candidates, all GPT-4o-labeled. Base Full
Ours scored `lp_plus` (original) and `lp_minus` (historical shuffled donor). 118 s, peak 22 GB.

Three-class (`SUPPORT_SEPARATION.csv`, top block):

| label | n_cand | n_img | gap_mean | gap_p10 | gap_p50 | gap_p90 | lp_plus_mean | lp_minus_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| supported | 678 | 254 | 3.511 | 1.427 | 3.567 | 5.378 | -0.989 | -4.500 |
| uncertain | 143 | 110 | 3.742 | 1.523 | 3.711 | 6.000 | -1.137 | -4.880 |
| unsupported | 151 | 114 | 3.634 | 1.411 | 3.672 | 5.980 | -1.435 | -5.069 |

Within-image, mixed-label images = 112 (>= 30), image-level bootstrap 5000 seed 42:

| signal | d_i (supported - unsupported) | 95% CI | concordance | 95% CI |
|---|---:|---|---:|---|
| delta_cf | -0.099 | [-0.414, 0.216] | 0.499 | [0.426, 0.570] |
| lp_plus | +0.398 | [0.247, 0.551] | 0.688 | [0.617, 0.760] |
| flat_support_floor (local) | +0.344 | [0.244, 0.439] | 0.718 | [0.658, 0.772] |

- `delta_cf` is ~flat across external classes and slightly negative within image; `lp_plus`
  alone separates cleanly. Because `lp_minus` also decreases for unsupported questions, the
  difference cancels the support signal that `lp_plus` carries.
- `GAP_LENGTH_SPEARMAN = -0.224` (weak; longer questions → smaller gap).

## 7. FACT — margin descriptive (Q5 inputs)

`MARGIN_DESCRIPTIVE.json`: n=84, min 0.338, p10 2.178, p25 2.741, p50 3.913, p75 4.741,
p90 5.968, max 7.510. Active fraction at margin 0.5 = 0.0119; at p25/p50/p75/p90 = 0.238 /
0.488 / 0.738 / 0.881. A larger active fraction only says more anchors produce a hinge
gradient; it does **not** imply Supported rises.

## 8. INTERPRETATION

- **Not an implementation failure.** Logprob encoding/shift/mask match an independent reference
  exactly; the CF gradient path is connected; pairing and donor rules are faithful.
- **Not primarily a margin failure for support quality.** `margin=0.5` makes the term inactive,
  but activating it would not help: the underlying `delta_cf` construct has no within-image
  Supported-vs-Unsupported separation, so scaling its gradient cannot be expected to raise
  Supported.
- **It is a signal/proxy failure.** The minuend that carries support information is `lp_plus`;
  subtracting `lp_minus` removes it. Optimising `delta_cf` rewards "different-image
  discriminability", which the Base policy already has, not visual support.
- **Failure class for the prior CER run: proxy failure**, not implementation and not merely a
  mis-set margin.

## 9. UNKNOWN

- Whether an active-margin CER run changes any external metric (not run; not justified).
- Whether a blur negative differs from the shuffle negative on this diagnostic.
- Whether SeeQ coverage relates to `delta_cf` (only descriptive SeeQ cache exists; not used for
  any selection).
- The exact GPT-4o-vs-SeeQ agreement on this candidate set (not audited here).

## 10. Files

`INPUT_MANIFEST.json`, `TRANSFORM_AUDIT.json`, `LOGPROB_CHECKS.csv`, `GRADIENT_CHECKS.json`,
`SMOKE_BATCH_RECONCILIATION.json`, `CANDIDATE_SCORES.jsonl`, `SUPPORT_SEPARATION.csv`,
`WITHIN_IMAGE_STATISTICS.json`, `MARGIN_DESCRIPTIVE.json`, `CASES.jsonl`,
`MISSING_INFORMATION.md`, `COMMANDS.md`, `GIT_DIFF_DIAGNOSTIC.patch`.

## 11. Next single action

Do **not** calibrate the margin of this `delta_cf` objective. If CER is revisited, the single
change to test is the objective/negative: keep the original-image term and select/replace the
negative so the contrast preserves the `lp_plus` support signal (e.g. test `lp_plus` alone as a
sanity baseline, and a content-matched negative). This is a new pre-registered experiment; none
was started here.
