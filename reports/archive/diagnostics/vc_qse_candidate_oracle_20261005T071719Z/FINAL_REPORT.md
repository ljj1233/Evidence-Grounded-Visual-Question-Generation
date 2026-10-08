# FINAL REPORT — VC-QSE Candidate-Pool Oracle Decomposition

Run: `reports/vc_qse_candidate_oracle_20261005T071719Z/` · seed 42 · git `9dea27f0`

**This is a candidate-level diagnostic.** Oracle labels are used only for offline
diagnosis. No model was trained. `final_holdout` was not accessed. The VC-QSE
selector was not modified. This is a **single-pass** evaluator result and is not
equivalent to multi-judge stability.

## 1. Input files (actual)

| role | path |
|---|---|
| manifest | `reports/vc_qse_20261005T035840Z/dev256_manifest.jsonl` |
| VC candidates (pre-selector) | `reports/vc_qse_20261005T035840Z/pilot256/vc/vc_candidates.jsonl` |
| VC selected | `reports/vc_qse_20261005T035840Z/pilot256/selected/vc_selected.jsonl` |
| C0 candidates (preferred) | `outputs/phase5b_direct4_uvfc/candidates_scored.jsonl` (filtered to dev256) |
| C0 selected | `reports/vc_qse_20261005T035840Z/pilot256/selected/c0_selected.jsonl` |
| VC local Flat support | `reports/vc_qse_20261005T035840Z/pilot256/vc/vc_support.jsonl` |
| VC adherence | `reports/vc_qse_20261005T035840Z/pilot256/vc/vc_adherence.jsonl` |
| SeeQ candidate cache (union) | `reports/vc_qse_candidate_oracle_20261005T071719Z/evaluation/seeq_cache.jsonl` |
| VC UVFC field | `reports/vc_qse_candidate_oracle_20261005T071719Z/evaluation/vc_uvfc.jsonl` |

C0 support fields are the frozen local Flat labels already present in the uvfc
scored pool; VC support fields are the frozen local Flat labels from
`vc_support.jsonl`. **No candidate-level GPT-4o support cache exists for these
two pools**; candidate-level support therefore uses the same frozen Flat verifier
that both selectors use. No candidate-level GPT-4o pass was run (the labelled
gate in the previous task already covered the 256 selected questions).

## 2. 256-image manifest

sha256 `5c1277796d776e630157f0ef610b7f54b9ff9c9e48863d3d8f4c5e9408ace5c8`
(development1000 sorted by integer image_id, first 256; seed 42). C0 and VC-QSE
use this same manifest; join key is `(image_id, image_hash, question)`.
Train overlap and `final_holdout` overlap are **0**.

## 3–8. Candidate counts and evaluator calls

| quantity | value |
|---|---:|
| C0 candidates (dev256) | **962** |
| VC-QSE candidates (raw) | **1024** |
| VC-QSE unique keys | 896 (128 duplicate rows collapsed) |
| total candidate rows | 1986 |
| **unique evaluator keys** | **1858** |
| **exact cache hits** | **1188** |
| **new evaluator calls** | **670** |
| evaluator failures (SeeQ) | **0** (1858/1858 ok) |
| candidate-level support failures | 0 |

Single-pass SeeQ: requested 1858, cached 1188, new 670, ok 1858. Cache keys:
SeeQ `sha256(version, model, prompt_hash, image_id, image_hash, question)`; each
key requested at most once (append-only resumable cache).

## 9–10. Candidate-level oracle comparison

| metric | C0_DIRECT4 | VC-QSE |
|---|---:|---:|
| candidates | 962 | 896 |
| candidates/image (min/med/max) | 1 / 4 / 4 | 1 / 4 / 4 |
| % images with ≥1 supported candidate | 96.48% | 93.75% |
| % images with ≥1 high-Coverage supported | 45.70% | 37.11% |
| **candidate oracle Coverage** | **2.5078** | **2.4727** |
| **support-constrained oracle** | **2.1498** | **2.0292** |
| conservative oracle | 2.1498 | 2.0292 |
| candidate mean Coverage | 1.6726 | 1.6540 |
| supported candidate rate | 0.6071 | 0.5982 |
| unsupported candidate rate | 0.3430 | 0.3438 |
| uncertain candidate rate | 0.0291 | 0.0446 |

`candidate oracle` = per-image max Coverage over all candidates.
`support_constrained_oracle` = per-image max Coverage over supported candidates
(diagnostic upper bound only; **not** a method). Missing supported candidate:
C0 9 images, VC 16 images.

## 9/12. Selector table (Coverage / Supported / Unsupported + regret + runtime)

| selector | pool | Coverage | Supported | Unsupported | fallback | Pareto-dom. | regret vs support-oracle | regret vs candidate-oracle |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| c0_original (=support_first_c0) | C0 | 1.8359 | 0.9648 | 0.0234 | 0.000 | 0.152 | 0.308 | 0.672 |
| **vc_selector** (actual) | VC | **1.5781** | 0.9648 | **0.0078** | 0.129 | 0.242 | **0.462** | 0.894 |
| support_first_vc (diag) | VC | 1.7461 | 0.9375 | 0.0430 | 0.000 | 0.188 | 0.296 | 0.727 |
| uvfc_vc (diag) | VC | 1.7930 | 0.8438 | 0.0430 | 0.000 | 0.238 | 0.250 | 0.680 |
| uvfc_c0 (diag) | C0 | 1.8750 | 0.8828 | 0.0234 | 0.000 | 0.188 | 0.267 | 0.633 |
| random_vc (seed 42) | VC | 1.6719 | 0.8672 | 0.0430 | 0.000 | 0.270 | 0.396 | 0.801 |
| random_c0 (seed 42) | C0 | 1.6836 | 0.8984 | 0.0234 | 0.000 | 0.270 | 0.482 | 0.824 |
| support_constrained_oracle | C0 / VC | 2.1498 / 2.0292 | — | — | — | — | — | — |
| conservative_oracle | C0 / VC | 2.1498 / 2.0292 | — | — | — | — | — | — |

VC_SELECTOR regret (vs support-oracle): median 0.0, p25 0.0, p75 2.0, exact-oracle
match 0.691, oracle-exceeded 0.016. Runtime: candidate build ~1 s, UVFC scoring
~GPU 2 min (1024 rows), full offline selector+oracle+bootstrap **9.2 s**, cases
~4 s.

## 13. Concept-type strata for VC (final VC_SELECTOR selection)

| type | images | candidates | cand mean cov | candidate oracle | support oracle | selected cov | Supported | Unsupported | confidence mean | adherence=yes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| object | 84 | 299 | 1.552 | 2.238 | 1.893 | 1.393 | 0.699 | 0.251 | 0.974 | 0.753 |
| attribute | 37 | 130 | 1.677 | 2.622 | 2.108 | 1.405 | 0.608 | 0.308 | 0.980 | 0.692 |
| relation | 43 | 155 | 1.632 | 2.465 | 1.854 | 1.558 | 0.555 | 0.374 | 0.976 | 0.755 |
| scene | 59 | 205 | 1.820 | 2.729 | 2.483 | 1.966 | 0.639 | 0.307 | 0.976 | 0.737 |
| fallback | 33 | — | — | — | — | — | — | — | — | — |

Correlations (VC candidates): concept_confidence vs Coverage **−0.058** (vs
Supported −0.052); adherence=yes vs Coverage +0.043; word length vs Coverage
+0.111. Concept confidence is near-constant (≈0.975) and slightly *negatively*
related to Coverage, so confidence-ordered selection is not Coverage-aware.

**scene paired check:** scene-selected images n=59, VC_SELECTOR − C0 mean
**−0.186**, 95% CI **[−0.542, +0.169]** (14 up / 21 down / 24 tie). The scene
point estimate is **not** a reliable positive.

## 14. Paired bootstrap (5000 resamples, seed 42, resampling images)

| comparison | mean Δ Coverage | 95% CI | pos / neg / tie |
|---|---:|---|---|
| candidate oracle VC − C0 | −0.035 | [−0.199, +0.129] | 75 / 81 / 100 |
| support-constrained oracle VC − C0 | −0.116 | [−0.302, +0.065] | 62 / 75 / 95 |
| **VC_SELECTOR − C0_original** | **−0.258** | **[−0.414, −0.105]** | 41 / 70 / 145 |
| support_first VC − C0 | −0.090 | [−0.254, +0.082] | 55 / 72 / 129 |
| uvfc VC − C0 | −0.082 | [−0.254, +0.094] | 57 / 70 / 129 |
| random VC − C0 | −0.012 | [−0.184, +0.160] | 61 / 61 / 134 |

## 15. Core A/B answer

- **A (pool lacks high-Coverage questions): NOT established.** The VC-QSE
  candidate-pool oracle is statistically indistinguishable from C0
  (2.473 vs 2.508, Δ −0.035, CI [−0.199, +0.129]); the support-constrained oracle
  is also not significantly lower (2.029 vs 2.150, Δ −0.116, CI [−0.302, +0.065]).
  VC-QSE does not demonstrably lack high-Coverage candidates relative to C0.
- **B (selector loses them): SUPPORTED.** VC_SELECTOR is the worst VC selector:
  1.578 vs same-pool support-first 1.746, UVFC 1.793, and even random 1.672. Its
  regret vs the support-constrained oracle (0.462) is ~1.6× support-first's
  (0.296). VC_SELECTOR − C0 is the only paired difference with a CI excluding 0
  (−0.258, CI [−0.414, −0.105]).

**Framing:** the candidate-generation stage is roughly on par with C0; the
`VC_SELECTOR` rule (one-per-type + concept-confidence priority) is the dominant
proximate loss. However, even the support-constrained oracle ceiling for VC-QSE
(2.029) is below C0's (2.150) on the point estimate, so **no supported selector
on the current VC pool can reach C0**. The route does not beat C0 at the oracle
level either.

## 16. Selector failure cases (VC_SELECTOR below oracle), 30

```
9077  sel 1 / oracle 5 regret 4 scene:indoor market   "Is this indoors?" || "How many different colors are the fruit?"
10591 sel 1 / oracle 5 regret 4 attribute:blue court "Is the man standing on a blue court?" || "How many people are in the picture?"
22257 sel 1 / oracle 5 regret 4 scene:savanna        "Is there a tree in the background?" || "How many animals are there?"
35899 sel 1 / oracle 5 regret 4 relation:leaves on ground "Is this a forest?" || "How many trees?"
78892 sel 1 / oracle 5 regret 4 (fallback)           "What is this train called?" || "Is this a modern city?"
```
Full 30: `cases/selector_failure_oracle_higher.jsonl`.

## 17. Candidate-generation failure cases (no high-Coverage candidate), 30

```
6847  oracle 1 attribute:white shirt  "Is the man wearing a shirt?"
7730  oracle 1 scene:indoor          "Is this an indoor or outdoor picture?"
9395  oracle 1 object:baseball pitcher "What is the number of the pitcher?"
14921 oracle 1 object:toilet          "What color is the toilet?"
15278 oracle 1 object:broccoli        "What is the green vegetable?"
```
Full 30: `cases/candidate_generation_no_high_coverage.jsonl`.

Other case files: `cases/selector_correct.jsonl` (20), `cases/c0_beats_vc.jsonl`
(30), `cases/vc_beats_c0.jsonl` (30), `cases/scene_positive.jsonl` (20),
`cases/{object,attribute,relation}_failure.jsonl` (20 each). Each record carries
image path, method, concept_type/text/confidence, selected question/coverage/
support, oracle question/coverage/support, regret, fallback, selector.

## 18–20. Permissions

| action | decision |
|---|---|
| train VC-QSE / Full Ours / D1 | **NOT ALLOWED** |
| generate concept train data | **NOT ALLOWED** |
| RL / GRPO / PPO | **NOT ALLOWED** |
| further selector modification | **NOT ALLOWED** (conservative branch) |

Because the VC-QSE oracle is not above C0 (point estimate lower on both oracles),
the route is archived as **`FAILED_AS_MAIN_COVERAGE_METHOD`**. The selector is the
larger *proximate* loss, but the pool's support-constrained oracle ceiling is
below C0, so recovering it would not produce a method that beats C0.

## 21. Commands

See `RUN_COMMANDS.md`.

## 22. New files

`scripts/vvqg/vc_oracle_build_candidates.py`, `vc_oracle_select.py`,
`vc_oracle_cases.py`, `vc_oracle_uvfc.py`; `reports/vc_qse_candidate_oracle_20261005T071719Z/**`.
No existing file modified; no prior result overwritten.

## 23. Copyable summary (≤15 lines)

```
VC-QSE candidate-pool oracle decomposition. Single-pass SeeQ, seed 42, git 9dea27f0.
Inputs: VC pilot candidates (1024 raw / 896 unique) + C0 phase5b_direct4_uvfc (962, dev256).
Candidate keys: 1858 unique; exact cache hits 1188; new evaluator calls 670; failures 0.
Oracle Coverage: C0 2.508 vs VC 2.473 (delta -0.035, CI [-0.199,+0.129]).
Support-constrained oracle: C0 2.150 vs VC 2.029 (delta -0.116, CI [-0.302,+0.065]).
Selected Coverage: C0 1.836 vs VC_SELECTOR 1.578 (delta -0.258, CI [-0.414,-0.105]).
Same VC pool: support-first 1.746, UVFC 1.793, random 1.672 -> VC_SELECTOR is the worst.
A: pool NOT shown to lack high-Coverage candidates (oracle ~on par). B: selector is the dominant proximate loss.
Even support-oracle VC (2.029) < C0 support-oracle (2.150) -> no supported selector can reach C0.
Strata: object 1.39, attribute 1.41, relation 1.56, scene 1.97 selected; scene CI [-0.542,+0.169] (n=59, not positive).
Decision: archive VC-QSE FAILED_AS_MAIN_COVERAGE_METHOD; NO train, NO concept data, NO RL, NO selector edits.
Cases: selector_failure 30, candidate-generation failure 30, c0_beats_vc 30, vc_beats_c0 30, correct 20, scene 20.
Diagnostic only; oracle labels offline; no final_holdout; single-pass evaluator (not a stability estimate).
```
