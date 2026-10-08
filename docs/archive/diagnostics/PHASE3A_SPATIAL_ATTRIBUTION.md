# Phase 3A.3 — UVFC Spatial Attribution Audit

Analysis only. No retraining, no UVFC/Flat/reward changes, no GRPO, `final_holdout`
untouched. All inputs are frozen artifacts on `development1000` or the Phase 2.6
candidate pool. Only missing field: P1/P2 GPT-4o question-type labels (SFT/P1strong
reused from the P1strong eval cache).

## 1. Is Spatial degradation statistically meaningful?

P1strong − SFT per image (normalized 0–1), one-sample paired bootstrap 5,000, seed 42:

| Metric | Δ | 95% CI |
|---|---:|---|
| Search | +0.0454 | [0.0344, 0.0560] |
| Coverage | +0.0658 | [0.0530, 0.0788] |
| Context | +0.0150 | [0.0068, 0.0234] |
| **Spatial** | **−0.0110** | **[−0.0202, −0.0016]** |

Per-image Spatial categories: improved 107 (10.7%), same 753 (75.3%), degraded 140
(14.0%). Degraded images are net-negative across Search/Coverage too, but the
aggregate Search/Coverage gains dominate. The Spatial decline is **statistically
meaningful** (CI excludes 0) but small (−0.011 normalized = −0.055 raw).

## 2. Raw Spatial distribution (0–5)

| System | mean | median | std | counts (0/1/2/3/4/5) |
|---|---:|---:|---:|---|
| Plain SFT | 0.366 | 0.0 | 0.707 | 767/101/131/1/0/0 |
| P1strong | 0.311 | 0.0 | 0.668 | 800/94/101/5/0/0 |

Transition matrix (rows = SFT, cols = P1strong):

```
       0    1    2    3   4   5
0 [ 678,  45,  43,   1,  0,  0]
1 [  55,  31,  15,   0,  0,  0]
2 [  67,  18,  43,   3,  0,  0]
3 [   0,   0,   0,   1,  0,  0]
```

The movement is bidirectional: 88 SFT-0 → higher, 55 SFT-1 → 0, 85 SFT-2 → lower
but 46 SFT-2 → higher. The net shift is a modest increase in zeros (767 → 800),
not a collapse of a well-populated moderate band. Spatial scores are sparse-by-
construction (most questions score 0), so single-step judge noise matters.

## 3. Question-type shift (GPT-4o, `phase25b` taxonomy)

| type | SFT | P1 | P2 | P1strong | Δ(P1strong−SFT) |
|---|---:|---:|---:|---:|---:|
| object | 0.231 | 0.248 | 0.257 | 0.185 | −0.046 |
| attribute | 0.073 | 0.062 | 0.058 | 0.044 | −0.029 |
| color | 0.145 | 0.133 | 0.186 | 0.091 | −0.054 |
| count | 0.015 | 0.043 | 0.030 | 0.110 | +0.095 |
| yes/no | 0.047 | 0.170 | 0.064 | 0.232 | +0.185 |
| action | 0.299 | 0.148 | 0.220 | 0.131 | −0.168 |
| relation | 0.001 | 0.008 | 0.008 | 0.000 | −0.001 |
| spatial | 0.070 | 0.059 | 0.061 | 0.045 | −0.025 |
| scene/context | 0.045 | 0.062 | 0.040 | 0.082 | +0.037 |

Top-3 increases: **yes/no +0.185, count +0.095, scene/context +0.037**.
Top-3 decreases: **action −0.168, color −0.054, object −0.046**. Spatial-type
rate also falls (−0.025). The policy trades open action/object/color questions for
grounded yes/no and counting questions; spatial phrasing is a minor casualty.

## 4. Offline UVFC spatial alignment (Phase 2.6 pool)

Per image, rank the four SFT candidates by UVFC; compare top vs bottom (Luna,
normalized, paired bootstrap 5,000, seed 42):

| Metric | UVFC top − bottom | 95% CI |
|---|---:|---|
| Search | +0.1132 | [0.0988, 0.1274] |
| Coverage | +0.1278 | [0.1108, 0.1442] |
| Context | +0.0266 | [0.0162, 0.0374] |
| **Spatial** | **+0.0184** | **[0.0056, 0.0310]** |

The offline UVFC ranking **prefers higher-Spatial questions**. The reward
construct does not underweight spatial; the decline is not present at the offline
selector level.

## 5. UVFC feature-family attribution (generated questions)

No feature semantic labels exist in the frozen SAE; raw indices/contributions only
(`feature_attribution.json`). Mean active features per question: SFT 73.7 → P1 73.9
→ P2 74.1 → P1strong 74.4. P1strong vs SFT: mean per-question Jaccard 0.643,
top-100 feature overlap 82/100. The active-feature repertoire is broadly stable
with a moderate shift; no single feature family was identified (and none invented).

## 6. UVFC vs length

| System | mean tokens | p95 | mean UVFC | Pearson(UVFC,len) | Spearman |
|---|---:|---:|---:|---:|---:|
| Plain SFT | 6.554 | 9 | 0.218 | 0.359 | 0.210 |
| P1 | 6.567 | 9 | 0.333 | 0.196 | 0.128 |
| P2 | 6.610 | 9 | 0.245 | 0.352 | 0.249 |
| P1strong | 6.661 | 9 | 0.454 | 0.163 | 0.145 |

Length is essentially flat (+1.6%), and the UVFC–length correlation **drops** in
P1strong (0.359 → 0.163). The UVFC increase is not a length artifact.

## 7. Supported-only robustness

Restricted to the 615 images where both the SFT and P1strong questions are
independently GPT-4o-`supported`:

| Metric | SFT | P1strong | Δ [95% CI] |
|---|---:|---:|---:|
| Search | 0.2732 | 0.3210 | +0.0478 [0.0351, 0.0615] |
| Coverage | 0.2676 | 0.3210 | +0.0533 [0.0380, 0.0689] |
| Context | 0.2556 | 0.2631 | +0.0075 [−0.0003, 0.0153] |
| Spatial | 0.0790 | 0.0680 | −0.0111 [−0.0231, +0.0007] |

The Spatial decline persists on supported-only questions but becomes borderline
(CI marginally includes 0). It is therefore not an artifact of unsupported
questions alone, though support conditioning weakens it.

## 8. Decision

`outputs/phase3a_spatial_audit/decision.json`:

```json
{"decision": "OPTIMIZATION_ARTIFACT", "final_holdout_touched": false,
 "uvfc_modified": false, "grpo_started": false, "new_training": false}
```

Reasoning (per §5's explicit rule): the offline UVFC ranking does **not** hurt
Spatial (top−bottom = +0.018, CI positive), so the reward construct is not the
cause; the decline is introduced during policy optimization. The proximate
mechanism is the distributional shift toward yes/no and counting questions, with
a small reduction in spatial-type questions (−0.025), which the UVFC reward only
weakly penalizes. The decline is modest (−0.011 normalized), partly bidirectional
judge noise on a sparse 0–5 axis, and the same update raises Search/Coverage/
Context substantially.

This is an optimization/transfer side-effect, not a reward-construct defect, and
no reward change is authorized by this audit.
