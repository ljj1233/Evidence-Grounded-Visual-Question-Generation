# Confirmation report — frozen Full Ours + Offline DPO Refinement (step_00207)

Frozen checkpoint: `reports/offline_dpo_unified_refinement_20261008/dpo/checkpoints/step_00207`
(adapter sha256 `32384eea…`). The checkpoint was **not modified**. DEV256 and DEV744
are reported separately; the pooled DEV1000 number is descriptive only.

## DEV256 (model-selection subset, n=256) — external (vs Base, paired bootstrap n=10000 seed42)

Source: `PHASE1/STAGE1EXT.json`.

| metric | Base | step_00207 | delta | 95% CI |
|---|---:|---:|---:|---|
| GPT Supported | 0.844 | 0.926 | +0.082 | [+0.043, +0.125] |
| GPT Unsupported | 0.059 | 0.031 | −0.027 | [−0.055, +0.004] |
| SeeQ Coverage | 0.372 | 0.425 | +0.053 | [+0.026, +0.081] |
| Search | 0.355 | 0.404 | +0.049 | [+0.025, +0.074] |
| Context | 0.275 | 0.225 | −0.051 | [−0.064, −0.038] |
| Spatial | 0.056 | 0.074 | +0.017 | [−0.000, +0.035] |

Local (`PHASE1/TRAJECTORY_TABLE.csv`): Flat 0.749→0.788; BLEU-4 0.234→0.223;
CIDEr 0.433→0.428; Q-BLEU4 0.226→0.217; what 0.254→0.246; count+yes/no 0.668→0.606;
type H 1.836→1.818; len 5.89→6.27; dup 0.188→0.234; **JS vs Base 0.046**.
K=4 norm-unique 0.804→0.704, all-same-type 0.199→0.402.

## DEV744_SECONDARY_CONFIRMATION (n=744)

`DEV744_MANIFEST.jsonl` sha `a6db294a…`. External (`DEV744_EXT.json`):

| metric | Base | step_00207 | delta | 95% CI |
|---|---:|---:|---:|---|
| GPT Supported | 0.847 | 0.931 | +0.085 | [+0.062, +0.109] |
| GPT Unsupported | 0.039 | 0.026 | −0.013 | [−0.026, −0.001] |
| SeeQ Coverage | 0.373 | 0.424 | +0.051 | [+0.036, +0.067] |
| Search | 0.347 | 0.410 | +0.063 | [+0.050, +0.077] |
| Context | 0.277 | 0.229 | −0.047 | [−0.055, −0.040] |
| Spatial | 0.064 | 0.064 | +0.000 | [−0.010, +0.009] |

Local: Flat 0.746→0.789; BLEU-4 0.268→0.256; CIDEr 0.413→0.408; Q-BLEU4 0.257→0.245;
what 0.204→0.203; count+yes/no 0.729→0.660; type H 1.768→1.781; len 5.83→6.38;
dup 0.327→0.354; **JS vs Base 0.055**.

## Pooled DEV1000 (development-only, descriptive; n=1000 support / 995 SeeQ)

| metric | Base | step_00207 | delta | 95% CI |
|---|---:|---:|---:|---|
| GPT Supported | 0.846 | 0.930 | +0.084 | [+0.064, +0.105] |
| GPT Unsupported | 0.044 | 0.027 | −0.017 | [−0.029, −0.005] |
| SeeQ Coverage | 0.373 | 0.424 | +0.052 | [+0.038, +0.065] |
| Search | 0.349 | 0.409 | +0.060 | [+0.048, +0.071] |
| Context | 0.276 | 0.228 | −0.048 | [−0.055, −0.041] |
| Spatial | 0.062 | 0.067 | +0.004 | [−0.004, +0.013] |

## Conclusion

The frozen DPO refinement **reproduces** on the secondary confirmation set: GPT
Supported **+0.085** (CI excludes 0), Coverage **+0.051** and Search **+0.063**
(CI exclude 0), Unsupported reduced, with a Context cost of −0.047 (smaller than the
bs8-original's −0.064) and a near-Base type distribution (JS 0.046/0.055). DEV256 and
DEV744 do not pool to hide heterogeneity — their per-set deltas agree closely
(Supported +0.082 vs +0.085), and both are reported.

No checkpoint change was made; no further seeds were run. `final_holdout` remains sealed.

API this stage (development evaluation only): SeeQ 335 + GPT-4o support 335 = 670 new
requests. Training-label API: 0.
