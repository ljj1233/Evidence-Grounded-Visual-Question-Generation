# E2 soft-evidence correspondence & decoder-use diagnostic

Run: `reports/e2_slot_use_diagnostic_20261005T022035Z/`
seed 42 · GPU 0 · git `9dea27f0` · read-only (no training/backward/evaluator/final_holdout)

## Setup

- C0: `outputs/ec_qse_aligned_direct4/full_v2/c0/checkpoint-final`
- E2: `outputs/ec_qse_aligned_direct4/full_v2/e2/checkpoint-final`
- 256 dev images, fixed manifest sha256 `809e9e0f…`, train overlap 0.
- slot stats: off-diagonal cosine **−0.290** (median −0.293) — slots are numerically distinct.
- 962 dev candidate questions, 803 unique; candidate source = A0 DIRECT4 dev pool.

## Q1 — do the slots carry image correspondence? **FAIL**

Retrieval of a question to its own image's slots among 256 images (max-over-slots):

| aggregation | Recall@1 | Recall@5 | MRR |
|---|---:|---:|---:|
| max | 0.0052 | 0.0239 | 0.0256 |
| mean | 0.0021 | 0.0146 | 0.0203 |
| top-2 | 0.0042 | 0.0166 | 0.0234 |
| chance | 0.0039 | 0.0195 | ~0.0156 |

Correct-image score vs controls (paired per question):

| control | absolute mean score | paired delta vs correct | 95% CI | positive frac |
|---|---:|---:|---|---:|
| **correct** | 0.0678 | — | [0.0658, 0.0696] | 1.0 |
| wrong image | 0.0679 | +0.00008 | [−0.00105, +0.00115] | 0.499 |
| mean slots | 0.0140 | −0.0538 | [−0.0572, −0.0504] | 0.182 |
| random slots | 0.0493 | −0.0185 | [−0.0213, −0.0157] | 0.338 |

- `margin_mean` (correct − mean(wrong)): **−0.0001**, CI [−0.0008, +0.0007] → crosses 0.
- `margin_max` (correct − max(wrong)): **−0.0394**, CI [−0.0406, −0.0381] → **negative**.
- image-level paired margin: −0.0003, CI [−0.0016, +0.0009] → crosses 0.
- Hungarian matched−shuffled: +0.0575 (but Hungarian is optimised on the same
  similarity matrix, so not independent evidence).
- slot-permutation test: score delta **0.000** (max/mean/top2 are
  permutation-invariant) → no slot-index bias.
- Correct is essentially indistinguishable from a wrong image and far above
  chance is absent. Root cause: the frozen question embeddings are themselves
  near-collapsed (mean off-diagonal cosine **0.992** among different questions),
  so max-cos to any image is dominated by a shared component.

**Q1 = FAIL** (no image-specific correspondence; result is not independent of
the anisotropic question embeddings).

## Q2 — does the decoder use the slots? **FAIL**

Gate ≈ **0.00975**, `out_proj` weight norm **4.59**. Layer-wise evidence
residual / hidden-norm ratio over all 36 decoder layers: mean **4.5e-4**
(range 3.8e-5 … 1.1e-3). The evidence contribution is ~0.04% of the hidden
state everywhere — the branch is effectively off.

Fixed-question (q_normal) per-token NLL:

| condition | NLL (per token) |
|---|---:|
| E2-normal (correct) | 0.28688 |
| E2-off | 0.28691 |
| E2-mean | 0.28709 |
| E2-wrong image | 0.28637 |
| E2-random | 0.28706 |
| C0 (control) | 0.29539 |

Paired NLL delta vs E2-normal (positive = worse without correct slots):

| delta | mean | 95% CI | pos frac |
|---|---:|---|---:|
| off − normal | +0.00003 | [−0.0016, +0.0018] | 0.47 |
| mean − normal | +0.00021 | [−0.0015, +0.0019] | 0.49 |
| wrong − normal | −0.00051 | [−0.0021, +0.0012] | 0.47 |
| random − normal | +0.00018 | [−0.0015, +0.0018] | 0.47 |
| C0 − normal | +0.00851 | [−0.0064, +0.0246] | 0.55 |

All decode-condition NLL differences are within noise (CIs cross 0). Removing or
corrupting the slots changes the teacher-forced NLL by ≈0.

Generation: 195/256 images emit the **identical** question under all five
conditions; only 13.7% differ between normal and off, and those differences also
appear for wrong/mean/random. Empty output 0; distinct ratio 0.78–0.80 across
conditions.

**Q2 = FAIL** (decoder does not rely on the evidence slots; residual ratio ≈0).

## Q3 — is the slot division useful for generation? **FAIL**

correct_slots is not better than mean_slots/wrong_image_slots/random_slots on
either retrieval or fixed-question NLL; the low slot cosine is the only thing
established.

## Conclusion

**v2 removes numerical slot collapse but does not establish image-grounded slot
semantics or decoder reliance.**

## Frozen question encoder audit

- LM frozen, projection frozen, eval mode, no dropout, no LoRA, no
  `requires_grad` parameters, text-only (no image input), projection sha256
  `60ebf490…`, repeat-consistent and padding-invariant.
- **But** different questions have mean off-diagonal cosine **0.992** (anisotropy),
  which caps retrieval/assignment.
- It was created fresh (base model, seed 42) in the packet builder and never
  trained; no LoRA parameters exist in it. It is genuinely frozen — but its
  representation is near-degenerate.

## Edit-distance cases

`cases_normal_off.jsonl`, `cases_normal_wrong.jsonl`, `cases_normal_mean.jsonl`,
`cases_normal_random.jsonl`, `cases_c0_e2.jsonl`, `cases_no_difference.jsonl`
(195 no-difference images). See `FINAL_REPORT.md` for 20-row samples.
