# Phase 4A — UVFC Rejection Sampling Fine-Tuning (RFT)

Tests whether the offline UVFC selection signal improves VQG through data
selection + SFT, without policy optimization. Not GRPO. No UVFC/Flat/reward
changes, no online rollout, no `final_holdout`.

> **Correction (Phase 4A.1):** the first Phase 4A run used `train_sft.py`, which
> builds a **fresh random LoRA** via `get_peft_model` and does not load an
> adapter, so it did *not* start from the Final SFT checkpoint as specified. The
> trainer now supports `sft_checkpoint` (`PeftModel.from_pretrained(...,
> is_trainable=True)`) and **all Phase 4A results below were re-run from the Final
> SFT adapter** (seed 42, 2 epochs). Numbers here supersede the earlier draft.

## Candidate-source correction

The spec named `outputs/reward_calibration/sft_candidates_4x.jsonl` and
`development1000`; these are the **same 1,000 images** (overlap 1000/1000). Using
that pool would leak. With approval the candidate source is the clean, frozen,
image-disjoint Phase 3A train-only pool:

```
outputs/phase3a_grpo_pilot/preflight/{rollouts_uvfc,rollouts_flat}.jsonl
2,000 train2014 images x 4; UVFC + Flat; 0 overlap with dev1000 / val_debug / final_holdout
```

## Selection audit

| Selection | images | mean UVFC | random-from-4 | gain | confirmed violations |
|---|---:|---:|---:|---:|---:|
| UVFC-only | 2,000 | 0.848 | 0.504 | +0.344 | 747 (37%) |
| UVFC+Violation | 2,000 | 0.717 | 0.504 | +0.213 | 54 (2.7%) |

Dataset SHA256 (violation) `dd559d3d...`; identical for seeds 42/43/44. No
independent GPT-4o/Luna labels exist for the train2014 pool (documented, not
guessed).

## Training

Final SFT adapter + LoRA (r16/a32/d0.05), 2 epochs, lr 2e-4, batch 2 × accum 8
(250 steps), seed 42; the selected 2,000 pairs only. `uvfc_only` loss 2.06→0.37;
`uvfc_violation` 1.94→0.34 (both clearly lower than the earlier scratch-LoRA run,
consistent with starting from SFT).

## Comparison (development1000, greedy)

| Model | Search | Coverage | Context | Spatial | Supported | Unsupported | BLEU | CIDEr | Distinct-2 | Dup | SemDiv |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT | 0.2746 | 0.2598 | 0.2704 | 0.0732 | 0.725 | 0.159 | 0.3172 | 0.4550 | 0.1395 | 0.570 | 0.5360 |
| **UVFC-RFT** | **0.3948** | **0.4112** | 0.2862 | 0.0570 | 0.752 | 0.105 | 0.2283 | 0.3875 | 0.2248 | 0.422 | 0.4190 |
| **UVFC+Violation-RFT** | **0.3620** | **0.3804** | 0.2648 | 0.0564 | **0.821** | **0.056** | 0.2781 | 0.4729 | 0.2181 | 0.397 | 0.4427 |

Paired bootstrap vs SFT (5,000, seed 42):

| Metric | UVFC-RFT − SFT | UVFC+Violation-RFT − SFT |
|---|---:|---:|
| Search | +0.1202 [0.1056, 0.1350] | +0.0874 [0.0736, 0.1008] |
| Coverage | +0.1514 [0.1352, 0.1678] | +0.1206 [0.1048, 0.1360] |
| Supported | +0.027 [−0.011, 0.066] | +0.096 [0.064, 0.129] |
| Unsupported | −0.054 [−0.085, −0.024] | −0.103 [−0.130, −0.077] |

## Interpretation

- **Data selection alone reproduces and roughly doubles the GRPO gain.** From-SFT
  UVFC-RFT Search +0.120 / Coverage +0.151 is ~3.0× / ~2.4× the Phase 3A.4
  P1strong GRPO effect (+0.040 / +0.062). Answer to the phase's question: **(A)
  better selected training data** is sufficient to move these axes.
- **The violation constraint matters for support.** UVFC+Violation-RFT gives the
  best Supported (0.821) / Unsupported (0.056) with preserved CIDEr (1.04× SFT),
  but BLEU still falls to 0.88× SFT. UVFC-only drops BLEU to 0.72× SFT and lowers
  within-image semantic diversity — a real language/quality tradeoff.
- **Spatial**: both variants decline (−0.017 / −0.011), consistent with Phase 3A.3.

## Decision

`outputs/phase4a_uvfc_rft/decision.json`: **RFT_POSITIVE** (Search and Coverage
improve with CIs excluding zero; no severe support degradation). Recommended
variant: UVFC+Violation-RFT. See `docs/UVFC_VIOLATION_RFT_MULTI_SEED.md` for the
3-seed robustness and the tradeoff characterization.

## Caveats

- Candidate pool is the Phase 3A train-only pool (leakage fix), 2,000 train2014
  images.
- SeeQ uses the documented `gpt-5.6-luna` substitution; support uses GPT-4o.
- BLEU 0.72× (UVFC-only) and 0.88× (violation) are language-degeneration flags;
  the violation variant is much milder.
