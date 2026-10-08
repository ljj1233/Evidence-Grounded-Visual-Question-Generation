# UVFC+Violation-RFT Multi-Seed Confirmation (Phase 4A.1)

Diagnostic confirmation only. Only the training seed changes across seeds 42/43/44
(Python/NumPy/torch CPU+CUDA/DataLoader shuffle); everything else is frozen: the
same selected dataset (SHA256 `dd559d3d...`), selection rule, Final SFT starting
checkpoint, LoRA (r16/a32/d0.05), optimizer/lr/scheduler (AdamW 2e-4, cosine, 3%
warmup), 2 epochs / 250 steps, batch 2 × accum 8, decoding, and evaluators. No
SFT replay, no method change, no best-seed selection, no checkpoint averaging,
`final_holdout` untouched.

All three runs independently initialized from
`outputs/checkpoints/qwen3vl8b_lora_r16_vqav2_50k/checkpoint-final` (verified in
each `training_summary.json`: `init_adapter` = SFT), 2,000 records, zero overlap
with development1000 / val_debug / final_holdout, base frozen with 43,646,976
trainable LoRA parameters.

## Three-seed table (development1000, greedy)

| Metric | SFT | seed42 | seed43 | seed44 | Mean Δ vs SFT | Std Δ |
|---|---:|---:|---:|---:|---:|---:|
| BLEU-4 | 0.3172 | 0.2781 | 0.2676 | 0.2671 | −0.0463 | 0.0051 |
| CIDEr | 0.4550 | 0.4729 | 0.4443 | 0.4320 | −0.0052 | 0.0171 |
| Q-BLEU4-variant | 0.3107 | 0.2705 | 0.2609 | 0.2602 | −0.0469 | 0.0047 |
| Search | 0.2746 | 0.3620 | 0.3580 | 0.3434 | **+0.0799** | 0.0080 |
| Coverage | 0.2598 | 0.3804 | 0.3654 | 0.3606 | **+0.1090** | 0.0084 |
| Context | 0.2704 | 0.2648 | 0.2632 | 0.2796 | −0.0012 | 0.0074 |
| Spatial | 0.0732 | 0.0564 | 0.0610 | 0.0582 | −0.0147 | 0.0019 |
| Supported | 0.725 | 0.821 | 0.817 | 0.7918 | **+0.0849** | 0.0129 |
| Unsupported | 0.159 | 0.056 | 0.065 | 0.0741 | **−0.0940** | 0.0074 |
| Dup | 0.570 | 0.397 | 0.379 | 0.359 | −0.1917 | 0.0155 |
| SemDiv | 0.5360 | 0.4427 | 0.4444 | 0.4588 | −0.0874 | 0.0073 |

`positive_seed_count` (direction > 0): Search 3/3, Coverage 3/3, Supported 3/3,
Unique 3/3, CIDEr 1/3, Context 1/3, Spatial 0/3, BLEU 0/3, SemDiv 0/3;
Unsupported 0/3 positive (i.e. **all three reduce it**). BLEU ratio vs SFT:
0.877 / 0.843 / 0.842.

Paired bootstrap vs SFT (5,000, seed 42), estimate [95% CI]:

| Seed | Search | Coverage | Supported | Unsupported |
|---|---:|---:|---:|---:|
| 42 | +0.0874 [0.0736, 0.1008] | +0.1206 [0.1048, 0.1360] | +0.096 [0.064, 0.129] | −0.103 [−0.130, −0.077] |
| 43 | +0.0834 [0.0706, 0.0964] | +0.1056 [0.0912, 0.1208] | +0.092 [0.059, 0.125] | −0.094 [−0.121, −0.067] |
| 44 | +0.0688 [0.0566, 0.0814] | +0.1008 [0.0864, 0.1158] | +0.066 [0.032, 0.098] | −0.085 [−0.110, −0.059] |

Context per seed (CI includes 0 in all): 42 −0.0056 [−0.0146, 0.0030]; 43 −0.0072
[−0.0158, 0.0012]; 44 +0.0092 [0.0004, 0.0180]. Spatial all negative and
CI-separated: 42 −0.0168, 43 −0.0122, 44 −0.0150.

## Robust gains

- **Search** improves in 3/3 seeds (+0.069 to +0.087), CIs exclude zero.
- **Coverage** improves in 3/3 seeds (+0.101 to +0.121), CIs exclude zero.
- **Supported** improves in 3/3 seeds (+0.066 to +0.096), CIs exclude zero.
- **Unsupported** falls in 3/3 seeds (−0.085 to −0.103), CIs exclude zero.
- **Duplication** falls and **unique rate** rises in 3/3 seeds (+0.16 to +0.21).
- **CIDEr is preserved** (0.95–1.04× SFT; mean −0.005).

## Robust tradeoffs

- **BLEU-4 degrades consistently** (0.877 / 0.843 / 0.842 × SFT; std 0.005).
  Two seeds fall below the 0.90× band.
- **Within-image semantic diversity decreases consistently** (0.443 / 0.444 /
  0.459 vs 0.536; 0/3 positive).
- **Spatial decreases consistently and tightly** (−0.012 to −0.017; std 0.0019;
  0/3 positive) — the same optimization-side shift characterized in Phase 3A.3.
- **Context is not robustly changed** (1/3 positive; CIs include 0) — seed noise.

## Primary questions

1. Do all 3 seeds improve Search? **Yes.**
2. Do all 3 seeds improve Coverage? **Yes.**
3. Do all 3 seeds improve Supported? **Yes.**
4. Do all 3 seeds reduce Unsupported? **Yes.**
5. Is Context decline stable? **No** — seed-dependent and not significant.
6. Is Spatial decline stable? **Yes** — consistent, tightly clustered, negative.
7. Is BLEU degradation stable? **Yes** (0.84–0.88× SFT, low variance).
8. Is CIDEr preserved? **Yes** (~flat; one seed above SFT).
9. Is semantic diversity degradation stable? **Yes** (all three lower).

## Decision

`outputs/phase4a_uvfc_rft_multiseed/comparison/decision.json`:

```json
{"decision": "RFT_ROBUST_WITH_TRADEOFF", "final_holdout_touched": false,
 "sft_replay_added": false, "best_seed_selected": false, "checkpoints_averaged": false}
```

The visual-exploration and support gains of UVFC+Violation-RFT are robust across
seeds; so are the language (BLEU), semantic-diversity and Spatial costs. This is
the informative outcome the phase anticipated: **selection + SFT is a reliable
lever on exploration and support, but it is not free** — it consistently trades
language fidelity and within-image diversity. No method adjustment was made here;
any mitigation (e.g. replay/regularization) is deferred to a future phase.
