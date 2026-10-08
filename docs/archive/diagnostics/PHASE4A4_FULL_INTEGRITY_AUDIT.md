# Phase 4A.4 — Full Evaluation + Code Integrity Audit

```
Audit decision:              AUDIT_PASS_WITH_MINOR_FIXES
Critical bugs:               0
High-severity bugs:          2   (both fixed; no scientific conclusion changed)
Metrics missing before audit: none canonical; Uncertain was computed but omitted
                              from the Phase 4A.2 / 4A.3 comparison tables
Metrics recovered without API: Distinct-2, Uncertain, failure types, rollout unique,
                              empty / malformed / question-mark (all recomputed locally)
Metrics requiring API retry:  Replay42 SeeQ (1 transient 502);
                              PureRFT44 support (1 ReadTimeout)
Scientific conclusions changed: NO
```

Scope: Final SFT; corrected pure UVFC+Violation-RFT seed42/43/44; UVFC+Violation-RFT
+ 1:1 SFT replay seed42/43/44. Raw predictions, evaluator outputs, manifests and
source code were treated as primary evidence; markdown summaries as derived.
Audit artifacts: `outputs/phase4a4_integrity_audit/`.

## 1. Pipeline map

```
VQAv2 train2014 questions ──► SFT 50k {image,question} ──► Final SFT (LoRA r16/a32/d0.05)
                                   │                              │
Phase 3A rollouts (2,000 train imgs × 4, UVFC + Flat)             │
        │                                                          │
   UVFC+Violation selection (admissible > UVFC desc > sha tie)     │
        │                                                          │
   selected_uvfc_violation (2,000) ───────────────► Pure RFT (seed 42/43/44, 250 steps)
        │                                                          │
   + 2,000 human replay (NFKC/casefold sha pick) ──► Replay RFT (seed 42/43/44, 250 steps)
        │                                                          │
        └────────────► development1000 greedy + sampled4x ─────────┘
                              │
   BLEU/CIDEr/Q-BLEU4-v/Distinct2/Dup  (pycocoevalcap + local)   ── local
   SeeQ rubric GPT-5.6-Luna (Search/Coverage/Context/Spatial)    ── API
   GPT-4o visual support (Supported/Unsupported/Uncertain + failures) ── API
   Qwen3-Embedding-4B within-image semantic diversity            ── local
                              │
                     paired image bootstrap (5,000, seed 42)
```

## 2. Data lineage (`data_lineage.json`)

| set | records | unique images | sha256 (prefix) |
|---|---:|---:|---|
| sft_train50k | 50,000 | 9,184 | 17b7892d |
| phase3a_pool_uvfc | 8,000 | 2,000 | — |
| rft_selected_train | 2,000 | 2,000 | dd559d3d |
| replay_human | 2,000 | 2,000 | e632cdc9 |
| mixed_1to1 | 4,000 | 2,000 | df537d10 |
| development1000 | 1,000 | 1,000 | — |
| reward_calibration | 4,000 | 1,000 | — |
| final_holdout | 2,000 | 2,000 | — |

Replay identity: deterministic `SHA256(image_id + "\t" + normalized_question)`,
`replay_same_as_uvfc_count = 0`; replay questions come only from
`src/vvqg/data/sft/train.jsonl` (verified 2,000/2,000 images present).

## 3. Split leakage matrix (`leakage_matrix.json`)

Image-ID overlap (rows = sets):

| | sft50k | 3A pool | RFT | replay | dev1000 | rewardcal | holdout |
|---|---:|---:|---:|---:|---:|---:|---:|
| RFT selected | 2000 | 2000 | 2000 | 2000 | **0** | **0** | **0** |
| replay | 2000 | 2000 | 2000 | 2000 | **0** | **0** | **0** |
| dev1000 | 0 | 0 | 0 | 0 | 1000 | 1000 | 0 |
| holdout | 0 | 0 | 0 | 0 | 0 | 0 | 2000 |

`development1000` and `reward_calibration` are the same 1,000 val2014 images
(overlap 1000) — both are evaluation manifests, not training data. Textual
overlap: 88/2,000 RFT questions and 65/2,000 replay questions coincide with
development reference questions (disclosed VQAv2 template overlap; not image
leakage). `final_holdout` questions were not read.

## 4. Initialization verification (`initialization_audit.json`)

All six trained checkpoints (`PureRFT42/43/44`, `Replay42/43/44`) report
`init_adapter = outputs/checkpoints/qwen3vl8b_lora_r16_vqav2_50k/checkpoint-final`,
loaded with `PeftModel.from_pretrained(..., is_trainable=True)` (no
`get_peft_model`, no new random adapter, no adapter stacking). Every adapter:
`r=16`, `lora_alpha=32`, `lora_dropout=0.05`, 7 target modules, 504 adapter
tensors, base `/hpc2hdd/home/fye374/models/Qwen3-VL-8B-Instruct`. Manifests/smoke
reports record `base_params_trainable = 0`, `lora_trainable = true`,
`trainable_parameters = 43,646,976` (identical across seeds).

## 5. Candidate / UVFC / Flat alignment (`candidate_alignment_audit.json`)

Recomputed the UVFC+Violation winner from the raw Phase 3A rollouts using the
production `select()` rule (admissible partition → descending UVFC →
`sha256("42:<image>#<gen>")`):

- images 2,000; every group cardinality = 4 (8,000 candidates);
- selection question match **2000/2000 (100%)**;
- UVFC score match 2000/2000; Flat state match 2000/2000; generation_id match 2000/2000;
- merge key = stable `image + generation_id` (never row position).

## 6. Metric completeness (`metric_completeness.json`, `final_metric_matrix.json`)

Resolved statuses:

- `Distinct-2`: **COMPUTED_AND_REPORTED** (corpus-level, canonical
  `diversity.py`), recomputed exactly for all 7 systems.
- `Uncertain`: **COMPUTED_NOT_REPORTED** — present in every
  `visual_support.json` (SFT 0.116, Pure42 0.123, Pure43 0.118, Pure44 0.134,
  Replay42 0.119, Replay43 0.135, Replay44 0.134) but omitted from the Phase 4A.2
  / 4A.3 comparison tables. Now surfaced; no evaluator rerun needed.
- Failure-type categories: COMPUTED_AND_REPORTED.
- Rollout unique rate, SemDiv, empty/malformed/question-mark: COMPUTED.
- Sampled BLEU/CIDEr/Q-BLEU: **never part of the frozen protocol**
  (OPTIONAL_DIAGNOSTIC, intentionally not computed).

## 7. Metric recomputation (`metric_recomputation.json`)

BLEU-4, CIDEr (pycocoevalcap), Q-BLEU4-variant, Distinct-2, Dup, unique,
empty/malformed/question-mark were independently recomputed from frozen greedy
predictions against `val2014_sample1000.jsonl`. **Zero mismatches > 1e-6** for
all 7 systems. Diversity means recomputed from per-image files equal the reported
agency values exactly (`diversity_audit.json`).

## 8. Evaluator cache integrity (`seeq_cache_audit.json`, `support_cache_audit.json`, `cache_identity_provenance.json`)

- SeeQ: 1,000 rows, **1,000 ok**, 0 errors, 1,000 unique image ids, 0 duplicate
  (image, question) pairs, all scores are integer 0–5 with the 4 required dims,
  and no error row carries a score. (Replay42 repaired 999 → 1,000.)
- Support: 1,000 rows, **1,000 ok**, 0 errors, `Supported+Unsupported+Uncertain
  = 1.0` for every system, all `failure_type` and `overall_visual_support`
  labels valid. (PureRFT44 repaired 999 → 1,000.)
- Cache-identity provenance: 50/50 sampled predictions per system reproduce the
  exact SeeQ `cache_key` and GPT-4o support `cache_identity` from
  (version, model, prompt, image_id, image hash, question) — no cross-model or
  stale-prediction contamination.
- Failed requests are never scored as 0: errors are stored without `scores` and
  excluded from the denominator; SeeQ now supports `retry_failures` (was the
  confirmed HIGH bug).

## 9. Generation / sampled4 integrity (`generation_audit.json`)

All 7 systems: greedy 1,000 rows with 1,000 unique image IDs **exactly equal** to
the frozen development1000 list; sampled4x 4,000 rows = 1,000 image groups ×
exactly `{0,1,2,3}`, IDs equal to development1000. Decoding config is exactly
`temperature=0.8, top_p=0.9, max_new_tokens=32`. Empty 0, malformed 0,
question-mark rate 1.0 for every system.

Image-question pairing (`image_question_pairing_audit.json`): 50 dev + 50 train
paths verified on disk with COCO id ↔ zero-padded filename match; every system's
prediction `image` field matches the frozen `image_id → image_path` map (0
mismatches). Image-conditioning (`image_conditioning_audit.json`): correct-image
loss 0.882 vs blank 2.113 vs grid-matched mismatched 2.774 (16 pairs) — the
pipeline is strongly image-sensitive.

## 10. Bootstrap correctness (`bootstrap_audit.json`)

`paired_bootstrap_diff` resamples **paired image indices** and averages per-pair
differences (callers sort shared `image_id`s before alignment). Reimplementing
the canonical algorithm reproduced the stored Replay42−SFT Search result exactly
(`+0.0468 [0.0356, 0.0584]`, n=1000, 5,000 resamples, seed 42). Rotating one side
to break alignment leaves the point estimate unchanged but changes the CI
(`[0.0328, 0.0608]`), confirming CIs depend on correct pairing. `n` equals the
aligned image count.

## 11. Code bugs and 12. patches (`bugs.json`, `patches.json`)

| id | sev | area | finding | fix |
|---|---|---|---|---|
| BUG-1 | HIGH | SeeQ evaluator | cached `status="error"` rows were terminal; no `--retry-failures` | added `retry_failures` param + flag |
| BUG-2 | HIGH | evaluation coverage | n<1000 coverage entered means silently (Replay42 SeeQ 999; Pure44 support 999) | retried the 2 confirmed failed IDs; regenerated summaries |
| BUG-3 | MEDIUM | reporting | Uncertain omitted from comparison tables | surfaced in audit + docs |
| BUG-4 | LOW | normalization | metric `strip().lower()` vs NFKC+casefold | documented, unchanged (frozen definitions) |
| BUG-5 | LOW | script robustness | smoke_report wrote to hard-coded path; relative-config manifest crash | fixed (`output_dir`, resolve path, `--manifest-only`, grad check) |

No CRITICAL bug: no train/dev leakage, correct Final-SFT init, image-only
generation input (no reference contamination), no answer leakage
(`{image, question}` only), no image/question mispairing, 100% candidate
alignment, and no unacceptably incomplete aggregation remains.

Regression tests added (`tests/test_phase4a4_audit.py`): selection alignment,
replay/selected image-id equality, sampled4 four-per-image, paired bootstrap,
support class sums, SeeQ failed-ID retry (with/without flag), holdout
disjointness. All targeted tests pass (`34 passed`).

## 13. Reference / answer / holdout / normalization audits

- Reference contamination: generation input is image + instruction only; no
  reference/answer tokens in `inference.py`, `generator.py`, or the generation
  CLI (`reference_contamination_audit.json`).
- Answer leakage: training datasets contain only `{image, question}` (+`source`
  tag); no answer/label/reward field (`answer_leakage_audit.json`).
- `final_holdout`: only protection/leakage-audit mentions exist; no holdout
  prediction or scoring artifact; `final_holdout_touched = false`
  (`final_holdout_audit.json`).
- Normalization: two families (`strip().lower()` metric-internal;
  NFKC+casefold report/identity). Documented; comparisons unaffected
  (`normalization_audit.json`).

## 14. Corrected metric matrix

See `final_metric_matrix.json`; all canonical metrics for all seven systems are
**✓ validated** (no M/R/X). Corrected local values used by the comparisons:

| Metric | SFT | Pure42 | Pure43 | Pure44 | Replay42 | Replay43 | Replay44 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BLEU-4 | 0.3172 | 0.2781 | 0.2676 | 0.2671 | 0.3164 | 0.2984 | 0.3088 |
| CIDEr | 0.4550 | 0.4729 | 0.4443 | 0.4320 | 0.4847 | 0.4744 | 0.4807 |
| Q-BLEU4-v | 0.3107 | 0.2705 | 0.2609 | 0.2602 | 0.3103 | 0.2918 | 0.3031 |
| Search | 0.2746 | 0.3620 | 0.3580 | 0.3434 | 0.3214 | 0.3114 | 0.3056 |
| Coverage | 0.2598 | 0.3804 | 0.3654 | 0.3606 | 0.3234 | 0.3192 | 0.3126 |
| Context | 0.2704 | 0.2648 | 0.2632 | 0.2796 | 0.2562 | 0.2776 | 0.2698 |
| Spatial | 0.0732 | 0.0564 | 0.0610 | 0.0582 | 0.0684 | 0.0722 | 0.0704 |
| Supported | 0.725 | 0.821 | 0.817 | 0.792 | 0.786 | 0.755 | 0.747 |
| Unsupported | 0.159 | 0.056 | 0.065 | 0.074 | 0.095 | 0.110 | 0.119 |
| Uncertain | 0.116 | 0.123 | 0.118 | 0.134 | 0.119 | 0.135 | 0.134 |
| Distinct-2 | 0.1395 | 0.2181 | 0.2182 | 0.2245 | 0.1673 | 0.1826 | 0.1708 |
| Dup | 0.570 | 0.397 | 0.379 | 0.359 | 0.487 | 0.455 | 0.489 |
| Rollout unique | 0.9555 | 0.8040 | 0.8088 | 0.8133 | 0.9248 | 0.9263 | 0.9253 |
| SemDiv | 0.5360 | 0.4427 | 0.4444 | 0.4588 | 0.5085 | 0.5136 | 0.5057 |

## Conclusions that must be revised

None. The Phase 4A.2 decision **REPLAY_PARETO_IMPROVES** and the Phase 4A.3
decision **REPLAY_ROBUST_WITH_RESIDUAL_TRADEOFF** are unchanged. The only
revisions are (a) Replay42 SeeQ 999 → 1,000 and PureRFT44 support 999 → 1,000
cover, with metric changes of at most 6×10⁻⁴, and (b) Uncertain now reported.
No model was retrained; no method, UVFC, Flat, replay ratio, or `final_holdout`
was touched.
