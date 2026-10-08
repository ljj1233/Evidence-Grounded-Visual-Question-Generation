# SeeQ-style Self-Evolution Adaptation — Baseline Results

> **This experiment is a same-backbone SeeQ-style self-evolution adaptation, not an
> exact reproduction of SeeQ.**

Frozen naming: **SeeQ-style Self-Evolution Adaptation**. It must not be called
"SeeQ" or "SeeQ reproduction".

The method-faithfulness audit (`docs/SEEQ_FAITHFULNESS_AUDIT.md`,
`outputs/same_backbone_baselines/seeq_style/faithfulness_audit.json`) recommends
**Option C** (`can_call_SeeQ = false`, `can_call_reproduction = false`).

---

## 1. Pipeline as executed

```text
M0 = final VQAv2 SFT LoRA (qwen3vl8b_lora_r16_vqav2_50k/checkpoint-final)

R1:  M0 proposer | frozen M0 reference | propose -> answer -> rewrite -> filter
     canonical retention (image round-robin) -> 10,000 triples -> 20,000 QG/QA
     train M0 -> M1                          (round1_m1)

R2:  M1 proposer | frozen M0 reference | same stages
     canonical retention -> 10,000 triples -> 20,000 QG/QA
     train M1 -> M2                          (round2_m2)
```

Checkpoints (adapter weights SHA-256):

| Model | Path | SHA-256 |
|---|---|---|
| M0 (final SFT) | `outputs/checkpoints/qwen3vl8b_lora_r16_vqav2_50k/checkpoint-final` | `5fb66c36a2255fb3a9011db198fd55a8be8b9cbb26eef611727a816bc7934e0e` |
| M1 (SeeQ-style R1) | `outputs/same_backbone_baselines/seeq_style/round1_m1` | `55545b7fa416a8c3991db9d2c66c0cc4eddadae0fa2831414949e5c98bc0e135` |
| M2 (SeeQ-style R2) | `outputs/same_backbone_baselines/seeq_style/round2_m2` | `9d2f0055f5e5d6491a0336b957bcede7cc54ffa7f3c57a483c4115b00db28af2` |

Training (both rounds identical framework): LoRA r16/alpha32/dropout0.05, 1 epoch,
per-device batch 2, no accumulation, lr 2e-5, cosine, warmup_steps 1000,
gradient checkpointing, FA2, seed 42. M1: 10,000 steps, train_loss 0.4056,
runtime 5,830 s. M2: 10,000 steps, train_loss 0.1816, runtime 5,915 s.

### R2 provenance (this phase)

* Old M0-proposer R2 was **moved** to
  `round2_generation_invalid/` with
  `invalid_reason.json` (`INVALID_FOR_SEEQ_EVOLUTION`); it is not used for
  training or evaluation.
* New R2 generated with **M1 proposer + frozen M0 reference** into
  `round2_generation_m1/`; provenance in `generation_manifest.json`.
* Retention uses only the canonical `accepted_triples` / `_retain_round_robin`;
  `round2/` has exactly 10,000 retained and 20,000 formatted records, identical
  to `stage_e_retained.jsonl`.

---

## 2. R1 vs R2 generation statistics

| | R1 | R2 |
|---|---:|---:|
| proposals | 36,736 | 36,736 |
| accepted | 23,618 | 25,789 |
| acceptance rate | 0.643 | 0.702 |
| images covered (retained) | 6,572 | 6,924 |
| unique questions (retained) | 1,889 | 1,587 |
| unique (question, answer) | 4,649 | 4,272 |
| duplicate rate, (q,a) | 0.535 | 0.573 |

Rejection reasons are dominated by `hard_constraint_failed`
(R1 13,056; R2 10,889) with a handful of `missing_hard_constraints` and
`invalid_answerability`; no `invalid_judge`.

The M1 proposer raises the acceptance rate but reduces question variety
(fewer unique questions, higher duplicate rate) — the adapted loop drifts
toward a narrower question distribution (`round2/statistics.json`).

---

## 3. Evaluation protocol

* Images: frozen `development1000` (1,000 images; disjoint from train/calibration/holdout).
* References: `outputs/evaluation/sft_greedy/val2014_sample1000.jsonl` (same 1,000 images).
* Greedy: 1 question/image, `max_new_tokens=32`.
* Sampled: 4 questions/image, `temperature=0.8`, `top_p=0.9`, `max_new_tokens=32`.
* **Normalization (applied uniformly to M0/M1/M2):** the SeeQ-style adapters were
  trained on the `qg` target `question + " " + answer`, so they emit
  `question? answer` (100% of M1/M2 greedy rows). Predictions are truncated at
  the first `?` so all systems are scored as question generators. M0 output is
  unchanged by this step.
* `final_holdout` was **not** generated or evaluated.

Artifacts: `outputs/same_backbone_baselines/seeq_style/evaluation/`
(`summary.json`, predictions, per-metric dirs, judge outputs).

---

## 4. Results

### 4.1 Traditional greedy (1,000 images)

| Model | BLEU-4 | Q-BLEU-4 variant | CIDEr | Distinct-2 | dup rate |
|---|---:|---:|---:|---:|---:|
| M0 (SFT) | **0.3191** | **0.3126** | **0.4488** | **0.1357** | **0.579** |
| M1 (R1) | 0.3011 | 0.2947 | 0.4237 | 0.1258 | 0.594 |
| M2 (R2) | 0.3059 | 0.2998 | 0.4330 | 0.1230 | 0.614 |

### 4.2 Visual-centric difficulty (gpt-5.6-luna variant, 0–1)

| Model | Search | Coverage | Context | Spatial |
|---|---:|---:|---:|---:|
| M0 | 0.2824 | 0.2672 | 0.2694 | **0.0808** |
| M1 | 0.2894 | 0.2722 | 0.2832 | 0.0736 |
| M2 | **0.2908** | **0.2741** | 0.2804 | 0.0680 |

Judge is `gpt-5.6-luna` (`VARIANT_DOCUMENTED`, not the official GPT-5.4); same
judge used for the B2/B3 same-backbone baselines.

### 4.3 Independent visual support (gpt-4o, greedy)

| Model | Supported | Unsupported | Uncertain |
|---|---:|---:|---:|
| M0 | **0.737** | 0.003 | **0.260** |
| M1 | 0.701 | 0.001 | 0.298 |
| M2 | 0.701 | 0.001 | 0.298 |

### 4.4 Sampled diversity (4 questions/image, 4,000 questions)

| Model | Distinct-2 | dup rate | mean pairwise cosine distance | unique-question rate |
|---|---:|---:|---:|---:|
| M0 | **0.1916** | **0.347** | **0.524** | **0.959** |
| M1 | 0.0532 | 0.811 | 0.166 | 0.452 |
| M2 | 0.0417 | 0.857 | 0.097 | 0.361 |

---

## 5. Interpretation

* **Reference-based quality and diversity favor M0** on every metric
  (BLEU-4, Q-BLEU-4v, CIDEr, Distinct-2, duplicate rate, sampled diversity).
* **The adapted self-evolution does not improve over M0** on this image-only
  VQG evaluation. R2 (M2) is marginally better than R1 (M1) on reference
  metrics but remains below M0, and is *less* diverse than M1.
* **Visual-centric difficulty** moves slightly up for M1/M2 on
  Search/Coverage/Context but **Spatial drops**, and **independent visual
  support drops** from 0.737 (M0) to 0.701 (M1/M2). Combined with the large
  diversity loss, the adaptation trades breadth/grounding for a small
  difficulty shift.
* **Direction matches prior project evidence** that SFT is not improved by
  this self-training loop (`docs/SEEQ_JUDGE_ROBUSTNESS.md`,
  `docs/VISUAL_SUPPORT_DEPENDENCY_ABLATION_V1.md`).
* Caveat: M1/M2 answer-append was normalized away for question scoring; the raw
  outputs are retained in `*_greedy.jsonl` / `*_sampled4.jsonl`.

## 6. Preserved components (from the audit)

Iterative self-evolution loop; proposer improvement across rounds (M0→M1→M2);
stable-reference rewrite/filter (frozen M0); synthetic QA training; multiple
evolution rounds; VLM as proposer and judge.

## 7. Adaptations

* Filter/acceptance: static hard-constraint gate instead of official
  difficulty-improvement + model-following self-judge.
* No RAG image-fact cache, sample bank, or structured
  evidence/reasoning rewrite.
* Free-form rewrite prompt; QG+QA training format (Qwen3-VL-8B + LoRA).
* Retention is project-defined (deterministic image round-robin) because the
  official retention code is not released.

## 8. Limitations

* Not an exact SeeQ reproduction; official trainer, optimizer, and retention
  are unreleased (`UNKNOWN`).
* Visual-centric judge is a documented Luna variant, not official GPT-5.4.
* Single seed (42); no checkpoint selection and no development-score tuning.
* The `qg` target concatenates question and answer, so the adapter appends an
  answer that we strip at evaluation time; this is a property of the adapted
  training format.
* Prior `round2_generation` (M0 proposer) is invalid and excluded
  (`round2_generation_invalid/`).

## 9. Reproduction commands

```bash
# R2 generation (M1 proposer, frozen M0 reference)
.venv-vllm-cu128-v011/bin/python scripts/vvqg/run_seeq.py \
  --images outputs/same_backbone_baselines/manifests/train_unlabeled_images.jsonl \
  --stage-dir outputs/same_backbone_baselines/seeq_style/round2_generation_m1 \
  --base-model /hpc2hdd/home/fye374/models/Qwen3-VL-8B-Instruct \
  --adapter outputs/same_backbone_baselines/seeq_style/round1_m1 \
  --reference-adapter outputs/checkpoints/qwen3vl8b_lora_r16_vqav2_50k/checkpoint-final \
  --m0-hash 5fb66c36a2255fb3a9011db198fd55a8be8b9cbb26eef611727a816bc7934e0e \
  --current-m0-hash 5fb66c36a2255fb3a9011db198fd55a8be8b9cbb26eef611727a816bc7934e0e \
  --round 2 --retained 10000 --batch-size 128 --backend vllm --gpu-memory-utilization 0.7

# Canonical retention
.venv-vllm-cu128-v011/bin/python scripts/vvqg/prepare_seeq_round.py \
  --generation outputs/same_backbone_baselines/seeq_style/round2_generation_m1/stage_d_filters.jsonl \
  --output outputs/same_backbone_baselines/seeq_style/round2 \
  --m0-hash 5fb66c36a2255fb3a9011db198fd55a8be8b9cbb26eef611727a816bc7934e0e \
  --round 2 --retained 10000

# M1 -> M2
.venv/bin/python scripts/vvqg/train_seeq.py \
  --formatted outputs/same_backbone_baselines/seeq_style/round2/formatted_20000.jsonl \
  --base-model /hpc2hdd/home/fye374/models/Qwen3-VL-8B-Instruct \
  --start-adapter outputs/same_backbone_baselines/seeq_style/round1_m1 \
  --output outputs/same_backbone_baselines/seeq_style/round2_m2 \
  --dataloader-num-workers 8 --attn-implementation flash_attention_2
```
