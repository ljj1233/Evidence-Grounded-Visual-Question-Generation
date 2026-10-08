# FINAL REPORT — Joint soft-evidence question generation (E1_SOFT_EVIDENCE)

Run: `reports/joint_soft_evidence_qse_20261004T104324Z/`
Status: **STOPPED_AT_GATE** — Phase 0 PASS, Phase 1 GATE FAIL (evidence-slot
collapse); Phase 2–4 NOT_RUN.
Seed: 42. GPU: 0 (A800-80GB). git `9dea27f0`. No `final_holdout`. No RL.

## 1. Research hypothesis

A soft latent evidence branch, learned jointly with question decoding from the
shared Qwen3-VL image representation, will raise SeeQ visual-evidence Coverage
versus a strictly matched control (same data/backbone/seed/steps/decoding) while
keeping Supported and Unsupported within 0.01, and will make generation more
dependent on the actual image.

## 2. Implemented model structure (FACT)

`src/vvqg/evidence/model.py::SoftEvidenceQwen`, wrapping the frozen Peft backbone:

```
image → Qwen3VL vision tower (frozen)
      → merged image tokens (shared representation)
      → img_proj → learned slot queries (N=4) cross-attend → 4 soft evidence slots [B,4,512]
                 ├── gated cross-attention residual into every decoder layer (fuse_q/fuse_attn/out_proj, sigmoid gate)
                 ├── question decoder (LoRA), original image tokens preserved (bypass)
                 └── support_head (swap objective)
```
- Evidence slots are continuous vectors, never text; no external VLM planner; no
  hard region selection.
- Gate + `out_proj` zero-init → E1 starts exactly at C0.
- Inference computes evidence from the image alone; Flat/UVFC/GPT-4o are not used
  at generation time.
- Loss: `L = L_question + 0.10 L_align + 0.10 L_swap + 0.01 L_slot`.
  `L_align` is the documented question↔evidence contrastive fallback (Flat
  requirement *texts* are not persisted in any scored pool).
  `L_swap` uses the support head on in-batch image-question pairs.
  `L_slot` is confidence-weighted cosine repulsion.

## 3. C0 / E1 configuration (identical except the branch)

| setting | value |
|---|---|
| init | Final-SFT adapter `outputs/checkpoints/qwen3vl8b_lora_r16_vqav2_50k/checkpoint-final` |
| backbone | `/hpc2hdd/home/fye374/models/Qwen3-VL-8B-Instruct` (frozen) |
| LoRA | r16, α32, dropout 0.05, q/k/v/o/gate/up/down |
| optimizer | AdamW, lr 2e-4, betas (0.9,0.999), cosine, warmup 3% |
| batch / accum | 2 / 1 (Phase 1); seed 42; no GC |
| prompt | `Generate a question about this image.` |
| decoding (gen) | greedy, max_new_tokens 32 |
| E1 extra | N=4 slots, slot_dim 512, 8 heads, λ=(0.10, 0.10, 0.01) |

## 4. Commands actually run

```
.venv/bin/python scripts/vvqg/ec_evidence_smoke.py --records outputs/ec_qse_aligned_direct4/smoke_records.jsonl --outdir reports/joint_soft_evidence_qse_20261004T104324Z/smoke
.venv/bin/python scripts/vvqg/ec_aligned_direct4_generate.py --manifest outputs/ec_qse_aligned_direct4/phase1_manifest.jsonl --outdir outputs/ec_qse_aligned_direct4/phase1 --batch-size 16 --num-samples 4 --temperature 0.8 --max-new-tokens 64 --seed 42
.venv/bin/python scripts/vvqg/phase5b_score_candidates.py --candidates outputs/ec_qse_aligned_direct4/phase1/candidates.jsonl --out outputs/ec_qse_aligned_direct4/phase1/candidates_scored.jsonl --batch-size 32 --uvfc-batch-size 32
.venv-vllm-cu128-v011/bin/python scripts/vvqg/ec_build_targets.py --scored outputs/ec_qse_aligned_direct4/phase1/candidates_scored.jsonl --manifest outputs/ec_qse_aligned_direct4/phase1_manifest.jsonl --selector s3 --image-balanced --out outputs/ec_qse_aligned_direct4/phase1/targets_phase1.jsonl
.venv/bin/python scripts/vvqg/ec_evidence_train.py --variant c0 --targets outputs/ec_qse_aligned_direct4/phase1/targets_phase1.jsonl --outdir outputs/ec_qse_aligned_direct4/phase1/c0 --epochs 1 --batch-size 2 --grad-accum 1 --num-workers 2 --seed 42
.venv/bin/python scripts/vvqg/ec_evidence_train.py --variant e1 --targets outputs/ec_qse_aligned_direct4/phase1/targets_phase1.jsonl --outdir outputs/ec_qse_aligned_direct4/phase1/e1 --epochs 1 --batch-size 2 --grad-accum 1 --num-workers 2 --seed 42
.venv/bin/python scripts/vvqg/ec_evidence_generate.py --variant c0 --checkpoint outputs/ec_qse_aligned_direct4/phase1/c0/checkpoint-final --manifest outputs/ec_qse_aligned_direct4/phase1_manifest.jsonl --output outputs/ec_qse_aligned_direct4/phase1/gen_c0.jsonl --max-new-tokens 32
.venv/bin/python scripts/vvqg/ec_evidence_generate.py --variant e1 --checkpoint outputs/ec_qse_aligned_direct4/phase1/e1/checkpoint-final --manifest outputs/ec_qse_aligned_direct4/phase1_manifest.jsonl --output outputs/ec_qse_aligned_direct4/phase1/gen_e1.jsonl --max-new-tokens 32
.venv/bin/python scripts/vvqg/ec_phase1_health.py --c0 outputs/ec_qse_aligned_direct4/phase1/gen_c0.jsonl --e1 outputs/ec_qse_aligned_direct4/phase1/gen_e1.jsonl --e1-checkpoint outputs/ec_qse_aligned_direct4/phase1/e1/checkpoint-final --manifest outputs/ec_qse_aligned_direct4/phase1_manifest.jsonl --out reports/joint_soft_evidence_qse_20261004T104324Z/phase1_health.json
```

## 5. Data

| item | value |
|---|---|
| Phase-1 images / records | 64 / 64 (image-balanced, weight sum 64.0) |
| Phase-1 candidates | 244 (S3 targets; 2 fallback) |
| steps | 32 (= cap ≤ 50) |
| seed | 42 |

## 6. Smoke / 7. Formal training

Smoke PASS (`SMOKE_RESULTS.md`). Formal Phase-2 training **NOT_RUN** (Phase-1 gate
failed; see `PHASE1_RESULTS.md`).

## 8–11. Main metrics / CI / support-duplicate / intervention

NOT_RUN. No evaluation against SeeQ/GPT-4o was performed (no evaluator calls; no
new paid API used).

## 12. Evidence-slot collapse

**YES — collapsed.** Evidence-output mean off-diagonal cosine ≈ 1.000; per-slot
dispersion ≈ 7.2e-5. Learned queries stay distinct (cos 0.0148), so the
cross-attention output is query-insensitive.

## 13. Failure cases

`FAILURE_CASES.jsonl` — 40 Phase-1 C0/E1 pairs (26 differ), each with `image_id`,
`C0_question`, `E1_question`, `intervention_result` = NOT_RUN.

## 14. Gates

| gate | result |
|---|---|
| Phase 0 smoke | PASS |
| Phase 1 | **FAIL** (slot collapse; other checks pass) |
| Phase 2 main | NOT_RUN |
| Phase 3 intervention | NOT_RUN |
| Phase 4 ablation | NOT_RUN |

## 15. Next stage

NOT allowed. Failure occurs in **evidence learning** (soft evidence slots
collapsed). Secondary blocker: the prompt-aligned train pool does not exist and
costs ~9–10 h of frozen-Flat scoring to produce.
