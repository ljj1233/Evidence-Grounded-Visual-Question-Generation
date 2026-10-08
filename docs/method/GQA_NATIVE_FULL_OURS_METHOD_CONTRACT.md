# GQA-native Full Ours — frozen method contract

Status: **FROZEN** (2026-10-07, seed 42). Read-only archaeology plus a
GQA-native adaptation plan. No VQAv2 learned artifact may be reused; no GQA model
had been trained when this was written.

This document is the required "先冻结方法合同" deliverable. It records, for the
canonical Full Ours, every module's init source, fit image set, I/O dimensions,
loss/hyperparameters, checkpoint location and final-checkpoint rule, and then the
exact GQA-native substitution. Unresolvable historical ambiguities are listed in
§7 instead of being silently resolved.

## 1. What "canonical Full Ours" is

The frozen method is **Phase 5C G2** (docs/PHASE5C_EVIDENCE_CONSTRAINED_QSE.md §N
"FREEZE METHOD"), not the soft-evidence / VC-QSE / support-delta branches (which
this task explicitly excludes). One data flow:

```
public Qwen3-VL-8B-Instruct
  -> SFT (image -> question)
  -> UVFC build in the SFT representation space (SAE + v_j + u_j)
  -> Direct-4 candidates (plain prompt, K=4)
  -> local Flat verifier (Qwen2.5-VL-7B) + frozen UVFC scoring
  -> S3 ConstraintFirst selection (tau=0.5)
  -> D1 single-positive distillation
  -> G2 static-UVFC group-relative RL from D1
  -> Full Ours
```

Provenance of the reconstruction (read-only):
`reports/full_ours_architecture_audit_20261004T072232Z/` (ARCHITECTURE_AUDIT.md,
component_contract.json, missing_information.md) and
`reports/phase5c_followup/REVIEW.md`.

## 2. Canonical VQAv2 contract (reference, per stage)

| stage | init source | fit data (image_id set) | dims / loss / hyperparams | checkpoint |
|---|---|---|---|---|
| SFT | public `Qwen3-VL-8B-Instruct`, **fresh** LoRA | `src/vvqg/data/sft/train.jsonl` 50000 rec / 9184 imgs | LoRA r16 a32 drop0.05, 7 proj modules; lr2e-4 cosine warmup0.03, eff batch 16, 2 epochs, bf16, flash-attn2, max_seq512, img 4096/147456; CE on assistant tokens | `outputs/checkpoints/qwen3vl8b_lora_r16_vqav2_50k/checkpoint-final` |
| UVFC build | SFT adapter frozen | discovery 20000 SFT-train recs; calibration 5000 real + 5000 blank; policy = greedy on the 9184 SFT-train images | layer 18/36, hidden 4096, token cap 16 (first question tokens, max-pool); SAE tied-weight TopK dict4096 topk20, AdamW lr1e-3, 3 epochs, batch512, seed42; `v_j=max(0,p_real-p_blank)`, `u_j=max(0,log((p_anchor+1e-6)/(p_policy+1e-6)))`, `UVFC=sum_j active_j v_j u_j` | `outputs/phase26/vfc/{sae.pt,weights.pt}` |
| Direct-4 | SFT adapter frozen | 9184 train imgs (10000 rec) | `DIRECT4` prompt; K=4, T=0.8, top_p0.9, max_new64, seed42, batch16; within-image dedup, cap K | `outputs/phase5c_qse_teacher/pool/candidates.jsonl` (35097) |
| scoring | frozen UVFC + frozen Flat | same candidates | Flat: `Qwen2.5-VL-7B-Instruct`, greedy, max_new256, `support_floor=min(requirement scores)`, `flat_state` in {NO_CONFIRMED_VIOLATION, CONFIRMED_VIOLATION, UNCERTAIN, FAILED}; parse fail -> FAILED, floor=-1.0 | `.../pool/candidates_scored.jsonl` |
| S3 selector | none (deterministic) | 9184 train / 1000 dev | keep `state!=CONFIRMED_VIOLATION` and `support_floor>=tau`, pick max UVFC, tie `sha256("42:{image}#{gen}")`; S1 fallback if empty; **tau=0.5** (frozen, not re-calibrated) | selection jsonl |
| D1 target | none | method_train_10000 (10000 rec / 9184 imgs) | `best_question = S3` repeated for epoch1 and epoch2, one row per record | `outputs/phase5c_qse_teacher/targets_d1_single.jsonl` |
| D1 distill | **canonical Final SFT** | 10000 rec | init=Final SFT, LoRA r16 trainable; lr2e-4 cosine warmup0.03, batch2xaccum8=16, 2 epochs, 1250 steps, seed42, unweighted, CE on assistant tokens | `outputs/phase5c_distill_single/checkpoint-final` |
| G2 RL | **D1** (canonical) | method_train_10000 (10000 rec) | static UVFC reward `sum v_j u_j` + violation-first `p2` group rank; advantages rank slots (+1.5,+0.5,-0.5,-1.5) tie-mean population-standardised; group_size4, groups_per_step8, **inner_epochs=1**; lr1e-6 cosine, beta_kl0.02, clip0.2, log_ratio_clamp20, max_steps1000; frozen reference = D1; relative guard (below) | `outputs/phase5c_qse_relativeguard/` |
| final gen | G2 final adapter | development1000 | greedy T=0, `Generate a question about this image.`, max_new32 | `.../greedy1000.jsonl` |

G2 relative guard (verified in `online_uvfc_relative_grpo.py`): measure `D_init`
on a fixed 32-image seed-42 calibration subset (group4, T=0.8) before training and
`D_sft0` on the same subset with the SFT checkpoint;
`t_dup = D_init_dup + 0.2`, `t_length = D_init_len + 0.5*D_sft0_len`,
`t_malformed = D_init_malformed + 0.02`; abort KL > 0.2 for 10 steps, clip > 0.5
for 10 steps, degeneration for `DEGEN_STREAK` consecutive steps.
Canonical G2 VQAv2 values: t_dup=0.4578, t_length=10.0234, t_malformed=0.02,
stopped at step 256/1000 reason `DEGENERATION`.

Final-checkpoint rule: the last saved checkpoint at the guard stop
(`step_final`); no dev metric is used to extend or shorten training.

## 3. GQA-native substitution (this task)

Public frozen components reused: `Qwen3-VL-8B-Instruct` backbone,
`Qwen2.5-VL-7B-Instruct` Flat verifier, tokenizer/processor, method code, the
already-determined prompts/loss/selector/hyperparameters, and the frozen GQA data
split + evaluation protocol. **No VQAv2-learned artifact is loaded** (no SFT / D1 /
G2 adapter, no SAE / v / u / projection, no candidate pool, no optimizer state).

| stage | GQA init source | GQA fit data | produces |
|---|---|---|---|
| SFT | public backbone, fresh LoRA | `outputs/gqa_native/manifests/sft_train_50000.jsonl` 50000 rec / **32139 unique imgs** | `outputs/gqa_native/full_ours/sft/checkpoint-final` |
| UVFC build | GQA-SFT adapter frozen | discovery 20000 + calibration 5000 real + 5000 blank drawn seed-42 from the GQA-SFT-50K records; policy = greedy on `method_train_10000` (10000 imgs) | `outputs/gqa_native/full_ours/uvfc/{sae.pt,weights.pt}` |
| Direct-4 | GQA-SFT adapter frozen | `method_train_10000` (10000 imgs) | `.../direct4/candidates.jsonl` (~40000, <=4/img) |
| scoring | GQA UVFC + frozen Flat | same candidates | `.../direct4/candidates_scored.jsonl` |
| S3 | none | same candidates, tau=0.5 | `.../qse_teacher/selected/` |
| D1 target | none | `method_train_10000` (10000 rec, 1/img) | `.../qse_teacher/targets_d1_single.jsonl` |
| D1 distill | **GQA-SFT** | 10000 rec | `outputs/gqa_native/full_ours/distill_d1/checkpoint-final` |
| G2 RL | **GQA D1** | `method_train_10000` | `outputs/gqa_native/full_ours/g2/` |
| final gen | GQA G2 final | `dev1000_images.jsonl` | `.../g2/greedy1000.jsonl` |
| eval | — | `dev1000_refs.json` | `.../evaluation/dev1000/` |

GQA guard baselines: `guard_sft_checkpoint = GQA-SFT`,
`guard_calibration_manifest = method_train_10000`, n=32, seed42.

Evaluation reuses `scripts/vvqg/gqa_evaluate.py` (same SeeQ-Luna + GPT-4o + BLEU/
CIDEr/Q-BLEU4/Dist-2/Dup implementations as the completed GQA baselines) with the
frozen content-addressed evaluator cache.

## 4. Explicit non-reuse

- **VQAV2_LEARNED_WEIGHTS_REUSED: NO** — adapters, SAE, v/u, normalizers, candidate
  pools, distillation targets, reward-calibration samples, hidden caches.
- **BASELINE_LEARNED_WEIGHTS_REUSED: NO** — B2/B3 adapters, SeeQ M0-M2, guided heads.
- `final_holdout` is never read; dev1000 refs are evaluator-only and never used for
  training, fitting, calibration or checkpoint selection.

## 5. Dependencies and order

UVFC build and Direct-4 generation both depend only on GQA-SFT and are mutually
independent (canonical code: `phase26_vfc.py` SAE/v/u are fit without candidates;
`phase5b_direct4_generate.py` needs only the SFT adapter). Order:
SFT -> {UVFC build, Direct-4} -> score -> S3 -> D1 -> G2 -> eval.

## 6. Cost discipline

Default Direct-4 only on train (`method_train_10000`); dev1000 only for the final
Greedy generation. No dev Direct-4 for diagnostics. No external API for SFT/D1/
candidates. UVFC build is fully local (no API). Local smoke uses a small train
subset and never reads holdout. No hyperparameter grid, no auto epoch extension.

## 7. Recorded ambiguities / deviations (must be reported, not hidden)

1. **UVFC policy image set.** Canonical `phase26_vfc.py` derives the policy
   feature frequency from *all* unique images of the SFT train manifest (9184 for
   VQAv2). The frozen `docs/GQA_NATIVE_DATA_CONTRACT.md` §3/§5 assigns
   `method_train_10000` to Static UVFC. GQA-SFT-50K has 32139 unique images, so
   the two readings differ. **Default chosen: policy = `method_train_10000`
   (GQA contract); discovery/calibration = GQA-SFT-50K records (the 25k-record
   draw the canonical builder needs).** This mixes the policy image set with the
   discovery image set, a deviation from the VQAv2 identity, recorded here.
2. **Guard version of the historical G2 run.** The archaeology audit
   (`reports/full_ours_architecture_audit_20261004T072232Z/missing_information.md`
   item 3) records as **UNKNOWN** whether the canonical step-256 G2 run used the
   current `DEGEN_STREAK=3`; the current code would not stop at 256. GQA G2 uses
   the **current** `online_uvfc_relative_grpo.py` verbatim. This is recorded as an
   **implementation correction**, not as a byte-identical replay of the VQAv2 run.
   The VQAv2 step 256 is **not** reused as a GQA training length and no GQA dev
   metric will be consulted to extend/shorten GQA G2.
3. **inner_epochs.** The canonical configs (`qse_relativeguard.yaml`,
   `static_relativeguard.yaml`) both set `inner_epochs: 1`, confirmed by
   `reports/phase5c_followup/REVIEW.md`. GQA uses 1.
4. **Record weighting.** VQAv2 `train_10000.jsonl` is 10000 records over 9184
   images (some images repeated); GQA `method_train_10000` is 10000 unique images
   (1 record each). Records and unique images are reported separately.
5. **tau=0.5** is reused as a frozen method hyperparameter; it is **not**
   re-calibrated on GQA dev (no dev leakage). The canonical tau calibration set
   overlapped development1000; GQA avoids that by not recalibrating.
6. **No external API in the UVFC build.** The canonical UVFC construction is
   fully local, so no annotation budget is required for §5 of the task.

---

## Revision (2026-10-07): base-init, shared 10k manifest

The user removed the 50k GQA-SFT stage. The canonical GQA method is now:

```
base Qwen3-VL-8B-Instruct
  -> UVFC in the BASE representation (adapter=none)
       discovery 20000 + calibration 5000 real + 5000 blank (seed42, GQA train 50k)
       policy = greedy on method_train_10000
  -> Direct-4 from base on method_train_10000 (K=4, T=0.8, mn=64)
  -> Flat + UVFC -> S3 (tau 0.5) -> D1 distillation from base (10000 records)
  -> G2 static-UVFC GRPO from D1 -> Full Ours
```

Deviation records (supersede §2/§3 above for this revision):
- No GQA-SFT; D1 init is the base (materialized as a fresh zero-init LoRA,
  `full_ours_base/base_zero_lora`, mathematically the base weights).
- UVFC is fit in the base representation (was SFT representation).
- B2/B3 are retrained from base on the same 10k manifest with **10000 records**
  (one question per image, seed-42 selection), matching Full Ours' supervised
  budget; the previous B2/B3 checkpoints used 15569 records and are not reused.
- SeeQ unchanged (base M0, 10k manifest).
- `g2_base.yaml`: `sft_checkpoint` = D1; `guard_sft_checkpoint` = base zero-LoRA
  (D_sft0 estimated with the base policy); `uvfc_dir` = base UVFC.
- The earlier SFT-based run under `outputs/gqa_native/full_ours/` is retained but
  superseded.
