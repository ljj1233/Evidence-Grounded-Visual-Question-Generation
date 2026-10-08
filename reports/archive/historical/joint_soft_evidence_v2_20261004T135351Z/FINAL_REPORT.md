# FINAL REPORT — Joint Soft-Evidence QSE v2 (candidate-assigned visual slots)

Run: `reports/joint_soft_evidence_v2_20261004T135351Z/`
Status: **COMPLETE through Phase 4; MAIN GATE FAIL** (slot collapse solved,
Coverage not significantly improved). No RL; no intervention stage.
seed 42 · GPU 0 (A800-80GB) · git `9dea27f0` · pool sha256 `d1b8f978…`.

## 1. Hypothesis

Assigning each soft evidence slot to a distinct candidate question of the same
image (candidate-local Hungarian assignment) will prevent slot collapse at
init/training and yield a usable visual-evidence representation, and may improve
SeeQ Coverage over a strictly matched control.

## 2. v1 collapse diagnosis

Every v1 gradient-bearing objective pooled over slots (`evidence.mean(1)` in
`align`/`swap`); `out_proj=0` and a shared/unscaled read-out produced uniform
attention, so slots converged to the mean image vector. Root cause = absence of
per-slot supervision plus a forced-averaging read-out.

## 3. v2 structure

`SoftEvidenceQwenV2`: shared Qwen3-VL image representation → per-slot query
projection → cosine cross-attention over image tokens (shared K/V) → LayerNorm →
gated (≈0.01) cross-attention residual into every decoder layer; image bypass
preserved. Training adds a frozen text-only Qwen3-VL question encoder (detached,
frozen projection) and per-image Hungarian assignment; loss
`CE + 0.20·assign + 0.05·swap + 0.01·slot`. No text evidence, no external
planner, no hard regions, no RL; inference = image + instruction.

## 4. New files

`src/vvqg/evidence/model_v2.py`, `src/vvqg/evidence/assignment.py`,
`src/vvqg/evidence/__init__.py` (updated), `scripts/vvqg/ec_build_slot_packets.py`,
`scripts/vvqg/ec_slot_assignment_train.py`, `ec_slot_assignment_generate.py`,
`ec_slot_assignment_smoke.py`, `ec_phase1_v2_diagnostics.py`,
`ec_v2_compare.py`, `ec_v2_assignment_cases.py`,
`tests/test_slot_assignment_v2.py`. No existing core file modified.

## 5. Commands

See `PHASE1_RESULTS.md`, `PHASE2_RESULTS.md`, `PHASE3_4_RESULTS.md` and
`RUN_COMMANDS.md`.

## 6. Data / hashes

Train pool `outputs/phase5c_qse_teacher/pool/candidates_scored.jsonl`
(sha256 `d1b8f978a9fb9037fb292afdb0f547de874d6cd73bbdc71dc57d69a4c5fb1f63`,
35097 candidates / 9184 images). Dev manifest `development1000_images.jsonl`.
Frozen embeddings: 223 (Phase 1), 1439 (Phase 2), 14058 (Phase 3) × 512.

## 7. Gates

| gate | result |
|---|---|
| Phase 0 unit tests (10) | PASS |
| Phase 0 model smoke | PASS |
| Phase 1 (64 img, 50 steps) | **PASS** |
| Phase 2 (500 img, 200 steps) | **PASS** |
| Phase 3 (9184 img, 1148 steps) | PASS (no collapse) |
| Phase 4 main gate | **FAIL** (Coverage CI includes 0) |
| Image intervention | NOT RUN (gated on Phase 4) |

## 8. Headline numbers

- Slot collapse solved: slot off-diagonal cosine 1.000 (v1) → **−0.25** (v2, full
  training); slot std 1.02; assignment gap +0.054; max slot share 30.9%.
- Coverage: C0 1.937 vs E2 1.951, **Δ +0.014, 95% CI [−0.024, +0.053] (n=1000)**.
- Supported 0.837 → 0.8398; Unsupported 0.057 → 0.0541; duplicate excess 0.383 →
  0.387; mean words 5.82 → 5.83.

## 9. Conclusion

**candidate assignment solves slot collapse but does not yet improve VQG
Coverage.** The evidence representation is now usable (non-collapsed,
assignment-supervised) but the small aggregate Coverage change is not
statistically distinguishable from zero, and the per-image result is mixed
(59 up / 50 down; 941/1000 non-positive).

## 10. Artifacts

`AUDIT.md`, `READOUT_FIX.md`, `CONFIG_DIFF.md`, `PHASE1_RESULTS.md`,
`PHASE2_RESULTS.md`, `PHASE3_4_RESULTS.md`, `FINAL_REPORT.md`, `DECISION.md`,
`SMOKE_RESULTS.md`, `v2_assignment_cases.jsonl` (20), `FR_FAILURE_CASES.jsonl`
(40), `phase1/phase2/phase4/comparison.json`, `phase*_v2_diagnostics.json`.
