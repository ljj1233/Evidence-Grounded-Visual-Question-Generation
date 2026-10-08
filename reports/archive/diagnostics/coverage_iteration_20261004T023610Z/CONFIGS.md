# CONFIGS — frozen baseline facts

Recorded 2026-10-04T02:36Z before any new run. Nothing here was modified.

## Repository state

- git commit: `9dea27f09aa5c21f4fa53f730923ac1df64169eb` (branch `master`)
- working tree: dirty (pre-existing renames/modifications; NOT touched by this audit)
- protected paths intact: `outputs/checkpoints/**`, `data/vqav2/**`,
  `src/vvqg/data/sft/**`; `final_holdout` untouched.

## Environments

| env | use |
|---|---|
| `.venv` | generation, UVFC/Flat scoring, SFT/distill/RL training |
| `.venv-vllm-cu128-v011` | metrics / SeeQ / support evaluation |

Judge creds from `.env` (git-ignored): `LLM_JUDGE_BASE_URL`,
`LLM_JUDGE_API_KEY`, `LLM_JUDGE_MODEL_NAME=gpt-5.6-luna`.

## Baseline chain (A0)

Stage order: Final-SFT → Direct-4 (K=4) → frozen UVFC + Flat verifier →
S3 ConstraintFirst τ=0.5 (S1 fallback) → D1 single-positive distillation →
G2 Static-UVFC group-relative RL from the D1 init.

### Model / seed / split / checkpoint

- backbone: `Qwen3-VL-8B-Instruct` (`/hpc2hdd/home/fye374/models/Qwen3-VL-8B-Instruct`)
- LoRA: r=16, alpha=32, dropout 0.05, 7 modules, 43,646,976 trainable
- flat verifier: `Qwen2.5-VL-7B-Instruct`
- UVFC: frozen SAE + u at `outputs/phase26/vfc/`
- seed: 42 (all stages)
- train manifest: `outputs/phase4b_online_uvfc/data/train_10000.jsonl`
  (10000 records / 9184 images)
- dev: `outputs/same_backbone_baselines/manifests/development1000_images.jsonl`
  (development1000) — method-selection data, not test
- base SFT ckpt: `outputs/checkpoints/qwen3vl8b_lora_r16_vqav2_50k/checkpoint-final`
- D1 ckpt: `outputs/phase5c_distill_single/checkpoint-final`
- G2 output: `outputs/phase5c_qse_relativeguard/` (stopped step 256)

### Baseline config paths

- SFT: `configs/new_vqg/sft.yaml`
- Direct-4 candidate pool runtime: `outputs/phase5c_qse_teacher/pool/generation_runtime.json`
  (instruction = DIRECT4; K=4, T=0.8, mn=64, seed 42, sdpa, batch 16)
- selector logic: `scripts/vvqg/phase5c_select.py` (S3, τ=0.5) + `phase5b_select.py` (S1)
- distillation: `configs/phase5b/distill.yaml` (recipe), runner
  `scripts/vvqg/phase5c_distill.py`
- G2: `configs/phase5c/qse_relativeguard.yaml`,
  `scripts/vvqg/train_online_uvfc_relative.py`

### Commands (as used / reconstructed from runtime artifacts)

```bash
# Direct-4 dev pool (A0 teacher dev)
.venv/bin/python scripts/vvqg/phase5b_direct4_generate.py \
  --outdir outputs/phase5b_direct4_uvfc \
  --manifest outputs/same_backbone_baselines/manifests/development1000_images.jsonl

# Direct-4 train pool (D1 teacher)
.venv/bin/python scripts/vvqg/phase5b_direct4_generate.py \
  --outdir outputs/phase5c_qse_teacher/pool \
  --manifest outputs/phase4b_online_uvfc/data/train_10000.jsonl

# UVFC + Flat scoring
.venv/bin/python scripts/vvqg/phase5b_score_candidates.py \
  --candidates <pool>/candidates.jsonl --out <pool>/candidates_scored.jsonl

# S3 selector (tau=0.5)
.venv/bin/python scripts/vvqg/phase5c_select.py \
  --scored <pool>/candidates_scored.jsonl --selector s3 --tau 0.5 --outdir <out>

# D1 targets + distillation
.venv/bin/python scripts/vvqg/phase5c_build_targets.py \
  --scored outputs/phase5c_qse_teacher/pool/candidates_scored.jsonl \
  --manifest outputs/phase4b_online_uvfc/data/train_10000.jsonl \
  --outdir outputs/phase5c_qse_teacher --tau 0.5
.venv/bin/python scripts/vvqg/phase5c_distill.py \
  --targets outputs/phase5c_qse_teacher/targets_d1_single.jsonl \
  --outdir outputs/phase5c_distill_single

# G2 RL
.venv/bin/python scripts/vvqg/train_online_uvfc_relative.py \
  --config configs/phase5c/qse_relativeguard.yaml

# checkpoint greedy generation
.venv/bin/python scripts/vvqg/phase5a_generate_checkpoints.py \
  --run-dir <run> --out <run>/checkpoints_eval --tags step_0000 step_0050 ...

# evaluation (SeeQ gpt-5.6-luna + GPT-4o support + traditional)
.venv-vllm-cu128-v011/bin/python scripts/vvqg/phase5a_evaluate.py \
  --predictions <greedy1000.jsonl> --system <name> --out <eval_dir>
```

### A0 measured baselines (development1000 greedy, 1 q/img)

| stage | Coverage | Supported | Unsupported | note |
|---|---:|---:|---:|---|
| SFT | 0.2598 | 0.725 | 0.159 | `outputs/checkpoints/.../checkpoint-final` |
| D1 single-pos | 0.3612 | 0.830 | 0.050 | `outputs/phase5c_distill_single/` |
| G2 (D1 init) | 0.3908 | 0.851 | 0.039 | **step 256, guard stop — not 1000** |
| G1 control (SFT init) | 0.3312 | 0.803 | 0.100 | 1000 steps, no trip |

G2 guard thresholds (from `training_summary.json`): `t_dup=0.4578125`,
`t_length=10.0234375`, `t_malformed=0.02`; duplicate metric = corpus duplicate
count / total on NFKC-normalized questions.

### Evaluator constants

- Coverage = SeeQ `visual_evidence_coverage` (0–5), `gpt-5.6-luna` variant,
  `--allow-variant`, reported mean/5.
- Supported/Unsupported/Uncertain = GPT-4o `visual-support-phase23-v1.0`
  `overall_visual_support`.
- references: `outputs/evaluation/sft_greedy/val2014_sample1000.jsonl`.
- judges are non-deterministic: existing follow-up review measured 39–64 support
  flips / 1000 and 412/999 SeeQ score-vector changes on **identical** questions.
