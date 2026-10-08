# Online UVFC + Violation GRPO — Phase 4B Results

Status: Phase 4B.0 and 4B.1 complete. `final_holdout` was not touched. This is a
research pilot, not a final / production GRPO system.

## 1. What this is

Tests whether UVFC can guide a policy when the reward is computed from **fresh
on-policy rollouts**. Two variants start from the identical Final SFT checkpoint:

- **Static UVFC (Variant A)** — frozen feature weights and frozen undercoverage
  statistics; the reward equals the Phase-3A fixed-rollout reward.
- **Dynamic UVFC (Variant B)** — the visual-sensitivity component `v_j` stays
  frozen; the undercoverage component `u_j` is refreshed from the current-policy
  feature distribution every 200 optimizer steps.

Feature extraction, the Top-K SAE and the UVFC formulation are unchanged; only
the policy coverage statistics move. Initialization is always
`outputs/checkpoints/qwen3vl8b_lora_r16_vqav2_50k/checkpoint-final` (never
P1strong / P2 / UVFC-RFT).

## 2. Protocol

| Item | Value |
|---|---|
| Reward | `UVFC` with a violation-first lexicographic group ranking |
| Verifier | frozen Flat verifier (prompt/parser/threshold/aggregation unchanged) |
| Rollouts | 4 questions/image/step, temp 0.8, top_p 0.9, max_new_tokens 32, no cache |
| Objective | Phase-3A clipped group-relative surrogate; completion tokens only; clip 0.2; reference KL 0.02; no value/reward model |
| Advantage | rank slots (+1.5, +0.5, -0.5, -1.5), tie-mean, population-standardized within group |
| Dynamic refresh | `u_j = max(0, log((p_anchor(j)+eps)/(p_policy(j)+eps)))` every 200 steps; `v_j` frozen |
| Data | feasibility 2,000 train records; main 10,000 train records (all 9,184 unique train2014 images + 816 repeats), seed 42 |
| Evaluation | development1000 greedy + sampled4x; SeeQ/Luna, GPT-4o support, BLEU/CIDEr, diversity |

Online UVFC was verified to reproduce the frozen Phase-2.6 scores exactly on 32
candidates with the frozen batch grouping (max abs UVFC error 0.0, 0 active
feature mismatches; `outputs/phase26/vfc/online_verification.json`).

## 3. Phase 4B.0 feasibility

2,000 train records, seed 42, 200 optimizer steps, group size 4. Wall 4,645 s
(77 min). No guard triggered. `feasibility/decision.json` = **FEASIBILITY_GO**
(fresh rollouts, UVFC scoring, violation constraint, GRPO update, checkpointing,
stability all true). Stability: KL ≤ 0.0024, grad norm ≈ 0.83, malformed 0,
duplicate rate ≤ 0.19, 2/1600 groups skipped.

## 4. Phase 4B.1 results

Both variants: 1000 optimizer steps, 8 groups/step, group size 4, identical
Final SFT start, no early stop. Wall: static 33,481 s (9.3 h), dynamic 31,268 s
(8.7 h).

### Final table (development1000, greedy1000)

| Model | Search | Coverage | Context | Spatial | Supported | Unsupported |
|---|---:|---:|---:|---:|---:|---:|
| SFT | 0.2746 | 0.2598 | 0.2704 | 0.0732 | 0.725 | 0.159 |
| Fixed-rollout P1strong | 0.3200 | 0.3256 | 0.2854 | 0.0622 | 0.734 | 0.114 |
| Static UVFC Online GRPO | **0.3316** | **0.3324** | 0.2554 | 0.0710 | **0.814** | **0.094** |
| Dynamic UVFC Online GRPO | 0.2924 | 0.2786 | 0.2498 | 0.0760 | 0.758 | 0.118 |

Support denominator 1000/1000 valid for both variants after retry. Extra fields
in `comparison/final_table.json`: Uncertain (SFT 0.116, P1strong 0.152, static
0.092, dynamic 0.124), BLEU (0.317 / 0.292 / 0.326 / 0.312), CIDEr (0.455 /
0.470 / 0.519 / 0.467), greedy dup (0.570 / 0.461 / 0.499 / 0.516), semantic
diversity (0.536 / 0.542 / 0.537 / 0.535).

### Paired bootstrap (5,000 resamples, seed 42, image-level; 95% percentile CI)

| Metric | Static − SFT | Dynamic − SFT | Dynamic − Static | Static − P1strong |
|---|---:|---:|---:|---:|
| Search | +0.0570 [0.0450, 0.0690] | +0.0178 [0.0096, 0.0260] | −0.0392 [−0.0510, −0.0280] | +0.0116 [0.0012, 0.0218] |
| Coverage | +0.0726 [0.0598, 0.0864] | +0.0188 [0.0086, 0.0288] | −0.0538 [−0.0668, −0.0416] | +0.0068 [−0.0050, 0.0192] |
| Unsupported | −0.0650 [−0.0870, −0.0430] | −0.0410 [−0.0610, −0.0210] | +0.0240 [0.0070, 0.0420] | −0.0200 [−0.0370, −0.0020] |

## 5. Interpretation

`comparison/decision.json` = **STATIC_WORKS_DYNAMIC_NOT_NEEDED**.

- Online Static UVFC transfers: Search and Coverage improve over SFT with CIs
  excluding zero, and it matches or exceeds the fixed-rollout P1strong baseline
  (Search +0.0116 with CI > 0, Coverage +0.0068 with CI spanning zero) while
  raising Supported to 0.814 and lowering Unsupported to 0.094.
- Dynamic UVFC also improves over SFT, but is strictly worse than Static on
  Search/Coverage/Supported and higher on Unsupported (all CIs exclude zero), so
  refreshing `u_j` from the current policy gives no extra gain.
- Mechanism: the refreshed `u_j` collapses as soon as the current policy covers
  the anchor features, removing the undercoverage signal. Bucketed mean UVFC
  (logged with the live `u`): Static 0.528 → 0.636 across 1000 steps; Dynamic
  0.528 → 0.073 after the first refresh at step 200 (`dynamic_uvfc/uvfc_refresh/`).
- Context/Spatial do not improve; the reward does not target those axes. No
  degeneration: malformed 0, length stable, duplicate rate bounded.

## 6. Outputs

```
outputs/phase4b_online_uvfc/
  feasibility/{config.json,train_log.jsonl,decision.json}
  static_uvfc/{checkpoints/,rollouts/,train_log.jsonl,evaluation/,decision.json,greedy1000.jsonl,sampled4x1000.jsonl}
  dynamic_uvfc/{checkpoints/,rollouts/,uvfc_refresh/,train_log.jsonl,evaluation/,decision.json,greedy1000.jsonl,sampled4x1000.jsonl}
  comparison/{final_table.json,paired_bootstrap.json,decision.json}
```
Checkpoints include adapter weights/config, optimizer/scheduler state, global
step, python/numpy/torch/CUDA RNG state and a SHA-256 manifest.

## 7. Reproduce

```bash
.venv/bin/python scripts/vvqg/phase4b_prepare.py
.venv/bin/python scripts/vvqg/phase4b_verify_uvfc.py
.venv/bin/python scripts/vvqg/train_online_uvfc_grpo.py --config configs/phase4b/feasibility.yaml
bash scripts/vvqg/phase4b_run_main.sh          # static + dynamic + evaluation + comparison
```
