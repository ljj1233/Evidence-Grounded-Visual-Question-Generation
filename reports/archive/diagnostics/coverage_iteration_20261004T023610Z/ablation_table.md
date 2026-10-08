# Ablation table — coverage iteration (Phase 1)

Development1000, one S3-selected question per image, same evaluator
(SeeQ `gpt-5.6-luna` Coverage / Search / Context / Spatial; GPT-4o support;
traditional BLEU/CIDEr/Distinct-2/Dup). Primary metrics are Coverage / Supported
/ Unsupported; the rest are auxiliary.

## Candidate pools (frozen UVFC + Flat, S3 τ=0.5 with S1 fallback)

| pool | images | raw cands | dedup cands | exact-dup ratio | within-image dup | S1 fallback | selected violation | mean floor | mean UVFC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 (SFT instruction) | 1000 | 4000 | 3828 | 0.354 | 0.043 | 0.031 | 0.026 | 0.938 | 0.702 |
| A2 (4 focus prompts) | 1000 | 4000 | 3841 | 0.358 | 0.040 | 0.017 | 0.014 | 0.943 | 0.685 |

A2 does not increase duplication (within-image dup slightly lower; corpus exact-dup
essentially equal). A2 raises object/attribute support (`support_floor` mean 0.743 /
0.730 vs 0.705 / 0.686) and spatial UVFC (0.550 vs 0.517), and lowers object/attribute
UVFC and scene support (see `candidate_pool_stats.json` per-focus).

## Primary metrics (single evaluator run)

| arm | Coverage | Supported | Unsupported | Search | Context | Spatial | Dup | Dist-2 | BLEU-4 | CIDEr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 S3 | 0.3754 | 0.7513 | 0.0971 | 0.3444 | 0.2762 | 0.0688 | 0.140 | 0.3265 | 0.182 | 0.305 |
| A2 S3 | 0.3864 | 0.7487 | 0.1061 | 0.3582 | 0.2758 | 0.0862 | 0.130 | 0.3123 | 0.173 | 0.277 |
| A2 − A1 | +0.0110 | −0.0026 | +0.0090 | +0.0138 | −0.0004 | +0.0174 | −0.010 | −0.014 | −0.008 | −0.028 |

## Primary metrics (mean over 3 independent evaluator runs)

| arm | Coverage | Supported | Unsupported | Uncertain |
|---|---:|---:|---:|---:|
| A1 S3 | 0.3761 | 0.7441 | 0.0986 | 0.1573 |
| A2 S3 | 0.3852 | 0.7292 | 0.1054 | 0.1654 |
| A2 − A1 | +0.0091 | −0.0149 | +0.0068 | +0.0081 |

## Paired Coverage bootstrap (A2 − A1, 982 common images)

| protocol | mean Δ | 95% CI | lower > 0? |
|---|---:|---:|---|
| single run (normalized) | +0.0111 | [−0.0012, +0.0241] | no |
| single run (raw 0–5) | +0.0557 | [−0.0061, +0.1204] | no |
| 3-run per-image mean (normalized) | +0.0091 | [−0.0023, +0.0202] | **no** |
| 3-run per-image mean (raw 0–5) | +0.0455 | [−0.0115, +0.1012] | no |

Per-image direction (3-run mean): Coverage up 281 / flat 465 / down 236;
Supported up 122 / down 122 (single-run flip counts).

## Phase-1 gate

| condition | threshold | result | pass |
|---|---|---:|---|
| Coverage paired 95% CI lower | > 0 | −0.0023 | NO |
| Supported drop | ≤ 0.01 | −0.0149 | NO |
| Unsupported rise | ≤ 0.01 | +0.0068 | yes |
| duplicate-rate increase | none material | A2 0.130 < A1 0.140 | yes |

**Gate: FAIL.** Phase 2 (D1-hard-A2 / D1-soft-A2) and Phase 3 (G2 RL) were not run.

## Not run

| arm | status |
|---|---|
| D1-hard-A2 | NOT RUN (Phase-1 gate failed) |
| D1-soft-A2 | NOT RUN (Phase-1 gate failed) |
| G2 from soft A2 | NOT RUN (Phase-1 gate failed) |
| A0 G2 1000-step rerun | scripted (patience guard added + unit-tested); job not launched |

Evaluator repeat noise: support-label flips 53/72 (A1) and 96/84 (A2) per 1000 vs
run 1; SeeQ score-vector flips 452/470 (A1) and 461/482 (A2). Coverage itself is
stable across runs (±0.0015) but Supported swings by up to ±0.03.
