# PILOT 64 — C0_DIRECT4 vs VC-QSE_PILOT (smoke)

Run: `reports/vc_qse_20261005T035840Z/pilot64/` · seed 42 · GPU 0
No training, no RL, no external evaluator (local Flat verifier + Qwen3-VL judge only).

- C0_DIRECT4: same 64 images, K=4, T=0.8, top_p 0.9, SFT checkpoint; final question by
  existing support-first selection (S1 semantics over local Flat support).
- VC-QSE: image-only concepts → up to one concept per type → 1 question each →
  local Flat support + concept adherence → `VC_SELECTOR`.

## Aggregate

| metric | C0_DIRECT4 | VC-QSE_PILOT |
|---|---:|---:|
| images | 64 | 64 |
| candidates | 244 | 256 |
| supported (local Flat) | 0.953 | **0.984** |
| unsupported | 0.031 | **0.000** |
| uncertain | 0.016 | 0.016 |
| fallback rate | 0.000 | 0.031 |
| duplicate rate | 0.031 | 0.063 |
| mean words | 5.55 | 5.56 |
| empty questions | 0 | 0 |

Candidate concept types: object 64 / attribute 64 / relation 64 / scene 64.
Per-image eligible type coverage: object 0.69, attribute 0.45, relation 0.44,
scene 0.50. VC candidate local supported rate 0.629; adherence yes/no/uncertain =
182/73/1; candidate duplicate rate 0.137.

The 64 result shows VC-QSE raises local Flat support without external
evaluation. This is a smoke signal only, **not** a Coverage result.
