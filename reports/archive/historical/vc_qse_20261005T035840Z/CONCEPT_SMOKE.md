# CONCEPT SMOKE — 64 development images

Run: `reports/vc_qse_20261005T035840Z/` · seed 42 · GPU 0 · greedy concept proposal
Model: frozen `/hpc2hdd/home/fye374/models/Qwen3-VL-8B-Instruct` (image-only)
Images: `dev64_manifest.jsonl` (development1000 sorted by integer image_id, first 64)

Concept prompt hash: `sha256(PROMPT)` recorded per record; raw responses cached in
`concepts/concepts_{original,blank,shuffled}.jsonl`.

## Results

| check | value | threshold | result |
|---|---|---|---|
| JSON parse success (original) | **1.000** | ≥ 0.95 | PASS |
| concepts / image (mean) | **13.70** | ≥ 2 | PASS |
| empty concept rate | 0.000 | — | PASS |
| within-image raw duplicate rate | 0.000 | — | PASS |
| global duplicate rate | 0.213 | — | — |
| most common concept | `tree` in 9/64 imgs (0.141) | not a fixed template | PASS |
| type distribution | obj 254 / attr 254 / rel 241 / scene 128 | — | PASS |
| confidence mean / min / max | 0.960 / 0.85 / 0.99 | — | — |
| question/answer leakage | 0 | 0 | PASS |
| blank inventory change | **1.000** | > 0.5 | PASS |
| shuffled inventory change | **1.000** | > 0.5 | PASS |
| fallback-marked samples | 0 | recorded | PASS |

Blank (grey) images returned an **empty inventory** `{"objects":[],...}` for all
64 images (valid JSON, `status=empty`, Jaccard 0.0 vs original) — the proposer is
not hallucinating concepts without image content. Shuffled images returned
different inventories (mean Jaccard 0.003) — the proposer reads spatial layout,
not just the prompt.

## Gate

**SMOKE PASS.** Proceeded to concept-conditioned generation.

JSON: `concept_smoke.json`.
