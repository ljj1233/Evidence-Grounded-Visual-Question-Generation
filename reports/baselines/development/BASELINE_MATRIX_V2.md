# Baseline Matrix v2

All direct rows use the same 1,000-image development manifest, frozen greedy decoding, and the protocol in `docs/EVALUATION_PROTOCOL_V1.md`. The SeeQ columns are the released SeeQ single-question rubric applied with the documented `gpt-5.6-luna` judge substitution; they are not paper-exact SeeQ scores (the paper uses GPT-5.4 and five questions per image).

## Table A — fair/direct baselines

| System | BLEU-4 | CIDEr | Q-BLEU variant | Search | Evidence coverage | Context | Spatial | Supported | Unsupported | Distinct-2 | Dup | Semantic Div. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot Qwen3-VL-8B | 0.089809 | 0.126362 | 0.087013 | 0.367968 | 0.402202 | 0.365966 | 0.124725 | 0.634000 | 0.230000 | 0.276117 | 0.121000 | 0.393691 |
| Final SFT Qwen3-VL-8B | 0.317225 | 0.454971 | 0.310714 | 0.274600 | 0.259800 | 0.270400 | 0.073200 | 0.725000 | 0.159000 | 0.139524 | 0.570000 | 0.536019 |

The 200-question Luna pilot is in `outputs/phase24/evaluation/seeq_luna/pilot200_summary.json`; the full 1,000+1,000 scores and failure counts are in `outputs/phase24/evaluation/seeq_luna/summary.json`. No master score is defined.

## Table B — external methods and input-setting status

| Method | Classification | Checkpoint | Required inputs / reason excluded from Table A |
|---|---|---|---|
| InfoMax VQG | DIFFERENT_INPUT_SETTING | unavailable | image plus answer/category; no released checkpoint |
| ConVQG | DIFFERENT_INPUT_SETTING | unavailable | text-guided and knowledge-enriched input; no frozen checkpoint |
| SeeQ method | UNAVAILABLE | unavailable by design | self-evolution/RAG method checkpoint and generated outputs are not released |
| Guiding VQG | UNAVAILABLE | unavailable | explicit variant needs answer/category and concept guidance; no checkpoint |
| Double Hints | DIFFERENT_INPUT_SETTING | unavailable | answer hint plus visual ROI hint |
| ED-VQG | DIFFERENT_INPUT_SETTING | unavailable | scene graph and selected subgraph/region conditioning |

Judge robustness is documented in docs/SEEQ_JUDGE_ROBUSTNESS.md; GPT-4o is a secondary audit and is not averaged with Luna. The released SeeQ evaluator is available. Its component status is: rubric/evaluator `AVAILABLE_WITH_JUDGE_SUBSTITUTION` (GPT-5.6 Luna), diversity evaluator `AVAILABLE_WITH_CARDINALITY_CAVEAT` using the downloaded Qwen3-Embedding-4B weights, and trained method `UNAVAILABLE`. Evidence and exact URLs are in `outputs/phase24/external_baselines/feasibility.json`.

## Table C — visual support and visual-centric capability

| System | Supported | Unsupported | Uncertain | Luna pilot Search | Luna pilot Coverage | Luna pilot Context | Luna pilot Spatial |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot Qwen3-VL-8B | 0.634 | 0.230 | 0.136 | 0.367968 | 0.402202 | 0.365966 | 0.124725 |
| Final SFT Qwen3-VL-8B | 0.725 | 0.159 | 0.116 | 0.284000 | 0.256000 | 0.262000 | 0.060000 |

Pilot scores are descriptive validation, with raw 0–5 scores retained. Table A reports the full normalized means and semantic within-image diversity. Parser/API failure counts are in the Luna summary; diversity standard deviations and bootstrap intervals are in `outputs/phase24/evaluation/seeq_diversity/summary.json`.
