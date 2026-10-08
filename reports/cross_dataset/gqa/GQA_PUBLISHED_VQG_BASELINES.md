# GQA Published VQG Baselines (Phase G1)

Status: **GQA PUBLISHED BASELINES BLOCKED** — all five rows ran on the frozen
GQA DEV1000; the mandatory 1000/1000 API-validity gate is not met (see §12).

Second-dataset (GQA) baseline phase of the GQA-native experiment, not
VQAv2→GQA zero-shot transfer. No proposed-method training was run.
`test1000_*` never read; `final_holdout` untouched.

## 1. Purpose

Establish the published VQG baseline landscape natively on GQA:
same-backbone trainable reproductions **A1 SeeQ**, **B2 ConVQG contrastive
grounding**, **B3 Guiding-VQG implicit guided**; and inference-only external
checkpoints **LOVA3 GenQA**, **Genixer-L Common-VQA**.

## 2. Frozen GQA data contract

`docs/GQA_NATIVE_DATA_CONTRACT.md` / `outputs/gqa_native/manifests/`. B2/B3 use
`method_train_10000.jsonl` (10000 shared method images) and the GQA records
filtered to those images (`published_baselines/records_method10000.jsonl`,
15569 records, SHA256 `37076a3e…`). DEV = `dev1000_images.jsonl` +
`dev1000_refs.json`; SANITY = `sanity50_*`. Generation reads only `*_images.jsonl`.

## 3. Baseline categories

| row | category | GQA training? |
|---|---|---|
| A1 SeeQ-style Reproduction | same-backbone adaptation of a published algorithm | yes |
| B2 ConVQG-inspired Contrastive Grounding | same-backbone adaptation of a published algorithm | yes |
| B3 Guiding-VQG Implicit Guided | same-backbone adaptation of a published algorithm | yes |
| LOVA3 GenQA (released external checkpoint) | external | no |
| Genixer-L Common-VQA (released external checkpoint) | external | no |

## 4. A1 — Faithful SeeQ-style GQA reproduction

M0 = base Qwen3-VL-8B-Instruct as a **zero-init LoRA** (`B=0`, adapter `seed_m0`
SHA256 `1a736f3f…`); no VQAv2-SFT/GQA-SFT init. Multi-intent proposal K=10,
frozen-M0 rewrite/filter, dual-format QG+QA SFT, M0→M1→M2.

| round | proposals | rewrites | accepted | acceptance |
|---|---:|---:|---:|---:|
| R1 | 99559 | 99558 | 1392 | 1.40% |
| R2 | 98728 | 98609 | 1427 | 1.45% |

M1 = 2784 dual records (1392 steps), M2 = 2854 (1427 steps). Acceptance is ~10x
lower than VQAv2's 13.6% because the frozen rewriter/filter is the base model.

Filter throughput was optimized without changing the rubric/prompt logic (merge
the three image-conditioned judge prompts into one vLLM call; merge seed+rewritten
scoring; backend image-preprocess cache; batch 128→256): 2.9 → 5.2 cand/s probe
(~1.26x in the real R2 run). vLLM greedy is **not bit-reproducible across batch
compositions** (same input: accepted 0 vs 2), so R1 (original grouping) and R2
(optimized grouping) are samples of the same method.

## 5. B2 — ConVQG-inspired Contrastive Grounding (GQA-native)

Frozen captions `t=Cap(I)` and image-only/text-only negative questions from the
base Qwen3-VL; Qwen3-Embedding-4B fixed `z_gt/z_img/z_txt`; trainable
LoRA+projection; alpha 0.2, margin 0.5, dual triplet + CE. LoRA r16/α32/drop0.05,
2 epochs, effective batch 16, bf16. Artifacts `published_baselines/b2_contrastive/`
(`cache/{captions,q_img,q_txt}.jsonl`, `cache/embeddings.json`, `full_fast/`
checkpoint-1948, `greedy1000.jsonl`, `sampled4x1000.jsonl`). Caption/negative cache
covers method_train_10000 + dev1000 (11000 images).

## 6. B3 — Guiding-VQG Implicit Guided (GQA-native)

Frozen detector cache (torchvision COCO Faster R-CNN ResNet50-FPN-V2 top-36
regions/boxes) over method_train_10000 (+ dev1000); guidance categories derived
from **GQA train answers** via the IQ answer-category map (training-time guidance
only, never an inference input); object targets by Qwen3-Embedding-4B question/
answer concept alignment. Same LoRA/training recipe as B2 (2 epochs, 1948 steps).
Artifacts `published_baselines/b3_implicit/` (`cache/{detector,detector_dev,
train_targets}.jsonl`, `full_fast/` checkpoint-1948 + `guided_head.pt`,
`greedy1000.jsonl`, `sampled4x1000.jsonl`). Targets manifest:
`cache/train_targets.manifest.json`.

## 7. LOVA3 / Genixer (external)

LOVA3 `hhenryz/LOVA3-llava-v1.5-7b` @ `0a7c184c…`, greedy, canonical GenQA
instruction. Genixer `Anonymous-G/Genixer-llava-v1.5-7b` @ `9401b250…`,
`--task-type common`, instruction + `"\nThis is the Common VQA task."`, released
`T=0.2` primary. Contracts in `<row>/inference_contract.json`.

## 8. SANITY integrity

SANITY50 (seed-42 GQA val): all rows 50/50 predictions, 50 unique IDs, 0 empty,
0 malformed.

## 9. DEV1000 main table

`outputs/gqa_native/published_baselines/comparison/table.json`. Corpus
BLEU-4/CIDEr/Q-BLEU4-v/Distinct-2/Dup; image-mean SeeQ dims and GPT-4o support.

| Method | BLEU-4 | CIDEr | Q-BLEU4-v | Search | Coverage | Context | Spatial | Supported | Unsupported | Uncertain | Dist-2 | Dup | SemDiv |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 Faithful SeeQ-style Reproduction | 0.0220 | 0.0015 | 0.0188 | 0.4771 | 0.5307 | 0.1796 | 0.3594 | 0.5267* | 0.2730* | 0.2002* | 0.2989 | 0.000 | 0.3315 |
| B2 ConVQG Contrastive Grounding | 0.4055 | 0.2368 | 0.3937 | 0.3698 | 0.3790 | 0.2060 | 0.2873 | 0.7110 | 0.2670 | 0.0220 | 0.0656 | 0.452 | 0.5210 |
| B3 Guiding-VQG Implicit Guided | 0.4293 | 0.2474 | 0.4177 | 0.3558 | 0.3656 | 0.2058 | 0.2428 | 0.7848 | 0.1852 | 0.0300 | 0.0724 | 0.444 | 0.5485 |
| LOVA3 GenQA | 0.3406 | 0.1404 | 0.3311 | 0.3280 | 0.3474 | 0.2848 | 0.1434 | 0.7750 | 0.1191 | 0.1062 | 0.0927 | 0.535 | 0.5290 |
| Genixer Common-VQA | 0.3928 | 0.1821 | 0.3827 | 0.3542 | 0.3506 | 0.2448 | 0.1844 | 0.7928 | 0.1502 | 0.0571 | 0.1397 | 0.349 | 0.4302 |

`*` SeeQ support over the 879/1000 labelable items (incomplete; §12). Facts only,
no winner claim. External rows are released LLaVA-v1.5-7B checkpoints, not
same-backbone controlled.

## 10. Sampled-4 diversity (`T=0.8, p=0.9, K=4, seed 42`)

| System | SemDiv (Qwen3-Embedding-4B) | unique/4 |
|---|---:|---:|
| A1 SeeQ-style Reproduction | 0.3315 | 1.000 |
| B2 ConVQG Contrastive Grounding | 0.5210 | 0.995 |
| B3 Guiding-VQG Implicit Guided | 0.5485 | 0.994 |
| LOVA3 GenQA | 0.5290 | 0.968 |
| Genixer Common-VQA | 0.4302 | 0.974 |

## 11. Cross-dataset descriptive comparison

`comparison/completeness.json → cross_dataset_descriptive` (VQAv2 vs GQA) for A1/
LOVA3/Genixer. B2/B3 VQAv2 rows exist only as traditional metrics (PHASE24C), so
they are excluded from the SeeQ/support cross table. Search/Coverage etc. are
comparable only within a dataset; no cross-dataset absolute-BLEU claim.

## 12. Completeness

| System | generation | SeeQ valid | Support valid | residual | complete |
|---|---|---|---|---|---|
| seeq (A1) | 1000/1000 | 999/1000 | 879/1000 | 121 | no |
| b2 | 1000/1000 | 999/1000 | 1000/1000 | 0 | no (1 SeeQ) |
| b3 | 1000/1000 | 999/1000 | 999/1000 | 1 | no |
| lova3 | 1000/1000 | 1000/1000 | 1000/1000 | 0 | **yes** |
| genixer | 1000/1000 | 999/1000 | 999/1000 | 1 | no |

Residual causes (all retried; none recovered): deterministic **gpt-4o
content-policy refusals** (`content: null`) — 121 SeeQ + 1 Genixer + 1 B3 — and
one **HTTP 400** on image `2351296` for the gpt-5.6-luna SeeQ rubric (seeq/b2/b3/
genixer each lose 1). Not pipeline bugs; the contract forbids zero-filling or
shrinking the denominator, so the phase is **BLOCKED** rather than COMPLETE.

## 13. Reproduce

`outputs/gqa_native/published_baselines/reproduce.sh` (project root; one GPU;
`taskset -c 0-7`, `OMP=8`). Chains: `scripts/vvqg/gqa_native_seeq_chain.sh`,
`scripts/vvqg/gqa_baselines_chain.sh`, `scripts/vvqg/gqa_b2b3_chain.sh`;
evaluation `scripts/vvqg/gqa_evaluate.py`; B3 targets
`scripts/vvqg/gqa_prepare_b3_targets.py`; table
`scripts/vvqg/gqa_baseline_table.py`. Logs under
`outputs/gqa_native/published_baselines/{seeq/logs,logs}/`.
