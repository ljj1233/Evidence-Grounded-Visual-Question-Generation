# GQA-Native Data Contract (Phase G-NATIVE-0)

Status: **FROZEN** (seed 42). Data only — no model was trained or evaluated.
This is a fully independent second-dataset experiment, **not** VQAv2→GQA
zero-shot transfer.

Research question: *Does the same proposed VQG method remain effective when
trained and evaluated natively on a second dataset, GQA?*

Creation script: `scripts/vvqg/freeze_gqa_native.py`. All artifacts under
`outputs/gqa_native/manifests/`. The VQAv2 train, development1000, final_holdout,
and Phase4/5 artifacts are unchanged; `docs/CROSS_DATASET_DATA_CONTRACT.md` and
`outputs/data_contract/` are untouched.

Only `image` and `question text` are used. `answer`, scene graph, functional /
semantic program, object annotation, bounding boxes, and official object
features are never model inputs. Training target is `Image -> Question`.

## 1. Source

Official GQA v1.2 balanced questions (SHA256 of the parsed JSON):

| File | SHA256 |
|---|---|
| `train_balanced_questions.json` | `c600b8301c67ee3eac1b4b9af8a35f776589f045f245fd5b15c50c97fa244d6b` |
| `val_balanced_questions.json` | `2f675643c0cfffe0485dbae0c0c0dfc16b547443309a98544502bc1ae87c0779` |
| `test_balanced_questions.json` | `1121fbbe11b04d3f2afe8049dadb61799a3517dc062bc8c69ee0d9b1f534c3d7` |

Images: `data/raw/gqa/images_pool/` (official GQA v1.2).

## 2. Frozen files

| File | Role | Records | Unique images | SHA256 |
|---|---|---:|---:|---|
| `sft_train_50000.jsonl` | GQA-SFT-50K (`image_id,image_path,question`) | 50000 | 32139 | `ed20f7b3d2ae53f03a7a17b189e9523d78479d4065bd9cab3cf5d6b0b2dc03b2` |
| `train_exposure_images.jsonl` | `GQA_TRAIN_EXPOSURE` | 32139 | 32139 | `5a750fb5c2ba3ccb4bb6a9eac5c551d0f9e4954c14449da8c7ce08be79ce2abf` |
| `method_train_10000.jsonl` | shared method-training images | 10000 | 10000 | `ec9a70ac051b4b1618c134604c6a9e5fe669719e213b33b8a50c2dc87ce0d67d` |
| `dev1000_images.jsonl` | GQA DEV generation manifest | 1000 | 1000 | `9e4c9d8263adf6ae714d8d8c5e4f85a25c1e0a124dbe7df35c050e6e0dbd957d` |
| `dev1000_refs.json` | evaluator-only references | 1000 | — | `e40e59a0eb3e022797cc92d842b8281e03246607c6b9ffe376585637bb71495c` |
| `sanity50_images.jsonl` | pipeline-sanity manifest | 50 | 50 | `214280d937fd506df91901a3000dcd6b7d29d0654cdcc139b6e884efd3b90bff` |
| `sanity50_refs.json` | evaluator-only references | 50 | — | `54489de73b32696255acc953cb35e31a467d192b89e19c7d1f18a0f4406d25eb` |
| `test1000_images.jsonl` | read-only G0 test copy | 1000 | 1000 | `b74f849943f9865286ab3b0ea0ba10bfc1e2653bf895cb07f1a346a41310eb80` |
| `test1000_refs.json` | read-only G0 test refs copy | 1000 | — | `79547802a40069e343ab42cb3b0324963e286d376a84861c298ea8716f8c59b5` |
| `split_image_hash_collisions.json` | recorded decoded-identity collisions | — | — | — |
| `data_contract_manifest.json` | full manifest | — | — | — |

Generation manifests expose only `{image_id, image_path, source, split}` (or
`{image_id, image_path, question}` for SFT). The `*_refs.json` files are
evaluator-only; generation code must never read them.

## 3. Sampling

- **GQA-SFT-50K**: all official `train_balanced` question-image pairs, in
  question-id order, `random.Random(42).sample(..., 50000)`. **No**
  rebalancing by question/answer/semantic/structural type — the official
  balanced distribution is used as-is. Integrity asserted: image exists,
  question non-empty, image_id valid, no val/test image in train.
- **GQA_TRAIN_EXPOSURE**: unique images of the 50K (32139). Teacher generation,
  distillation, Static UVFC GRPO, and Ours GRPO may only use these images; no
  extra GQA train image may be introduced, so SFT/Static/Ours share the same
  image permission.
- **method_train_10000**: exposure >= 10000, so seed-42 sample of exactly 10000
  unique images; no duplicate image padding. Static UVFC, QSE candidate
  generation, teacher construction, distillation source images, and Ours GRPO
  all use this one image manifest (distillation targets may differ).
- **DEV1000**: `val_balanced`, grouped `image_id -> all reference questions`,
  seed-42 sample of 1000 unique images; no question-type/category filtering.
  Usable for GQA method development, ablation, selector analysis, checkpoint
  diagnosis.
- **SANITY50**: fresh seed-42 sample of 50 val images, image-disjoint from
  DEV1000; pipeline debugging only.
- **TEST1000**: reuses the frozen G0 `gqa_generalization1000_*` files
  byte-for-byte. Verified source `test_balanced_questions.json`, 1000 unique
  images, seed-42 frozen sample (reconstructed and matched). Never used for
  hyperparameter/selector/checkpoint/prompt selection; runnable only after the
  GQA method is frozen.

## 4. Split integrity and leakage

Official split image-id intersections: `train∩val = 0`, `train∩test = 0`,
`val∩test = 0`. Selected-set intersections: exposure∩dev = 0, exposure∩test = 0,
dev∩sanity = 0, dev∩test = 0.

Decoded-identity check (exact file SHA256 and normalized-RGB SHA256 over all
selected images): **no cross-split duplicates**. One within-train duplicate pair
was recorded and left unfixed:
`2345907` and `2391510` are byte-identical / decoded-identical (500x374,
201814 bytes), both in `train_balanced`, neither in `method_train_10000` nor in
dev/sanity/test. This is a duplicate in GQA's own train pool and does not affect
split leakage. The official split is not modified.

## 5. Training routes (frozen contract)

| Route | Definition |
|---|---|
| GQA Zero-shot | `Qwen3-VL-8B-Instruct` -> DEV / TEST (no training) |
| GQA SFT | `Qwen3-VL-8B-Instruct` -> GQA-SFT-50K |
| GQA Static UVFC | GQA-SFT -> same Static UVFC algorithm, same `method_train_10000` |
| GQA Ours | GQA-SFT -> Question Space Expansion -> Evidence-Constrained Selection -> Distillation -> UVFC |

GQA SFT reuses the VQAv2 SFT setting as far as possible (LoRA rank/alpha/
dropout, optimizer, learning rate, batch, epochs, generation prompt format);
only the training dataset changes. The GQA Ours structure reuses the final VQAv2
method; no GQA-specific module, scene graph, relation supervision, or program
supervision is added.

## 6. Hyperparameter policy

Goal is method generality, so reuse the final frozen VQAv2 hyperparameters
(candidate K, sampling temperature, selector, support constraint, distillation
recipe, GRPO reward, GRPO steps, relative guard). No large GQA sweep. If a
dataset-dependent parameter cannot be reused directly, only a minimal GQA-DEV
calibration is allowed and must be documented explicitly.

## 7. External baselines

LOVA3 and Genixer are external released-model baselines requiring no GQA
retraining; they run inference-only on GQA DEV/TEST and must not be mixed into
same-backbone controlled claims.

## 8. Reporting

Future GQA main table: `BLEU, CIDEr, Search, Coverage, Context, Spatial,
Supported, Unsupported, Dup, Dist-2` for Base / GQA-SFT / GQA-Static UVFC /
GQA-Ours Distilled / GQA-Ours + UVFC. The primary check is
`GQA-Ours` vs `GQA-Static UVFC`: Search up, Coverage up, with Supported
maintained/raised and Unsupported maintained/lowered.

Cross-dataset reading compares **method deltas within each dataset** (VQAv2
`Ours - Static` and GQA `Ours - Static`), never absolute BLEU/CIDEr across
datasets.

## 9. Compute

Any later model run: `OMP/MKL/OPENBLAS/NUMEXPR/RAYON_NUM_THREADS=8`,
`TOKENIZERS_PARALLELISM=false`, vLLM via `taskset -c 0-7`, DataLoader workers
<= 4.

## 10. Reproduction

```
.venv-vllm-cu128-v011/bin/python scripts/vvqg/freeze_gqa_native.py
```

Deterministic (seed 42); re-running reproduces all file SHA256 above.
