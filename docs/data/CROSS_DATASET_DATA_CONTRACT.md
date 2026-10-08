# Cross-Dataset Generalization Data Contract (Phase G0)

Status: **FROZEN** (2026-10-04). No model was run, trained, or evaluated in this
phase. `final_holdout` was not accessed.

This contract freezes the data roles for the paper claim: 

```
VQAv2-derived training
  -> method development on the existing development1000
  -> frozen zero-shot cross-dataset evaluation on GQA
```

Creation script: `scripts/vvqg/freeze_data_contract.py` (seed 42).
All artifacts live in `outputs/data_contract/`.

## 1. Roles

| Role | Dataset | Split | Images | Used for weights? | Used for method selection? | References visible to generator? |
|---|---|---|---:|---|---|---|
| TRAIN | VQAv2-derived | frozen project train | 9184 | Yes | No | training only |
| DEV | VQAv2-derived | development1000 | 1000 | No | Yes | evaluator only |
| SANITY | GQA | balanced val | 50 | No | pipeline only | No |
| CROSS-TEST | GQA | balanced test | 1000 | No | No | No |
| FINAL | existing final_holdout | protected | UNKNOWN | No | No | untouched |

### TRAIN — VQAv2-derived training exposure (weights)

`outputs/data_contract/train_exposure_images.jsonl` — 9184 unique images, one
row per image, `used_by` merged across the stages that consumed it. All stages
are subsets of the frozen SFT train split (`src/vvqg/data/sft/train_manifest.jsonl`),
verified in the script:

| Stage | Manifest(s) | Images |
|---|---|---:|
| `sft` | `src/vvqg/data/sft/train.jsonl` (50000 records) | 9184 |
| `uvfc` (Phase4B / Static UVFC) | `outputs/phase4b_online_uvfc/data/{train_10000,train_2000}.jsonl` | 9184 / 2000 |
| `teacher_generation` (Phase5B) | `outputs/phase5b_teacher_explore_ground/train_10000.jsonl` | 9184 |
| `distillation` (Phase5B/5C) | `phase5b .../student_train_10000.jsonl`, `phase5c_qse_teacher/targets_d{1,2}_*.jsonl` | 9184 |
| `grpo` | pilot `train_subset_2000.jsonl` + Phase4B/5C online UVFC-GRPO runs on `train_10000` | 9184 |

Record schema: `canonical_image_key` (`coco2014:{image_id}`), `source_dataset`
(`VQAv2`), `source_split` (`train2014`), `image_path`, `image_id`, `used_by`.

No `reward_calibration` entry appears in TRAIN: the canonical reward-calibration
corpus (`outputs/evaluation/sft_greedy/val2014_sample1000.jsonl`) is **exactly
the same 1000 images as development1000** (verified set-equal), so it is
method-selection data (DEV), not weight data. It is nevertheless decontaminated
against as part of the exposure union (TRAIN ∪ DEV).

### DEV — development1000 (method selection)

`outputs/same_backbone_baselines/manifests/development1000_images.jsonl`,
1000 images, SHA256
`06782bedf1754fe59ff7609a06e2c79ce91403e4ecfcf9f2f295d6b104d27d47` — unchanged.
This set was repeatedly used for method selection, prompt diagnosis, reward
analysis, Phase5A/5B, and baseline comparison. **It must never be described as
final test.**

### FINAL — final_holdout

Not accessed in this phase (not listed, not hashed, not compared). The
cross-dataset protocol does not depend on final_holdout content.

## 2. GQA source

`outputs/data_contract/gqa_source_manifest.json` records version 1.2, official
source URLs, archive sizes/SHA256, the parsed question-file SHA256, and local
paths. The image pool is `data/raw/gqa/images_pool/` (148854 jpgs), extracted
from `data/raw/archives/gqa/images.zip`.

Only the **questions** are used, and only as offline reference text. Scene
graphs, object/spatial features, functional programs, and answers are never
model inputs. The image-only inference manifests never contain question, answer,
program, scene-graph, or question-type fields.

### Why no GQA scene graph

The evaluated task is zero-shot cross-dataset **visual question generation**.
The model sees only the image plus the normal VQG generation instruction. GQA
question text is used exclusively for offline reference evaluation.

## 3. Image-level sampling

GQA is question-level. Both splits are converted to an image-level dataset:
`image_id -> all balanced reference questions for that image`; each image appears
once in a generation manifest. Sampling is image-level (never per question row),
seed 42, stable order (numeric id sort), `random.Random(42).sample`.

- `GQA_SANITY_50` — from `val_balanced_questions.json`; pipeline sanity only
  (image loading, ID mapping, generation, parser, evaluator compatibility). Not
  for method/selector/tau/decoding/checkpoint selection.
- `GQA_GENERALIZATION_1000` — from `test_balanced_questions.json`; the formal
  cross-dataset generalization set. Model runs are gated on method freeze.
- The two sets are image-disjoint.

## 4. Overlap / leakage audit

Audited against the project exposure set = **TRAIN_EXPOSURE (9184) ∪ DEV (1000)**.

`source_id_mapping = unavailable`: GQA `imageId` is a Visual Genome id, not a
COCO id. Numeric collisions with project COCO ids were verified by exact and
normalized-RGB hashing to be different images (e.g. all 8 val id collisions had
different dimensions and pHashes). No exclusion is made on raw id overlap.

Three content layers were applied: exact file SHA256, normalized-RGB SHA256
(width + height + raw RGB bytes), and pHash (32x32 grayscale DCT, 8x8 low
frequency) with Hamming distance <= 4, confirmed by a second-level resized-image
correlation (>= 0.90). Only true same/near-identical source images are excluded;
purely semantic similarity is not.

`outputs/data_contract/gqa_overlap_audit.json`:

| Split | Candidates | SHA256 overlap | RGB overlap | near-dup flagged | near-dup excluded | Remaining |
|---|---:|---:|---:|---:|---:|---:|
| test_balanced | 2987 | 0 | 0 | 0 | 0 | 2987 |
| val_balanced | 10234 | 54 | 0 | 413 | 358 | 9822 |

The 412 excluded val images (54 byte-identical + 358 near-identical) are real
COCO/VQAv2 leakage into the GQA val pool; they are removed before sanity
sampling. `outputs/data_contract/gqa_overlap_exclusions.jsonl` records every
pair with `reason` and `distance_or_score`. **The generalization test split has
zero detected overlap** with the project exposure union.

## 5. Reference statistics (descriptive only)

From `outputs/data_contract/data_contract_manifest.json`:

| Set | Images | Total refs | Mean/img | Median/img | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| generalization1000 | 1000 | 31782 | 31.78 | 29 | 2 | 116 |
| sanity50 | 50 | 607 | 12.14 | 9.5 | 1 | 40 |

Sanity (val) structural: choose 77, compare 24, logical 75, query 304, verify 128.
Sanity semantic: attr 208, cat 46, global 26, obj 72, rel 256. GQA test
annotations are hidden, so structural/semantic types are unavailable there.
These statistics are for description only and never used to re-sample.

## 6. Evaluation rules (frozen)

- BLEU / CIDEr / Q-BLEU are comparable **only within the same GQA split**.
  Do **not** state "GQA BLEU is higher/lower than VQAv2 BLEU, therefore
  generalization is better/worse". Cross-dataset reading relies on Search,
  Coverage, Context, Spatial, Supported, Unsupported and each method's change
  relative to the same-split baseline.
- Future GQA evaluation objects, frozen now:
  1. SFT
  2. Static UVFC
  3. QSE Distilled
  4. QSE + UVFC final
  5. LOVA3 GenQA (external released checkpoint)
  6. Genixer Common-VQA (external released checkpoint)
  7. Prompt-CoT (optional inference reference)
  8. GPT-5.6-Luna Direct (optional inference reference)
  The controlled cross-dataset conclusion comes from objects 1-4 only; 5-6 are
  marked external, not same-backbone controlled.
- **Zero GQA in training.** SFT, teacher generation, distillation, GRPO, and
  reward calibration use 0 GQA images/questions/answers/scene graphs.
- Paper-safe wording: "Zero-shot cross-dataset transfer with respect to our VQG
  task training and post-training data." / "No GQA samples are used in our
  task-specific training, distillation, reward calibration, or reinforcement
  learning." Do **not** claim the pretrained VLM never saw GQA/Visual
  Genome/COCO during pretraining.

## 7. Artifacts

| File | Records | SHA256 |
|---|---:|---|
| `train_exposure_images.jsonl` | 9184 | see manifest |
| `gqa_generalization1000_images.jsonl` | 1000 | see manifest |
| `gqa_generalization1000_refs.json` | 1000 | see manifest |
| `gqa_sanity50_images.jsonl` | 50 | see manifest |
| `gqa_sanity50_refs.json` | 50 | see manifest |
| `gqa_overlap_audit.json` | — | see manifest |
| `gqa_overlap_exclusions.jsonl` | 412 | see manifest |
| `gqa_source_manifest.json` | — | see manifest |
| `data_contract_manifest.json` | — | — |

`data_contract_manifest.json` lists every file with path, record count, unique
images, SHA256, creation script, and seed. The generation manifests expose only
`{image_id, image_path, source, split}`; the `*_refs.json` files are
evaluator-only and must never be read by generation code.

Reproduce: `.venv-vllm-cu128-v011/bin/python scripts/vvqg/freeze_data_contract.py`.

## 8. Integrity checks (all pass)

- generalization images == 1000, unique == 1000
- sanity images == 50, unique == 50
- generalization ∩ sanity == empty
- generalization ∩ TRAIN_EXPOSURE == empty; generalization ∩ development1000 == empty
- every selected image file exists and is readable via PIL
- every selected image has >= 1 balanced reference question
- generation manifests contain no question/answer/scene-graph/program/type keys
- `train_10000.jsonl` and `development1000_images.jsonl` SHA256 unchanged
