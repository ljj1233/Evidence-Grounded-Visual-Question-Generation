# DATA PROTOCOL — new VQG (reward-driven)

Status: **protocol + schema only**. No data was downloaded, no API was called, no
sample was generated, no training was implemented. This document freezes the
three data formats used by the reward-driven VQG pipeline.

## Frozen task

| Item | Value |
|---|---|
| Input | `image` (reference to an image; not embedded bytes) |
| Output | `question` |
| Base model | Qwen3-8B (`~/models/Qwen3-8B`) |
| Future training | Stage 1: SFT · Stage 2: GRPO |
| Reward directions | 1. question validity · 2. information usefulness · 3. visual grounding |

## Format

All datasets are **JSONL** (UTF-8, one JSON object per line, blank lines skipped).
Every record is validated against a frozen JSON Schema file:

```text
src/vvqg/data/sft/schema.json
src/vvqg/data/calibration/schema.json
src/vvqg/data/rollout/schema.json
```

Constraints shared by all records:

- `image` is a **path/reference string** (`minLength: 1`), never base64/bytes.
- Text fields are non-empty strings.
- **Unknown fields are rejected** (`additionalProperties: false`); the schema is
  exact. Extra metadata must live in a sidecar, not inline.
- Scores are finite JSON numbers (JSON cannot represent `NaN`/`Infinity`).

---

## 1. SFT dataset

Path: `src/vvqg/data/sft/` · Schema: `schema.json`
用途: 监督微调 (Stage 1).

```json
{
  "image": "...",
  "question": "..."
}
```

| Field | Type | Required | Constraint | Meaning |
|---|---|---|---|---|
| `image` | string | yes | minLength 1 | Input image reference |
| `question` | string | yes | minLength 1 | Target question for the image |

**Phase 1 (VQAv2) status**

- purpose: **train basic question generation**.
- size: **50k initial** (train); plus 2k dev and 2k test for internal sanity.
- **not used: evaluation** — the SFT splits are not an evaluation benchmark; the
  reserved eval pool is VQAv2 `val2014` (+ frozen `val_debug`).
- forbidden fields: `answer`, `rationale`, `reward`, `evidence`. The inference
  stage only generates a `question`, so the schema allows exactly
  `{image, question}` (`additionalProperties: false`).
- source: VQAv2 `train2014` only; splits are **image-disjoint**.

## 2. Reward calibration dataset

Path: `src/vvqg/data/calibration/` · Schema: `schema.json`
用途: 验证 reward.

```json
{
  "image": "...",
  "question": "...",
  "validity_label": 0.0,
  "grounding_label": 0.0,
  "information_gain_label": 0.0
}
```

| Field | Type | Required | Constraint | Meaning |
|---|---|---|---|---|
| `image` | string | yes | minLength 1 | Input image reference |
| `question` | string | yes | minLength 1 | Question under calibration |
| `validity_label` | number | yes | 0 ≤ x ≤ 1 | Question validity score (reward direction 1) |
| `grounding_label` | number | yes | 0 ≤ x ≤ 1 | Visual grounding score (reward direction 3) |
| `information_gain_label` | number | yes | 0 ≤ x ≤ 1 | Information usefulness/gain (reward direction 2) |

Mapping to reward directions: `validity_label` → question validity,
`information_gain_label` → information usefulness, `grounding_label` → visual
grounding. Labels are normalized to `[0, 1]`.

## 3. RL rollout dataset

Path: `src/vvqg/data/rollout/` · Schema: `schema.json`
用途: GRPO 缓存.

```json
{
  "image": "...",
  "prompt": "...",
  "response": "...",
  "reward": 0.0
}
```

| Field | Type | Required | Constraint | Meaning |
|---|---|---|---|---|
| `image` | string | yes | minLength 1 | Input image reference |
| `prompt` | string | yes | minLength 1 | Image-conditioned prompt given to the policy |
| `response` | string | yes | minLength 1 | Sampled policy response (generated question) |
| `reward` | number | yes | any finite number | Scalar reward for this response |

`reward` is a scalar (the aggregation of the three reward directions is a
training-time concern, not part of this transport schema).

---

## Directory layout

```text
src/vvqg/data/
├── sft/
│   ├── schema.json
│   ├── train.jsonl / train_manifest.jsonl   # 50,000 samples
│   ├── dev.jsonl   / dev_manifest.jsonl     # 2,000 samples
│   ├── test.jsonl  / test_manifest.jsonl    # 2,000 samples
│   └── dataset_manifest.json                # provenance (seed, sources, hashes)
├── calibration/schema.json
└── rollout/schema.json
```

`calibration/` and `rollout/` currently contain **only** the schema files.

Manifest row schema (one row per image): `{image_id, image_path, split}`.
Manifests are metadata: training data discovery ignores `*_manifest.jsonl`.

## Validation

```bash
# SFT dataset + manifests (schema, image existence, split, no eval leakage)
python scripts/vvqg/validate_sft_dataset.py

# Generic per-record schema check
python scripts/new_vqg/check_dataset.py --kind sft         --path <file.jsonl>
python scripts/new_vqg/check_dataset.py --kind calibration --path <file.jsonl>
python scripts/new_vqg/check_dataset.py --kind rollout     --path <file.jsonl>
```

- Schema defaults to `src/vvqg/data/<kind>/schema.json`; override with
  `--schema`.
- `--max-errors N` caps printed errors (default 50).
- Exit codes: `0` all valid · `1` validation errors · `2` usage/IO error.
- Validation is stdlib-only (no external dependency).

## Non-goals

- No reward labels, no preference pairs, no API calls.
- No SFT/GRPO training.
- No preference-pair / DPO training data is produced or consumed by this protocol.

SFT data is prepared locally from existing VQAv2 raw files; nothing is downloaded
and no annotation (`answer`) is used.
