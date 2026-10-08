# Published VLM Question-Generation Baselines (Phase 5B)

Inference-only evaluation of two released published VLM question-generation
checkpoints on the frozen `development1000` image set, using the same canonical
metrics as SFT / Static UVFC / Phase 4A. No training, no finetuning, no
checkpoint modification, no GT answer/reference-question input, no
`final_holdout`.

## 1. Systems and paper-safe labels

| row | label |
|---|---|
| A | LOVA3 GenQA (released LLaVA-v1.5-7B checkpoint) |
| B | Genixer-L Common-VQA Generator (released LLaVA-v1.5-7B checkpoint) |

Both are external released-model baselines (category B). They are **not**
retrained on our 50k SFT split; this is intentional. They are kept separate from
the controlled same-backbone comparisons (category A). Both backbones are
LLaVA-v1.5-7B, but we do **not** claim same-backbone control over our Qwen3-VL
pipeline. Genixer is used as its released **data generator**, not as the
downstream `llava-Genixer-915K-FT-8K-v1.5-7b` answering model, and is not
labeled a "VQAv2 model".

## 2. Provenance

| item | value |
|---|---|
| HF endpoint | `https://hf-mirror.com` |
| LOVA3 repo | `hhenryz/LOVA3-llava-v1.5-7b` @ `0a7c184c5d37112f038d3ebb3c612036fa6d7991` |
| Genixer repo | `Anonymous-G/Genixer-llava-v1.5-7b` @ `9401b250a704a0806618d508664f24d69057dda4` |
| shared vision tower | `openai/clip-vit-large-patch14-336` @ `ce19dc912ca5cd21c8a653c79e251e808ccabcd1` (reused, already local) |
| local paths | `~/models/{LOVA3,Genixer}-llava-v1.5-7b`, `~/models/clip-vit-large-patch14-336` |
| official code | `showlab/LOVA3` @ `ef9f98ec19159bc4c0979682296eff2d338c667`; `zhaohengyuan1/Genixer` @ `22e9c0ad833e160ec05f15f4071b2fec15bd9c30` |
| code location | `third_party/LOVA3`, `third_party/Genixer` (read-only; adapters under `scripts/vvqg/`) |
| envs | `.venv_lova3`, `.venv_genixer` (isolated; reuse project torch via `.pth`, `transformers==4.31.0` matching the checkpoint's native LLaVA code) |
| hashes | `outputs/published_vqg_baselines/download_manifest.json` |

Both repos' `llava` packages are byte-identical; each branch still runs its own
repo root and environment. The checkpoints store the frozen CLIP vision tower,
which the official `load_pretrained_model` path reloads from CLIP-336 (expected;
tower was frozen during LLaVA-1.5 training).

## 3. Recovered inference contracts

Full records: `lova3/inference_contract.json`, `genixer/inference_contract.json`.

- **LOVA3 GenQA.** The released mixed training data
  (`hhenryz/Mixed_VQA_GenQA_EvalQA_1.5M`) formats GenQA as
  `<image>\n<instruction>` → `Question: ...\nAnswer: ...`, where the instruction
  is drawn from the 58 generic question-and-answer generation instructions
  documented by Genixer (`GenQA_general_instructions.json`). Verified by
  inspecting GenQA records in the training jsonl. Conversation mode
  `vicuna_v1` (LOVA3's own eval scripts use `vicuna_v1` and temperature 0).
  Primary row: greedy, `num_beams=1`, `max_new_tokens=1024`, canonical fixed
  instruction `Please provide a clear and direct question and answer after
  examining the image.`
- **Genixer-L Common-VQA.** Official `model_genixer_eval.py --task-type common`
  does `qs = <generic instruction> + "\nThis is the Common VQA task."` with
  `vicuna_v1`, `max_new_tokens=1024`, and `temperature=0.2` (sampling) in
  `scripts/eval_genixer/specific_common_generation.sh`. The referenced
  `GenQA_short.json` is not shipped; the documented released equivalent
  `GenQA_general_instructions.json` (58 instructions) is used, entries stripped
  of the inline `<image>` token because the official path prepends
  `DEFAULT_IMAGE_TOKEN`. The per-image instruction is fixed to the same canonical
  instruction for reproducibility; decoding keeps the official `T=0.2`.
- **Parser (both).** Extract the text after the first `Question:` up to
  `\nAnswer:`; take the first line ending in `?`; strip `Question:`/`ASSISTANT:`
  wrappers and `</s>`. Deterministic formatting only; question content is never
  rewritten. Full raw `Question+Answer` is preserved in `raw_outputs.jsonl`.

## 4. Main comparison (development1000, `comparison/table.json`)

Corpus BLEU-4/CIDEr/Q-BLEU4-v/Distinct-2/Dup; image-mean SeeQ dims and GPT-4o
support rates. No ranking and no winner is claimed.

| Method | BLEU-4 | CIDEr | Q-BLEU4-v | Search | Coverage | Context | Spatial | Supported | Unsupported | Uncertain | Distinct-2 | Dup | SemDiv |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT | 0.317 | 0.455 | 0.311 | 0.275 | 0.260 | 0.270 | 0.073 | 0.725 | 0.159 | 0.116 | 0.140 | 0.570 | 0.536 |
| GPT-5.6-Luna Direct | 0.080 | 0.119 | 0.077 | 0.421 | 0.445 | 0.271 | 0.257 | 0.930 | 0.036 | 0.034 | 0.352 | 0.006 | — |
| Prompt-CoT | 0.074 | 0.073 | 0.071 | 0.503 | 0.467 | 0.249 | 0.223 | 0.883 | 0.069 | 0.048 | 0.263 | 0.021 | — |
| Faithful SeeQ M2 | 0.166 | 0.292 | 0.158 | 0.371 | 0.381 | 0.348 | 0.179 | 0.664 | 0.010 | 0.326 | 0.213 | 0.177 | 0.520 |
| LOVA3 GenQA | 0.139 | 0.204 | 0.135 | 0.328 | 0.351 | 0.290 | 0.127 | 0.737 | 0.128 | 0.135 | 0.100 | 0.514 | 0.529 |
| Genixer Common-VQA | 0.181 | 0.327 | 0.176 | 0.342 | 0.345 | 0.251 | 0.154 | 0.771 | 0.143 | 0.086 | 0.139 | 0.371 | 0.448 |
| Static UVFC GRPO | 0.326 | 0.519 | 0.319 | 0.332 | 0.332 | 0.255 | 0.071 | 0.814 | 0.094 | 0.092 | 0.174 | 0.499 | 0.537 |

`—` = no 4-question within-image embedding-diversity artifact exists for that
system (Luna/Prompt-CoT were generated greedy-only); reported as null in
`table.json` and counted in `completeness.json`.

The two published systems land mid-table on the visual-centric dimensions:
LOVA3 `0.328/0.351/0.290/0.127`, Genixer `0.342/0.345/0.251/0.154` (Search/
Coverage/Context/Spatial), both below the strongest same-backbone rows on
Search/Coverage but with higher Context/Spatial than SFT. Absolute values are
not comparable to the same-backbone rows as a controlled effect.

## 5. Integrity and completeness

- Both: exactly 1000 greedy predictions, 1000 unique image IDs, image-ID set
  **equal** to `development1000` (`manifest_sha256 06782bed…`, identical to
  SFT/Luna/Prompt-CoT/Static UVFC).
- empty = 0, malformed = 0; parser audit over all 1000 and 50 inspected
  examples: 0 missing questions, 0 multiple questions, 0 answer leakage,
  0 wrapper contamination (`parser_audit.json`).
- SeeQ-rubric GPT-5.6-Luna: **1000/1000** valid for both (failures retried,
  never zeroed). GPT-4o visual support: **1000/1000** valid for both.
- `completeness.json` records per-row present/13 measurements.

## 6. Sampled-4 diversity (`sampled4x.jsonl`, project canonical T=0.8 / p=0.9)

Greedy outputs were not overwritten.

| System | sampled Distinct-2 | sampled Dup | Qwen3-Embedding-4B SemDiv | unique/4 |
|---|---:|---:|---:|---:|
| LOVA3 GenQA | 0.157 | 0.262 | 0.529 | 0.962 |
| Genixer Common-VQA | 0.164 | 0.270 | 0.448 | 0.972 |

Diversity summary: `comparison/diversity/summary.json` (seed 42, 5000-resample
image bootstrap).

## 7. Fairness, provenance and limitations

- No VQAv2 reference question, answer, or development annotation is used as
  model input; input is image + the official method-specific generation
  instruction only. No `final_holdout`. No checkpoint was modified.
- Genixer's primary row inherits the official `T=0.2`; LOVA3's primary row is
  greedy (its released eval uses temperature 0). This decoding difference follows
  each official release and is disclosed.
- The Genixer instruction list is the documented released equivalent of the
  unshipped `GenQA_short.json`; the LOVA3 GenQA instruction family was verified
  against its released training data. Exact per-image instruction sampling in the
  official Genixer eval is unseeded; we fix one canonical instruction for
  reproducibility.
- These rows answer "how do released published VLM question-generation systems
  perform when directly evaluated on our common image set and evaluator", not a
  controlled same-backbone comparison.

## 8. Reproduce

```bash
# generation (one checkpoint per process; official LLaVA code path)
cd third_party/LOVA3 && .venv_lova3/bin/python scripts/vvqg/published_vqg_inference.py \
  --method lova3 --repo <repo> --model-path ~/models/LOVA3-llava-v1.5-7b \
  --manifest outputs/same_backbone_baselines/manifests/development1000_images.jsonl \
  --out outputs/published_vqg_baselines/lova3 --schema greedy --instruction "<canonical>"
# metrics (same implementation as SFT / Static UVFC / Phase 4A)
.venv-vllm-cu128-v011/bin/python scripts/vvqg/phase5a_evaluate.py \
  --predictions outputs/published_vqg_baselines/lova3/predictions.jsonl --system lova3 \
  --out outputs/published_vqg_baselines/lova3/evaluation --concurrency 16
# assemble table
.venv/bin/python scripts/vvqg/published_vqg_compare.py
```
