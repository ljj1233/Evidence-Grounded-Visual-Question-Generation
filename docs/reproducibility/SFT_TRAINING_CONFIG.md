# SFT TRAINING CONFIG — image-conditioned VQG (Qwen3-VL-8B-Instruct + LoRA)

Phase 1.6. Config: `configs/new_vqg/sft.yaml` · Entry: `src/vvqg/training/train_sft.py`
· Smoke: `scripts/vvqg/smoke_sft.py`. No reward/GRPO/evaluation changes; dataset
format `{image, question}` unchanged.

## Model path

| | |
|---|---|
| Intended (requested) | `~/models/Qwen/Qwen3-VL-8B-Instruct` |
| **Actually present / used** | `~/models/Qwen3-VL-8B-Instruct` |
| `model_type` | `qwen3_vl` |
| architecture | `Qwen3VLForConditionalGeneration` |
| size on disk | 17 GB (4 safetensors shards) |

> **Path mismatch (recorded).** The requested `~/models/Qwen/Qwen3-VL-8B-Instruct`
> does not exist on this host; `~/models/Qwen/` holds no Qwen3-VL. The checkpoint
> present is `~/models/Qwen3-VL-8B-Instruct`. The config points at the real path
> directly — **no symlink**. This is the multimodal Qwen3-VL model, **not** the
> text-only `~/models/Qwen3-8B`.

## LoRA

```yaml
lora:
  enabled: true
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

Smoke run: **252 modules matched** by the adapter. Precision: `bf16: true`.

## Batch configuration

```yaml
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  effective_batch_size: 16      # 2 x 8 x 1 GPU
  num_train_epochs: 2
  learning_rate: 2.0e-4
  warmup_ratio: 0.03            # applied as warmup_steps = round(total_steps * 0.03)
  logging_steps: 10
  save_steps: 500
  save_total_limit: 3
  dataloader_num_workers: 8
  gradient_checkpointing: true
  attn_implementation: flash_attention_2
```

Sequence length: `max_seq_length: 512`. Image: fed through the Qwen3-VL
processor; labels are masked to the assistant answer only.

## GPU assumption

- **A800 80GB** (verified: `NVIDIA A800-SXM4-80GB`, 81,920 MiB).
- Smoke peak memory: **allocated 19.9 GB**, **reserved 28.2 GB** (bf16 8B + LoRA
  + batch 2, gradient checkpointing). Fits comfortably; ~50 GB headroom.

## Smoke test result

Command: `.venv/bin/python3 scripts/vvqg/smoke_sft.py` (1000 samples, 20 steps).
Report: `checkpoint/sft_smoke/smoke_report.json`. **Status: PASS.**

| Check | Result |
|---|---|
| Image processor loaded | `Qwen3VLProcessor` (image_processor loaded) |
| Forward batch keys | `input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw`, `mm_token_type_ids`, `labels` |
| `pixel_values` | shape `[2640, 1536]`, non-zero |
| `image_grid_thw` | shape `[2, 3]`, values `[[1,30,40],[1,36,40]]` |
| **Model consumes image** | loss with image `3.969` vs zeroed image `4.517`, Δ `0.548` → **True** |
| Loss behavior | first `3.295` → last `1.372` (min `1.259`) → **decreased** |
| GPU memory | allocated `19.889 GB`, reserved `28.242 GB` |
| Generation (image only) | `COCO_...509482.jpg` → *"What is the cake on?"*; `COCO_...535138.jpg` → *"What is the boy wearing on his head?"* (differs across images) |

The image-consumption check is explicit: the same batch is forwarded once with
the real `pixel_values` and once with `pixel_values` zeroed; a non-trivial loss
delta proves the vision tensors influence the model (the run would be rejected
otherwise).

## Full training command (NOT run in this phase)

```bash
.venv/bin/python3 src/vvqg/training/train_sft.py --train
```

- Trains on `src/vvqg/data/sft/train.jsonl` (50,000 samples), 2 epochs,
  effective batch 16, LoRA output to `checkpoint/sft/` (`checkpoint-final`).
- Dry-run only (no model load): `... train_sft.py --dry-run`.
- Smoke only: `... scripts/vvqg/smoke_sft.py` (default 1000 samples / 20 steps).

## Environment notes

- `transformers 5.17.0`: `TrainingArguments` has **no `warmup_ratio`** (removed);
  warmup is applied via computed `warmup_steps`. `Trainer` takes
  `processing_class=processor` (not `tokenizer`).
- `flash_attention_2` is available and used; the loader falls back to `sdpa` on
  failure. Manual forwards must move the model/batch to CUDA (flash-attn is
  CUDA-only).
- Model loads with `local_files_only=True` (offline; no downloads).
