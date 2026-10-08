# FINAL REPORT — Offline DPO pilot (Flat support preference)

Low-cost offline DPO pilot. No GRPO/IPO/DAPO, no external API during training, no `final_holdout`.
Base/reference = original Full Ours checkpoint; Support-Delta step300 not used.

## Headline

- **STATUS: FAIL_PILOT**
- Supported improved strongly (+0.0703, 95% CI [+0.0078, +0.1406]) and Coverage was maintained
  (+0.0234, CI [−0.0094, +0.0578], lower bound within the −0.01 margin).
- **Unsupported increased by +0.0156 (> +0.01), which fails the pre-registered gate.**
- The Supported gain is mostly Uncertain→Supported (Uncertain −0.0859).
- `NEXT_ACTION: STOP` (no expansion to 256).

## Data construction

- Reused the existing prompt-aligned Direct-4 candidate pool
  (`outputs/ec_qse_aligned_direct4/dev/candidates_scored.jsonl`, instruction
  "Generate a question about this image."). The aligned train2014 pool had only 64 images
  (< the 256 required), so the 1000-image dev pool was used with an **image-level** split.
- Pairs = 706 (from 1000 images; 294 images lacked both a supported and an unsupported candidate).
  chosen = Flat `NO_CONFIRMED_VIOLATION`; rejected = Flat `CONFIRMED_VIOLATION`; uncertain /
  parser-failure / malformed / duplicate questions skipped; UVFC used only for tie-break.
- Split by `image_id` hash: train 578, pilot 128, zero image overlap.

## Training

- LoRA-only DPO from Full Ours `step_final` (`52ee692c…`), β=0.1, 1 epoch, seed 42, batch 1, lr 5e-6.
- 300 optimizer steps; checkpoints step100/200/300. Reference frozen.
- Final preference accuracy 1.0; final DPO loss 0.609; chosen/ref KL ≈ 0.166; duplicate rate 0.
- `final_holdout` not accessed; no external API used in training.

## External evaluation (single pass)

- Base and DPO generated greedily (T=0, max_new_tokens=32, seed 42) on the fixed 128 pilot images.
- SeeQ Coverage once; GPT-4o Supported/Uncertain/Unsupported once. 0 parser failures.
- API calls: GPT-4o 181 + SeeQ 256 = 437.

| metric | n | Base | DPO | delta | 95% CI |
|---|---:|---:|---:|---:|---|
| coverage | 128 | 0.3906 | 0.4141 | +0.0234 | [−0.0094, +0.0578] |
| supported | 128 | 0.8203 | 0.8906 | +0.0703 | [+0.0078, +0.1406] |
| unsupported | 128 | 0.0234 | 0.0391 | +0.0156 | [−0.0156, +0.0469] |
| uncertain | 128 | 0.1563 | 0.0703 | −0.0859 | [−0.1484, −0.0234] |

## Gate

| condition | value | result |
|---|---|---|
| Coverage delta CI lower ≥ −0.01 | −0.0094 | PASS (marginal) |
| Supported delta mean > 0 | +0.0703 | PASS |
| Unsupported delta mean ≤ +0.01 | +0.0156 | **FAIL** |

→ **FAIL_PILOT**; expansion to 256 images is forbidden.

## INTERPRETATION

- Offline DPO on Flat support preferences produces a large Supported gain and keeps Coverage
  within the exploratory margin, but the Supported gain comes mostly from converting Uncertain
  cases while Unsupported edges up (3→5 images). Under the pre-registered rule this is not a clean
  pass; the pilot is not expanded.
- 3→5 Unsupported images is small (CI crosses 0), but the gate is defined on the mean, so the
  conservative verdict is FAIL.

## UNKNOWN

- Whether the Unsupported increase is real or noise (single evaluator pass; CI includes 0).
- Whether the Supported gain generalizes beyond the pilot 128 or to held-out data.
- Whether the effect is due to DPO or to question-length/template shifts (not ablated).

Not a final/paper result; no novelty or grounding claim.
