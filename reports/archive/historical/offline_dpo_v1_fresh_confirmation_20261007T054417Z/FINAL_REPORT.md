# FINAL REPORT — DPO-v1 fresh confirmation (fresh294)

Read-only evaluation. No training, no `final_holdout`, no Risk-aware DPO-v2, no IPO/GRPO/DAPO.

## Headline

- **STATUS: DPO_V1_FRESH_CONFIRMATION_PASS** (all 7 pre-registered conditions met).
- On 294 images excluded from DPO-v1 training and from the old 128 pilot:
  Supported **+0.0442** (95% CI [+0.0136, +0.0782]), Coverage **+0.0245** (CI [+0.0041, +0.0442]),
  SeeQ Search **+0.0381** (CI [+0.0204, +0.0565]), Unsupported **−0.0136** (CI [−0.0306, +0.0000]),
  Uncertain **−0.0306**.
- The fresh294 result is directionally consistent with the old 128 pilot.

## FACT

Splits (`SPLIT_AUDIT.json`): pool 1000 = 578 DPO-v1 train + 128 old pilot + **294 fresh**; fresh overlaps
0 with train, 0 with old pilot, 0 with final_holdout (`fresh_size_ok = true`).

Generation: Base and DPO-v1 on all 294 fresh images, greedy T=0, max_new_tokens 32, seed 42, paired by
image_id, 0 failures, 0 empty. Checkpoints: Base `52ee692c…`, DPO-v1 `385f3514…`.

Evaluation: one pass, SeeQ (`gpt-5.6-luna`) + GPT-4o support; 0 parser failures; valid_n 294 for both
systems and both evaluators; 984 new API calls.

Paired (image-level bootstrap 5000, seed 42):

| metric | base | DPO | delta | 95% CI | up/down/tied |
|---|---:|---:|---:|---|---|
| coverage | 0.3735 | 0.3980 | +0.0245 | [+0.0041, +0.0442] | 33/21/234 |
| search | 0.3503 | 0.3884 | +0.0381 | [+0.0204, +0.0565] | 43/17/234 |
| supported | 0.8503 | 0.8946 | +0.0442 | [+0.0136, +0.0782] | 19/6/269 |
| unsupported | 0.0374 | 0.0238 | −0.0136 | [−0.0306, +0.0000] | 1/5/288 |
| uncertain | 0.1122 | 0.0816 | −0.0306 | [−0.0612, +0.0000] | 6/15/273 |

Transition matrix (Base→DPO): supported→supported 244, supported→uncertain 5, supported→unsupported 1,
uncertain→supported 15, uncertain→uncertain 18, uncertain→unsupported 0, unsupported→supported 4,
unsupported→uncertain 1, unsupported→unsupported 6. Coverage rises most for uncertain→supported (+0.107,
n=15); the one supported→unsupported case drops −0.20.

Question stats: changed_rate 0.347, exact_match 0.653, length +0.29 tokens (7.05→7.34), duplicate
0.190→0.204, generic 0.0→0.0.

Old 128 pilot (historical, separate; not pooled): Supported +0.0703, Coverage +0.0234, Unsupported +0.0156.

## INTERPRETATION

- DPO-v1's Supported gain **reproduces** on images never used in training or in the original pilot, with
  a positive Supported CI and a matching Coverage gain (within the exploratory margin, in fact positive).
- Unsupported is controlled (−0.0136, CI upper 0.0), unlike the old 128 pilot (+0.0156). The gain is
  mostly Uncertain→Supported (15 cases), consistent with raising support-label confidence rather than
  removing violations.
- Directions agree with the old 128 pilot for Supported and Coverage; Unsupported is better here.
- The confirmation set is a held-out subset of the aligned pool, not a fully independent test set; this
  is not a final generalization claim.

## UNKNOWN

- Whether the effect holds on a truly independent distribution (only one holdout exists and was not read).
- Whether the effect is causal vs a judge/length artifact (no ablation; length changed only +0.29 tokens).

## Answers

1. Supported gain reproduces on fresh294: **yes** (+0.0442, CI lower +0.0136 > 0).
2. Coverage maintained/improved: **yes** (+0.0245, CI lower +0.0041).
3. Unsupported controlled: **yes** (−0.0136, CI upper 0.0 ≤ +0.01).
4. fresh294 vs old128 direction: **consistent** (Supported and Coverage positive in both; Unsupported better on fresh294).
5. Expand DPO-v1 to full dev1000: **allowed** (fresh confirmation passes), but dev1000 itself is NOT an
   independent set because 578 images were used in training; expansion was not started.
6. No follow-up training started.
