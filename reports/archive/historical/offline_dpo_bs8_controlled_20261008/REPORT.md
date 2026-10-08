# Controlled batch-8 DPO — report

Strict control. Re-initialized from Base `step_final`; same frozen 6633 S>X pairs;
beta 0.1, lr 5e-6, constant LR, same LoRA/optimizer/seed/reference; **only change:
effective batch size = 8** (batch_size=8, no gradient accumulation). One full pass.
Checkpoints 100/200/400/600/800 + final. Reference sanity passed
(initial policy ≡ reference: max|margin|=0, loss=ln2; `REFERENCE_SANITY.json`).
No training API. DEV256 identical to Task 1.

Training: 830 optimizer steps, 6640 pair exposures (7 tail repeats; the last batch
pads with 7 already-seen pairs), final loss 0.161, final pref-acc 1.0, **0% steps
with loss>5**, peak 73.5 GB, 1432 s. No hard stop triggered.

## Comparison (DEV256, local metrics only — no API for bs8 tonight)

| system | updates | Flat sup (ΔCI) | what | count+yn | BLEU-4 | CIDEr | len | type H | dup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 0 | 0.749 | 0.254 | 0.668 | 0.234 | 0.433 | 5.89 | 1.84 | 0.188 |
| formal step1000 | 1000 | 0.802 (+.054 [−.012,+.120]) | 0.098 | 0.805 | 0.182 | 0.360 | 6.57 | 1.13 | 0.289 |
| formal final | 6633 | 0.885 (+.139 [.074,.205]) | 0.000 | 0.945 | 0.132 | 0.320 | 7.18 | 1.25 | 0.160 |
| bs8_0100 | 100 | 0.785 (+.033 ns) | 0.258 | 0.652 | 0.216 | 0.424 | 6.08 | 1.86 | 0.215 |
| bs8_0200 | 200 | 0.772 (+.020 ns) | 0.203 | 0.672 | 0.216 | 0.415 | 6.18 | 1.81 | 0.219 |
| bs8_0400 | 400 | 0.778 (+.029 ns) | 0.137 | 0.723 | 0.204 | 0.392 | 6.33 | 1.42 | 0.262 |
| bs8_0600 | 600 | 0.786 (+.037 ns) | 0.106 | 0.770 | 0.194 | 0.374 | 6.36 | 1.21 | 0.273 |
| bs8_final | 830 | 0.762 (+.012 ns) | 0.102 | 0.809 | 0.183 | 0.367 | 6.54 | 1.14 | 0.258 |

(formal rows are the Task-1 trajectory on the same DEV256; bs8 Flat supported CIs
all cross 0. GPT-4o support was **not** run for bs8.)

## Read against H1 / H2

- **Over-update (H1) is real.** The bs=1 run's coverage/search peak at step1000–2000
  and erode to ns by final, with mass-collapse (chosen log-ratio −8.7 nats/token).
  Batch-8 keeps the optimizer healthy (`chosen` log-ratio −1.89/token, rejected
  −5.42, 0% loss>5) and the endpoint is far less collapsed: what-rate 0.102 vs 0.000,
  BLEU-4 0.183 vs 0.132, length 6.54 vs 7.18.
- **Type/lexical migration (H2) is not removed by batch-8.** Even here, what-rate
  falls 0.254 → 0.102 and count+yes/no rises 0.668 → 0.809 across the single epoch;
  at comparable update counts bs8_final ≈ formal step1000 in type composition
  (0.102 vs 0.098), i.e. the shift is driven by the objective/data, not the batch
  size. The same-type audit (Task 4) shows matching the broad type only lowers the
  text-only shortcut 0.706 → 0.629.
- Flat supported gain under bs8 is small and CI-crossing-zero (≤ +0.037); whether the
  independent GPT-4o support gain survives batch-8 must be checked with 1–2 API
  checkpoints in the morning (not done tonight, by design).

## Note on exposure accounting

Pair exposures 6640 vs 6633 unique pairs (7 tail repeats) because `ceil(6633/8)=830`.
Optimizer steps 830, consumed pairs 6640. Reported, not hidden.
