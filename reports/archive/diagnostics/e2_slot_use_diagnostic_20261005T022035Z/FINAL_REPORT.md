# FINAL REPORT — E2 slot-use diagnostic

Run: `reports/e2_slot_use_diagnostic_20261005T022035Z/`
Status: COMPLETE, read-only. Q1 FAIL, Q2 FAIL, Q3 FAIL.
Conclusion: **v2 removes numerical slot collapse but does not establish
image-grounded slot semantics or decoder reliance.**

See `AUDIT.md`, `CONFIG.md`, `DIAGNOSTIC_REPORT.md`, `DECISION.md`,
`correspondence/correspondence.json`, `decoder/decoder_diagnostics.json`,
`analysis/analysis.json`.

## Retrieval (256 images, 962 questions; chance R@1 0.0039, R@5 0.0195)

| agg | R@1 | R@5 | MRR |
|---|---:|---:|---:|
| max | 0.0052 | 0.0239 | 0.0256 |
| mean | 0.0021 | 0.0146 | 0.0203 |
| top-2 | 0.0042 | 0.0166 | 0.0234 |

Correct 0.0678 vs wrong 0.0679 (Δ +0.00008, CI [−0.0011,+0.0012]); mean 0.0140;
random 0.0493. `margin_mean` −0.0001 [−0.0008,+0.0007]; `margin_max` −0.0394
[−0.0406,−0.0381]. Hungarian gap +0.0575 (not independent). Permutation delta 0.

## Decoder-use

gate 0.00975, out_proj norm 4.59, residual/input ratio mean 4.5e-4 (layer 0
1.1e-3 → layer 34 3.8e-5). NLL/table and deltas in `DIAGNOSTIC_REPORT.md`.

## Question change

| pair | exact | NED | overlap | type-change |
|---|---:|---:|---:|---:|
| normal/off | 0.863 | 0.069 | 0.914 | 0.051 |
| normal/wrong | 0.883 | 0.063 | 0.923 | 0.043 |
| normal/mean | 0.906 | 0.055 | 0.935 | 0.043 |
| normal/random | 0.883 | 0.064 | 0.922 | 0.047 |
| c0/e2-normal | 0.766 | 0.126 | 0.850 | 0.082 |

195/256 images identical across all conditions. 0 empty; distinct 0.78–0.80.

## Sample cases

normal/off (20):
100234 boats/boats · 10040 people in photo/people wearing boots · 100990 donut
shop/restaurant · 101567 dog/dog · 10211 giraffes/giraffes · 102503 pillows/
pillows · 102594 teddy color/teddy color · 103017 time/time · 103379 scissors/
scissors · 105052 glasses/glasses · 105455 time/time · 10591 happy/happy ·
106392 kitchen/kitchen · 107020 cake design/cake design · 108253 green thing/
green thing · 108531 mirror/mirror · 109118 city/city · 11034 holding/holding ·
111031 black shirt/black shirt · 111367 cows/cows

normal/wrong (20): identical for all 20 except 106392 (kitchen / is there a TV in
the kitchen); the rest reuse the normal/off list above.

normal/mean (20): identical except 10040 (people in photo/people wearing boots),
106392 (kitchen/is there a TV in the kitchen), 109118 (men/players).

C0/E2 (20): 100234 same; 10040 boots/people in photo; 100990 restaurant/donut
shop; 103017 wall/time; 10591 sport/happy; others same as listed.

No-difference images (all 5 conditions identical), 20 of 195:
100234, 101567, 10211, 102503, 102594, 103017, 103379, 105052, 105455, 107020,
108253, 108531, 11034, 111031, 111367, 112269, 112702, 113097, 113570, 117095.

Full per-condition files: `analysis/cases_*.jsonl`.

## Permissions

Modify model / retrain / RL: NO for this task. Recommended next: (1) fix the
question-encoder anisotropy (whitening/L2-norm/decorrelation) so assignment has a
meaningful signal; (2) increase the evidence path strength (gate/out_proj init or
a residual scale) so the decoder can use the slots; (3) re-run this exact
diagnostic before any retraining.
