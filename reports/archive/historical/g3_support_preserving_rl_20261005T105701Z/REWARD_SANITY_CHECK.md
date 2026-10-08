# REWARD SANITY CHECK — read-only, existing A0 oracle labels

Source: `reports/a0_candidate_oracle_20261004T080307Z/candidate_external_scores.jsonl`
(3811 candidates / 1000 images; local frozen Flat + UVFC + external SeeQ/GPT-4o labels).
No external evaluator call. No new data.

## Gate result: **True**

| check | value | pass |
|---|---|---|
| mean support reward: supported > unsupported | 0.4754 > -0.3997 | True |
| support reward vs external support correlation > 0.1 | Spearman 0.417 | True |
| UVFC vs external Coverage correlation > 0 | Spearman 0.320 | True |

## 1. Flat support separation (support_continuous by external label)

| external support | n | mean support_continuous |
|---|---:|---:|
| supported | 2622 | 0.4754 |
| uncertain | 588 | 0.0111 |
| unsupported | 598 | -0.3997 |

## 2. Correlations

- support_continuous vs external Supported: Pearson 0.341, Spearman 0.417.
- UVFC vs external Coverage: Pearson 0.383, Spearman 0.320.
- G3 reward vs external Supported Spearman 0.229; vs external Coverage Spearman -0.020.

## 3. Component distribution (3811 candidates)

| component | mean | std | p25 | p75 |
|---|---:|---:|---:|---:|
| support_continuous | 0.267 | 0.972 | -0.500 | 1.000 |
| coverage_anchor | -0.000 | 0.757 | -1.000 | 1.000 |
| duplicate_penalty | 0.000 | 0.000 | 0.000 | 0.000 |
| generic_penalty | 0.006 | 0.079 | 0.000 | 0.000 |
| g3_reward | 0.266 | 0.982 | -0.567 | 1.067 |

## 4. Degeneration bias

- generic-question rate in A0: **0.0063**; generic questions in the top-100 reward: **0**.
- flat_state counts: {'NO_CONFIRMED_VIOLATION': 2312, 'CONFIRMED_VIOLATION': 1274, 'UNCERTAIN': 150, 'FAILED': 75}.
- Example generic questions flagged: ['Is this a color photo?', 'What is in the picture?', 'Is this photo in color?', 'What is in the picture?', 'What are these?', 'What is this?', 'What is in the picture?', 'What are these?', 'What color is the photo?', 'What color is the photo?'].

The support term separates the three external classes monotonically, and the
composite reward is not dominated by generic/short templates. **Proceed to
training.**
