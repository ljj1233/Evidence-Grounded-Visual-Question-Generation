# QUESTION SPECIFICITY 1+4 AUDIT v1 (Phase 2.2C, revised)

> **No final Question–Image Specificity metric or training reward is selected.**
> This phase evaluates a cheaper protocol — 1 source image + 4 CLIP hard negatives (5 images) —
> and checks whether it preserves the Phase 2.2B scientific conclusions.

## Scope

- Candidate set: **1 source image I+ + top-4 CLIP hard negatives I- (5 images)** taken from the existing
  `outputs/question_specificity_v1/hard_negative_manifest.jsonl`. Hard negatives were **not recomputed**.
- Systems (requested): `sft_baseline` (baseline SFT) and `vanilla` (standard / ordinary DPO) **only**.
  `mdpo`, `pdpo`, `random` are out of scope for this run.
- Full evaluation: 2000 judgments (2 systems × 1000 images); prompt version `specificity_1plus4_v1`.
- The `sft_baseline` row is included at explicit request; Metric Bake-off v1 previously flagged that training run as
  incomplete, so its numbers should be read with that caveat.

## Protocol

- For each question the source image and its 4 nearest CLIP neighbours are shown in **randomized order**; the source index is hidden.
- GPT-4o returns the same structured setwise output used in Phase 2.2B:
  `applicable_images` (indices providing sufficient visible evidence), `best_matching_image` (index or null),
  `unique_best_match` (bool), `reason`.
- Decoded outcomes: `source_win` = best match is the source; `negative_win` = best match is a hard negative;
  `tie` = no best match returned. Applicability is not judged by word overlap.
- `pairwise portability` = P(at least one hard negative applicable | source applicable); lower means stronger source specificity.
- Source-image `visual_support` (Phase 2.2A) is kept separate and neither recomputed nor merged.

### Randomization / cache policy

- Display order randomized per (seed, system, image_id); source position histogram: {'1': 399, '2': 399, '3': 405, '4': 412, '5': 385}.
- Cache key = sha256(question | ordered candidate image ids | judge model | prompt version); incremental JSONL append;
  resume, exponential backoff, timeout / rate-limit handling, separate failed-item log, no recomputation of successful calls.
  A distinct prompt version isolates this protocol from the Phase 2.2B cache.

## Table 1 — Protocol validation against the existing setwise audit

Validation subset: 50 source images per system. Agreement is only meaningful for systems present in Phase 2.2B
(so `sft_baseline` is n/a). Protocols differ (4 vs 9 negatives), so equality is not expected.

| System | n | Source-applicability agreement | Source-best-match agreement |
|---|---:|---:|---:|
| overall | 49 | 0.878 | 0.796 |
| sft_baseline | n/a | n/a | n/a |
| vanilla | 49 | 0.878 | 0.796 |

JSON validity: 100/100 (1.000); API failures: 0.

## Table 2 — Main metrics by system

| System | N | Source Applicable [95% CI] | Hard-Neg Applicable [95% CI] | Mean applicable-neg fraction | Source Win [95% CI] | Tie [95% CI] | Negative Win [95% CI] | Unique Source Best-Match | Pairwise Portability [95% CI] | Supported Source-Win (diagnostic) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sft_baseline | 1000 | 0.784 [0.758, 0.810] | 0.498 [0.468, 0.528] | 0.193 | 0.617 [0.586, 0.646] | 0.043 [0.032, 0.056] | 0.340 [0.310, 0.370] | 0.617 | 0.372 [0.341, 0.406] | 0.617 [0.586, 0.646] |
| vanilla | 1000 | 0.925 [0.908, 0.941] | 0.601 [0.570, 0.631] | 0.313 | 0.631 [0.602, 0.660] | 0.062 [0.048, 0.077] | 0.307 [0.280, 0.334] | 0.631 | 0.571 [0.538, 0.602] | 0.631 [0.602, 0.660] |

`Hard-Neg Applicable` = P(at least one of the 4 hard negatives is applicable). `Pairwise Portability` and the
`Supported Source-Win` joint rate are **diagnostics only** and are not combined into any single final score.

## Table 3 — Paired system comparison

Only one comparison is in scope: `sft_baseline` vs `vanilla`. Paired bootstrap by `image_id` (resamples = 1000).

| Metric | Difference (sft_baseline − vanilla) [95% CI] |
|---|---:|
| Source applicable | -0.141 [-0.173, -0.112] |
| Hard-neg applicable | -0.103 [-0.142, -0.067] |
| Mean applicable-neg fraction | -0.120 [-0.144, -0.099] |
| Source win | -0.014 [-0.052, 0.026] |
| Tie | -0.019 [-0.037, 0.001] |
| Negative win | 0.033 [-0.005, 0.068] |
| Unique source best-match | -0.014 [-0.052, 0.026] |
| Pairwise portability | -0.198 [-0.240, -0.159] |
| Supported source-win | -0.014 [-0.052, 0.026] |

_No winner is named when the interval overlaps 0 substantially._

## Table 4 — Display-order consistency audit

50 source images per system were re-judged with a different randomized display order. Reordered calls are excluded
from the main metric denominator. This is an order-robustness check, not an accuracy measure.

| System | n | Outcome consistency | Source-applicability consistency | Source-best-match consistency | Unique-best consistency |
|---|---:|---:|---:|---:|---:|
| overall | 100 | 0.840 | 0.930 | 0.870 | 0.870 |
| sft_baseline | 50 | 0.900 | 0.880 | 0.920 | 0.920 |
| vanilla | 50 | 0.780 | 0.980 | 0.820 | 0.820 |

## Table 5 — Comparison with existing automatic metrics and the 10-image setwise audit

Existing BLEU-4 / CIDEr / Q-BLEU4 / Distinct-2 / duplicate rate / CLIPScore (Metric Bake-off v1) and CLIP retrieval
R@1 / R@5 / MRR (Phase 2.2B) are reused, not recomputed. No combined score is formed.

| System | BLEU-4 | CIDEr | Q-BLEU4 | Distinct-2 | Dup rate | CLIPScore | CLIP R@1 | CLIP R@5 | CLIP MRR | 1+4 source win | 10-img best-match | 1+4 mean neg fraction | 10-img mean neg fraction | 1+4 portability (any of 4) | 10-img any-neg (of 9) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sft_baseline | 0.1690 | 0.1868 | 0.1644 | 0.1231 | 0.4630 | 0.589 | n/a | n/a | n/a | 0.617 | n/a | 0.193 | n/a | 0.372 | n/a |
| vanilla | 0.2187 | 0.3562 | 0.2130 | 0.2097 | 0.2910 | 0.646 | 0.353 | 0.841 | 0.553 | 0.631 | 0.580 | 0.313 | 0.221 | 0.571 | 0.635 |

_The two portability definitions differ: 1+4 P(any of 4 negatives applicable | source applicable) and 10-image P(any of 9 negatives applicable). `mean neg fraction` is the mean per-question applicable-negative share and is the closest apples-to-apples quantity across the two protocols. `sft_baseline` has no Phase 2.2B setwise row (it was excluded there); its CLIP retrieval row is also absent, so those cells are n/a._
_The two systems are **not** separated by unconditional source-win (1+4 spread 0.014; the paired CI includes 0). However `sft_baseline` has substantially lower source applicability (spread 0.141) and lower conditional pairwise portability than `vanilla`. Lower portability is nominally more source-specific, but it comes with far more unsupported source questions, so no system is declared better here._

## Table 6 — Runtime / API throughput

| Quantity | Validation | Full evaluation | Reorder audit |
|---|---:|---:|---:|
| API requests | 100 | 2000 | 100 |
| Images per request | 5 | 5 | 5 |
| Concurrency | 32 | 32 | 32 |
| New API calls | 100 | 2000 | 100 |
| Cache hits | 0 | 0 | 0 |
| Retries | 0 | 0 | 0 |
| Failures | 0 | 0 | 0 |
| Wall-clock (s) | 23.52 | 350.65 | 23.6 |

- Total API calls: 2200; total wall-clock: 397.77 s; requests/sec (new calls): 5.531.
- Images per question: **5** (this protocol) vs **10** (Phase 2.2B setwise); ~50% fewer images per question.
- Phase 2.2B setwise ran 800 ten-image calls in 188.97 s on 800 questions; this run judged 2000 questions with 5-image calls.

## Limitations

- The 4 hard negatives are CLIP-image-similarity neighbours, not question-relevant counterfactuals; some may be easy or accidentally applicable.
- The GPT-4o judge is an automated anchor, not human annotation.
- `sft_baseline` was previously flagged incomplete-training; treat its row as diagnostic.
- Only 2 systems and 4 negatives are evaluated here; no system is declared best unless the paired CI supports it.
- Pairwise portability is a diagnostic, not a final metric.

## Outputs

`outputs/question_specificity_1plus4_v1/`: `candidate_sets_manifest.jsonl`, `validation_judgments.jsonl`,
`validation_agreement.json`, `pairwise_judgments.jsonl`, `oneplusfour_summary.json`, `paired_system_analysis.json`,
`order_swap_judgments.jsonl`, `order_consistency.json`, `runtime_report.json`.

## Closing statement

No final Question–Image Specificity metric or training reward is selected in Phase 2.2C.
This phase evaluates whether a cheaper 1-source + 4-hard-negative GPT-4o protocol can stand in for the more
expensive ten-image setwise audit while preserving the same scientific conclusions.
