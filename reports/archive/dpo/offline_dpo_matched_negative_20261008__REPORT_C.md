# PART C — Matched-negative dataset construction + probes

Frozen rule (written before probing): keep the original chosen; for the rejected,
among that image's **Flat-unsupported candidates of the same broad type**, pick
deterministically by (same interrogative head, smallest token-length gap, highest
pool-TFIDF cosine, question string). No dev result used. No training, no API.

## Dataset

| quantity | value |
|---|---:|
| original pairs | 6633 |
| matched images **|M|** | **2733** (41.2%) |
| type histogram | yes/no 1630, what 794, count 241, color 68 |
| CONTROL type-match rate (original rejected) | 0.780 |
| TREATMENT type-match rate (matched rejected) | **1.000** |
| CONTROL mean abs token-length gap | 1.480 |
| TREATMENT mean abs token-length gap | 1.397 |
| CONTROL mean TF-IDF cosine (chosen↔rejected) | 0.115 |
| TREATMENT mean TF-IDF cosine | 0.132 |

CONTROL and TREATMENT have identical images, identical chosen, identical size; only
the rejected text changes.

## Text-only shortcut probes (image-grouped 5-fold pairwise ranking)

| probe | CONTROL | TREATMENT | Δ |
|---|---:|---:|---:|
| NB bag-of-words | 0.690 | **0.620** | −0.070 |
| TF-IDF + logistic regression | 0.715 | **0.623** | −0.092 |

Per-fold TREATMENT: NB 0.612/0.634/0.592/0.621/0.641; TF-IDF LR 0.642/0.634/0.585/0.625/0.630.

## Conclusion

Type-matched negatives **reduce but do not remove** the text-only separability
(~0.62 residual, vs 0.69–0.71 CONTROL; earlier full-set same-type 0.629). The residual
is within-type lexical/phrasing, not just type mismatch. |M| = 2733 pairs — enough for
the controlled training in Part D. Proceeding.
