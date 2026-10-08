# EVALUATION METRIC CONTRACT v1

Status: active. This document is the authoritative definition of the active
evaluation metrics.

> **Post DPO-removal note (v1):** DPO/preference data and all DPO-system
> predictions/results were deleted outright. Historical evaluation archives
> (`archive/evaluation_legacy/`, `outputs/archive/evaluation_legacy/`) no longer
> exist. Aggregated audit metadata remains in `outputs/evaluation/evaluation_audit_v1/`. Every active reported metric must satisfy:

```
declared definition == implemented formula == actual input data == aggregation == reported number
```

The machine-readable registry is `outputs/evaluation/evaluation_audit_v1/metric_registry.json`.

## Active metrics

| metric_id | display name | unit | scale | implementation | reference |
|---|---|---|---|---|---|
| `bleu4` | Corpus BLEU-4 | corpus | 0–1 | `src/vvqg/evaluation/metrics/bleu.py:corpus_bleu4` | coco-caption / `pycocoevalcap 1.2` |
| `cider` | CIDEr | corpus | 0–10 | `src/vvqg/evaluation/metrics/cider.py:cider_scores` | `pycocoevalcap 1.2` |
| `distinct_2` | Distinct-2 | corpus | 0–1 | `src/vvqg/evaluation/metrics/diversity.py:distinct_2` | Li et al. 2016 |
| `normalized_excess_duplicate_rate` | Normalized excess duplicate rate | corpus | 0–1 | `src/vvqg/evaluation/metrics/diversity.py` | excess-duplicate-rate (variant A) |
| `qbleu4_variant` | Q-BLEU4-variant (NOT standard Q-BLEU4) | corpus | 0–1 | `src/vvqg/evaluation/metrics/qbleu.py:qbleu4_variant` | local reimplementation |

### BLEU-4 (corpus)
- Definition: corpus-level BLEU-4; n-gram precisions summed over all pairs,
  combined with brevity penalty. Not the mean of per-sentence BLEU.
- Tokenization: lowercase `[a-z0-9']+`. Effective reference length: closest.
- Smoothing: Chen & Cherry method-1. This is a documented variant of the
  coco-caption additive smoothing (they agree except on zero-count orders).
- Missing data: empty candidate scores 0; images with no reference contribute
  nothing and lower the brevity penalty.
- Report field: `corpus_bleu4`. Per-sample `bleu4` is single-pair sentence BLEU
  and must never be averaged and called corpus BLEU.

### CIDEr
- Definition: official `pycocoevalcap` `Cider().compute_score` called **once**
  over the corpus; IDF document frequency is corpus-wide **within that call**.
  The system value is the mean of the per-sample scores.
- Do not shard the corpus and average shard scores; the implementation never
  does this. Parallel *preprocessing* is allowed; the metric must stay a single
  corpus call.
- Policy: fix one reference-corpus protocol across systems. The historical
  bake-off computed CIDEr per system (each system's own 1000 samples), which is
  recorded here as a protocol choice, not an implementation bug.

### Distinct-2
- `|unique word bigrams| / |all word bigrams|` over the whole list, 0–1.
- Bigrams never cross question boundaries; questions are **not** deduplicated;
  questions with fewer than two tokens contribute 0.

### Normalized excess duplicate rate
- `1 - |unique(strip().lower())| / N`, 0–1. For `[A, A, B]` this is `1/3`.
- This is **not** the repeated-group membership rate (`2/3`) and not an exact
  duplicate rate (punctuation is not stripped). `duplicate_rate` is a
  backwards-compatible alias.

### Q-BLEU4-variant
- Weighted corpus BLEU-4: matched n-grams weighted 1.0 when the candidate
  question type appears among the reference types, else 0.5.
- The paper's tuned component weight matrix is not reproduced and the official
  package is not installed, so this **must not** be labelled standard Q-BLEU4.

## Diagnostic metrics (not final metrics)

| metric_id | display name | unit | implementation | notes |
|---|---|---|---|---|
| `clipscore_raw_question` | CLIPScore (raw-question adaptation) | image-question | `src/vvqg/evaluation/metrics/clip.py:ClipScorer` | question encoded as-is; no `"A photo depicts"` prefix; not an exact reproduction |
| `clip_retrieval` | hard-negative CLIP retrieval | image-set | (protocol purged with the 2.2B/2.2C audit) | source rank 1-based; ties favour the source; 5-image and 10-image runs must not be mixed |

## Missing-data / statistics policy

- Failed or malformed judgments are **excluded** from denominators, never
  treated as `false`; failure counts are reported in runtime reports.
- Rates use percentile bootstrap 95% CIs (`common/bootstrap.py`).
- Paired system comparisons resample by stable `image_id`.
- Conditional statistics recompute numerator and denominator inside the
  bootstrap.
- System-level numbers are never copied onto per-sample rows.
- Histograms, means, threshold rates and N must use the same valid sample set.

## Excluded / deprecated

- `bertscore` — EXCLUDED (not implemented in the active package; a legacy copy
  exists only in the separate `shared/evaluation` preference subsystem).
- `pac_s` — EXCLUDED/BLOCKED (checkpoint unavailable).
- `vqascore` — EXCLUDED.
- `gpt_visual_dependence_1_5` — DEPRECATED (saturated absolute answer-reliance
  rubric; not a specificity metric).
- `gpt_setwise_specificity_10image`, `gpt_oneplusfour_specificity` — DEPRECATED
  historical diagnostics (purged; textual observations retained).
- `gpt_pairwise_2image` — DEPRECATED, removed before this audit (no raw evidence).

## Pending (PROPOSED_NOT_RUN)

- `single_image_decomposed_validity`
- `same_image_two_question_comparative`

These are **not** relabellings of the old applicability/winner outputs.
