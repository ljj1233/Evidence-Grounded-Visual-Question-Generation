# EVALUATION ACTIVE PROTOCOL

Status: active. This is the single description of the currently supported VQG
evaluation system.

> **Post DPO-removal note (v1):** the repositories's DPO/preference data and all
> DPO-system predictions/results (mdpo/pdpo/random/vanilla) were deleted outright
> in the DPO purge, including the legacy evaluation archive
> (`archive/evaluation_legacy/` and `outputs/archive/evaluation_legacy/`). Paths
> to those archives in this document are historical-only and no longer resolve.
> The active evaluation audit metadata remains in `outputs/evaluation/evaluation_audit_v1/`.

## A. Active evaluation stack

Reference-based and diversity metrics only, computed by
`scripts/vvqg/evaluate.py` (library: `src/vvqg/evaluation/metrics`):

The legacy `shared/evaluation/metrics.py` registry was archived under
`archive/phase21_22/shared_evaluation/metrics.py`; it included duplicate
BLEU/CIDEr/diversity implementations and an excluded BERTScore path. The
canonical `vvqg` package is the only active metric implementation.

- `bleu4` — corpus BLEU-4 (method-1 smoothing), field `corpus_bleu4`
- `qbleu4_variant` — Q-BLEU4-variant (not standard Q-BLEU4), field `corpus_qbleu4_variant`
- `cider` — official `pycocoevalcap` CIDEr, field `corpus_cider`
- `distinct2` — corpus Distinct-2, field `distinct_2`
- `dup_rate` — normalized excess duplicate rate, field `normalized_excess_duplicate_rate`

Definitions and policies: `docs/EVALUATION_METRIC_CONTRACT_V1.md`.

### Canonical usage

```bash
.venv-vllm-cu128-v011/bin/python scripts/vvqg/evaluate.py \
  --predictions <predictions.jsonl> \
  --references  data/vqav2/splits/val_debug.jsonl \
  --metrics bleu4,qbleu4_variant,cider,distinct2,dup_rate \
  --out outputs/<name>
```

`--help` lists only supported metrics. Reference metrics require
`--references`; `clipscore` additionally requires `--device/--batch-size`.

### Tests

The active suite is collected from `tests/` only (`pyproject.toml`
`testpaths = ["tests"]`):

```bash
.venv-vllm-cu128-v011/bin/python -m pytest tests/test_evaluation_metrics.py -q
```

It covers the K=4 golden fixtures, distinct-2 / duplicate-rate definitions, and
the BLEU-4 differential against `pycocoevalcap` (skipped if unavailable).

## B. Diagnostic-only stack

- `clipscore_raw_question` — raw-question CLIPScore adaptation
  (`src/vvqg/evaluation/metrics/clip.py`). Diagnostic only; not an exact
  reproduction of Hessel et al.
- CLIP hard-negative retrieval diagnostics — designed with the (now purged) 2.2B
  protocol; 1-based source rank, ties favour the source, 5-image and 10-image
  runs are not mixed.

Diagnostics must never be presented as the final metric.

## C. Archived / deprecated stack

Removed from the active path. Code and DPO-system outputs were **purged** in the
DPO-removal phase; judge prompt text is retained in
`outputs/evaluation/evaluation_audit_v1/judge_prompts.json`.

| protocol / metric | status | location |
|---|---|---|
| metric bake-off v1 (BLEU/CIDEr/Q-BLEU4/CLIP/PAC-S runner + GPT dependence) | DEPRECATED (purged) | historical paths `archive/evaluation_legacy/metric_bakeoff/` no longer exist |
| 10-image setwise specificity (2.2B) | DEPRECATED (purged) | historical observation retained in `docs/archive/evaluation/QUESTION_SPECIFICITY_AUDIT_V1.md` |
| 1 source + 4 hard-negative specificity (2.2C) | DEPRECATED (purged) | historical observation retained in `docs/archive/evaluation/QUESTION_SPECIFICITY_1PLUS4_V1.md` |
| 2-image pairwise specificity | REMOVED_OBSOLETE | removed before the audit; no raw evidence retained |
| absolute GPT visual_dependence 1–5 | DEPRECATED (purged) | rubric text retained in `outputs/evaluation/evaluation_audit_v1/judge_prompts.json` |
| BERTScore | EXCLUDED | not in the active vvqg evaluation package |
| PAC-S / VQAScore | EXCLUDED | not in the active vvqg evaluation package |

The DPO-system raw outputs and the legacy evaluation archive were **purged**
(not retained) in the DPO-removal phase; only aggregate audit metadata remains,
in `outputs/evaluation/evaluation_audit_v1/`.

## D. Verified definitions (traceability)

For every active metric the chain is:

```
docs/EVALUATION_METRIC_CONTRACT_V1.md (definition)
  -> src/vvqg/evaluation/metrics/<module>.py (implementation)
  -> prediction/reference JSONL keyed by image_id (raw inputs)
  -> common/bootstrap.py + corpus aggregation (aggregation)
  -> scripts/vvqg/evaluate.py system_summary.json (report field)
```

Registry with all fields (unit, numerator/denominator, missing-data policy,
reference implementation, conformance status, archive location):
`outputs/evaluation/evaluation_audit_v1/metric_registry.json`.

No specifity protocol, reward, or new final metric is promoted in this phase.
