# METRIC BAKE-OFF v1

> **No final evaluation metric is selected in Metric Bake-off v1.** The purpose of this
> experiment is to measure discrimination, counterfactual sensitivity, judge correlation,
> and computational cost. BERTScore is excluded; VQAScore and PAC-S are deferred/blocked.

## Table A — system-level automatic metrics

| system | n | BLEU-4 (corpus) | CIDEr (corpus) | Q-BLEU4 (corpus) | Distinct-2 | Dup rate | CLIPScore (mean) |
|---|---:|---:|---:|---:|---:|---:|---:|
| mdpo | 1000 | 0.1860 | 0.2911 | 0.1802 | 0.2068 | 0.2500 | 0.642222 |
| pdpo | 1000 | 0.2209 | 0.3636 | 0.2152 | 0.2166 | 0.2780 | 0.649518 |
| random | 1000 | 0.2213 | 0.3682 | 0.2155 | 0.2144 | 0.2840 | 0.649523 |
| sft_baseline | 1000 | 0.1690 | 0.1868 | 0.1644 | 0.1231 | 0.4630 | 0.5893 |
| vanilla | 1000 | 0.2187 | 0.3562 | 0.2130 | 0.2097 | 0.2910 | 0.6462 |

## Table B — correlation with GPT-4o judge

| metric | visual_dependence (rho/tau) | visual_support (rho/tau) | naturalness (rho/tau) |
|---|---|---|---|
| bleu4 | -0.007234/-0.005941 (n=5000) | 0.015739/0.012868 (n=5000) | -0.007078/-0.0058 (n=5000) |
| cider | 0.035907/0.029322 (n=5000) | 0.038656/0.031405 (n=5000) | 0.035929/0.029341 (n=5000) |
| qbleu4 | 0.011506/0.009424 (n=5000) | 0.046308/0.03773 (n=5000) | 0.018614/0.015249 (n=5000) |
| clipscore | 0.000345/0.000287 (n=5000) | -0.038133/-0.030909 (n=5000) | 0.03297/0.026919 (n=5000) |

## Table C — CLIPScore source-vs-mismatch diagnostic

- CLIP source mean: 0.635353
- CLIP mismatch mean: 0.447773
- CLIP gap (source - mismatch): 0.18758

_Counterfactual metric diagnostic (not a proposed metric). Mismatched images are a
deterministic rotation over sorted val_debug image ids; PAC-S diagnostic blocked._

## Table D — runtime / throughput / hardware

```json
{
  "records": 5000,
  "num_workers": 32,
  "seed": 42,
  "reference_metrics_seconds": 5.12,
  "clip": {
    "device": "cuda",
    "batch_size": 64,
    "unique_images": 1000,
    "seconds": 35.04,
    "peak_gpu_mb": 674.3
  },
  "pacscore": {
    "available": false,
    "reason": "official PAC-S checkpoint not available locally (no download in this environment)"
  },
  "judge": {
    "status": "ok",
    "model": "gpt-4o",
    "requested": 5000,
    "judged": 5000,
    "failures": 0,
    "cache_hits": 5000,
    "concurrency": 32,
    "seconds": 0.54
  },
  "wall_clock_seconds": 60.07
}
```

## Implementations (source / backbone / protocol)

| metric | implementation | backbone / checkpoint | preprocessing / prompt |
|---|---|---|---|
| BLEU-4 | this repo (`metric_bakeoff/metrics.py`), corpus-level, method-1 smoothing | — | lowercase regex tokens `[a-z0-9']+` |
| CIDEr | official `pycocoevalcap 1.2` (`Cider`) | — | default tokenizer |
| Q-BLEU4 | **re-implementation** of Nema & Khapra, EMNLP 2018 (no official package installed) | — | question-type-weighted n-grams; weights 1.0 match / 0.5 mismatch (paper weight matrix not reproduced) |
| Distinct-2 / Dup rate | this repo (`metric_bakeoff/metrics.py`) | — | lowercase regex tokens |
| CLIPScore | official Hessel et al. 2021 protocol (`2.5 * max(cos, 0)`) | `openai/clip-vit-base-patch32` (local `~/models/clip-vit-base-patch32`) | text = question as-is (no prompt prefix); RGB image; transformers 4.57.6 |
| PAC-S | official `aimagelab/pacscore` — **blocked** | checkpoint unavailable locally | not run |
| GPT-4o judge | configured endpoint (`AUDIT_JUDGE_MODEL=gpt-4o`), temperature 0, prompt_version `bakeoff_v1` | gpt-4o | structured JSON: visual_dependence/visual_support/naturalness/failure_type |

Environment: transformers 4.57.6, torch 2.9.0+cu128, scipy 1.15.3.

## Outputs

`outputs/metric_bakeoff_v1/`: `per_sample_metrics.jsonl`, `system_summary.json`,
`gpt4o_judge.jsonl`, `correlations.json`, `counterfactual_summary.json`, `runtime_report.json`.

## Restrictions respected

No final metric selected; no reward/SFT/GRPO/dataset/training-config change; no BERTScore;
no VQAScore; no new combined score; GPT-4o used only as an independent anchor.

## GPT-4o Judge Analysis of Completed DPO Systems

Offline analysis of cached GPT-4o judgments (no new API calls). Primary systems: mdpo, pdpo, random, vanilla. `sft_baseline` is excluded (incomplete-training diagnostic).

### Table 1 — system-level dependence / support / naturalness

| system | dep N | dep mean | dep median | dep p25/p75 | dep ≥4 | dep 95% CI | support mean | full=2 | part=1 | unsup=0 | nat mean | nat ≥4 |
|---|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|
| mdpo | 1000 | 4.943 | 5.0 | 5.0/5.0 | 1.0 | [4.929, 4.957] | 1.923 | 0.945 | 0.033 | 0.022 | 4.995 | 1.0 |
| pdpo | 1000 | 4.955 | 5.0 | 5.0/5.0 | 1.0 | [4.941, 4.968] | 1.94 | 0.962 | 0.016 | 0.022 | 4.997 | 1.0 |
| random | 1000 | 4.953 | 5.0 | 5.0/5.0 | 1.0 | [4.939, 4.965] | 1.943 | 0.963 | 0.017 | 0.02 | 4.997 | 1.0 |
| vanilla | 1000 | 4.956 | 5.0 | 5.0/5.0 | 1.0 | [4.943, 4.968] | 1.951 | 0.967 | 0.017 | 0.016 | 4.998 | 1.0 |

Dependence histograms (1/2/3/4/5): mdpo=0/0/0/57/943; pdpo=0/0/0/45/955; random=0/0/0/47/953; vanilla=0/0/0/44/956

Support histograms (0/1/2): mdpo=22/33/945; pdpo=22/16/962; random=20/17/963; vanilla=16/17/967

### Table 2 — failure-type distribution

| system | none | generic | false_premise | wrong_relation | ambiguous_reference | unobservable_target | unnatural | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mdpo | 0.945 | 0.000 | 0.001 | 0.000 | 0.009 | 0.045 | 0.000 | 0.000 |
| pdpo | 0.962 | 0.000 | 0.001 | 0.000 | 0.005 | 0.032 | 0.000 | 0.000 |
| random | 0.963 | 0.000 | 0.001 | 0.000 | 0.005 | 0.031 | 0.000 | 0.000 |
| vanilla | 0.967 | 0.000 | 0.001 | 0.000 | 0.006 | 0.026 | 0.000 | 0.000 |

_20 most common questions per system are in `gpt4o_failure_matrix.json`._

### Table 3 — dependence × support diagnostic quadrants

| system | generic-safe (d≤2,s=2) | desired (d≥4,s=2) | specific-but-unsupported (d≥4,s=0) | weak-and-unsupported (d≤2,s=0) |
|---|---:|---:|---:|---:|
| mdpo | 0.000 | 0.945 | 0.022 | 0.000 |
| pdpo | 0.000 | 0.962 | 0.022 | 0.000 |
| random | 0.000 | 0.963 | 0.020 | 0.000 |
| vanilla | 0.000 | 0.967 | 0.016 | 0.000 |

### Table 4 — paired system comparisons (by image_id)

| comparison | dep mean Δ [95% CI] | dep win/tie/loss | support mean Δ [95% CI] | support win/tie/loss |
|---|---|---|---|---|
| pdpo vs vanilla | -0.001 [-0.019, 0.016] | 0.007/0.985/0.008 | -0.011 [-0.037, 0.015] | 0.004/0.987/0.009 |
| mdpo vs vanilla | -0.013 [-0.032, 0.004] | 0.021/0.945/0.034 | -0.028 [-0.056, -0.001] | 0.012/0.953/0.035 |
| random vs vanilla | -0.003 [-0.021, 0.014] | 0.007/0.983/0.01 | -0.008 [-0.033, 0.02] | 0.004/0.988/0.008 |
| pdpo vs random | 0.002 [-0.016, 0.019] | 0.005/0.992/0.003 | -0.003 [-0.029, 0.026] | 0.002/0.995/0.003 |
| mdpo vs pdpo | -0.012 [-0.03, 0.005] | 0.025/0.938/0.037 | -0.017 [-0.046, 0.012] | 0.016/0.95/0.034 |

### Automatic metrics vs GPT-4o (no recomputation)

| system | BLEU-4 | CIDEr | Q-BLEU4 | Distinct-2 | Dup rate | CLIPScore | GPT dep mean | GPT support mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mdpo | 0.1860 | 0.2911 | 0.1802 | 0.2068 | 0.2500 | 0.6422 | 4.943 | 1.923 |
| pdpo | 0.2209 | 0.3636 | 0.2152 | 0.2166 | 0.2780 | 0.6495 | 4.955 | 1.94 |
| random | 0.2213 | 0.3682 | 0.2155 | 0.2144 | 0.2840 | 0.6495 | 4.953 | 1.943 |
| vanilla | 0.2187 | 0.3562 | 0.2130 | 0.2097 | 0.2910 | 0.6462 | 4.956 | 1.951 |

_pdpo / random / vanilla have very close reference metrics (BLEU-4 0.219–0.221, CIDEr 0.356–0.368); their GPT-4o dependence/support are reported above. No causal claim is made._

### Saturation audit

| dimension | modal category | modal fraction | entropy | effective categories |
|---|---:|---:|---:|---:|
| visual_dependence | 5 | 0.9517 | 0.19333 | 1.2133 |
| visual_support | 2 | 0.9593 | 0.198559 | 1.2196 |
| naturalness | 5 | 0.9968 | 0.021864 | 1.0221 |

Saturation (e.g. naturalness/support concentrated in one category) limits the dynamic range available for sample-level correlation, which helps explain the near-zero metric↔judge correlations reported in Table B.

### Appendix — excluded row

- `sft_baseline`: **incomplete-training diagnostic; excluded from scientific comparison.** Its Distinct/duplicate/GPT scores are not used to support claims about a properly trained SFT model.

