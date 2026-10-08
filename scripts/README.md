# `scripts/` — execution interface

Current pipeline only: **VQAv2 → Qwen3-VL SFT → Reward Calibration → GRPO**.
No DPO / preference-pair workflow remains.

## Layout

```text
scripts/
├── build_image_split_manifest.py   # build data/train_resources/image_split_manifest.json
├── new_vqg/check_dataset.py        # dataset checks
└── vvqg/
    ├── generate.py                 # CANONICAL generation (greedy / N-sample), fast
    ├── evaluate.py                 # canonical active evaluation CLI (metrics)
    ├── serve_vllm_baseline.py      # optional vLLM server (used by GRPO rollouts)
    ├── prepare_sft_data.py         # build src/vvqg/data/sft/{train,dev,test}.jsonl
    ├── validate_sft_dataset.py     # SFT dataset validator
    ├── smoke_sft.py                # SFT smoke test
    └── diagnose_candidates.py      # candidate-pool diagnostics
```

## Generation (canonical)

`scripts/vvqg/generate.py` (library `src/vvqg/generation/generator.py`) is the
single generation entry point:

```bash
.venv/bin/python scripts/vvqg/generate.py --data <manifest.jsonl> \
    --output <out.jsonl> --schema greedy
.venv/bin/python scripts/vvqg/generate.py --data <manifest.jsonl> \
    --output <out.jsonl> --schema sampled --num-samples 4
```

Efficiency notes:
- The **PIL image-processor backend** is ~100x faster than the torchvision one
  (0.01 vs 0.96 s/img); the default is `--backend pil`. Use
  `--backend torchvision` only for exact reproduction of pre-optimization runs.
- Cross-image batching (`--batch-size`, default 16) and N samples per image in
  one batched call; measured 8.5 img/s greedy and 5.1 img/s for four-sample pools on the Phase 2.3 A800 run.
- Model load (~5 min) dominates a single run; running several jobs in one
  process amortizes it.

The former duplicate generators (`eval_generate.py`, `generate_sampled.py`,
`generate_reward_calibration.py`, `generate_val_debug_vllm.py`,
`generate_val_debug_concurrent.py`) were deleted; their functionality is fully
covered by `generate.py`.

Phase 2.1–2.2 calibration, controlled-pair, dependency-ablation, and
information-selection runners are historical and live under
`archive/phase21_22/scripts/`. They are not active entry points.

The DPO / preference-construction scripts (`aligned_preference_baselines/`,
`preference_baselines/`, ECQ verifier, pool/contamination auditors, progress
metrics) were deleted in the DPO-removal phase; see `docs/DPO_REMOVAL_REPORT.md`.
