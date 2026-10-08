# CUDA 12.8 training environment

The paper training environment is Python 3.10.13, PyTorch 2.8.0+cu128, and
FlashAttention 2.8.3.post1. The target hardware is A800 (`sm_80`); it does not
alter the image processor or its pixel settings.

Build it from a clean checkout with:

```bash
scripts/bootstrap_training_env.sh
```

The script refuses to overwrite an existing virtual environment, downloads the
official FlashAttention wheel for PyTorch 2.8/CUDA 12/C++11 ABI with a pinned
SHA-256 checksum, and runs a BF16 CUDA kernel smoke test. CUDA 12.8 is supplied
by the PyTorch `cu128` wheel and is compatible with the installed 570.86.10
driver. It writes `requirements-training-cu128.lock` after the initial
successful build. Later recreations use that lock automatically.

For another Python location or a separate environment path:

```bash
PYTHON_BIN=/path/to/python3.10 VENV=/path/to/.venv scripts/bootstrap_training_env.sh
```

`scripts/run_pilot_training.sh` uses FlashAttention by default. To run the
standard PyTorch SDPA backend for an ablation, set
`VQG_ATTN_IMPLEMENTATION=sdpa`; the chosen backend is saved in
`training_config.json`.

The runner also uses 24 DataLoader workers per process and a persistent input
cache under `.cache/vqg_sft_inputs`. The cache stores deterministic tokenizer
and image-processor outputs without changing image pixel settings. Set
`VQG_PREPROCESS_CACHE_DIR=` to disable it, or choose another cache root.
