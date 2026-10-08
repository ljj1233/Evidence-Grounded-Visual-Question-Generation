# Reproducing repository checks

Run commands from the clone root with Python 3.10+. The tests need Python libraries, not pretrained weights, datasets, GPU access, API keys or model downloads. The existing verification environment used Python 3.10 and the library versions in [requirements-test.txt](../../requirements-test.txt); its CUDA-enabled PyTorch was exercised with GPUs hidden. A new CPU environment was not installed during this audit.

## Fresh CPU environment

```bash
python3 -m venv .venv-test
.venv-test/bin/python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
.venv-test/bin/python -m pip install -r requirements-test.txt
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=1 .venv-test/bin/python -m pytest tests/ -ra
```

Package installation uses the network; the test run does not need model downloads or external services. `pyproject.toml` supplies `src` on pytest's import path. Do not install the generation/vLLM stacks simply to run CPU unit tests.

For an existing environment, use its Python executable in place of `.venv-test/bin/python`. The audit ran the full suite using the original metrics environment without changing or copying that environment. Main training/generation import smoke used the separate original generation environment, which contains PEFT; evaluation import smoke used the metrics environment. CPU unit-test dependencies alone do not provide every training import.

## Test policy

```bash
python -m pytest tests/ -ra                         # units plus explicit absent-asset skips
python -m pytest tests/ -m 'not integration' -ra    # units only
python -m pytest tests/ -m external_artifact -ra   # frozen artifact integrity checks
```

The original 17 failures were classified before any fixes in [TEST_FAILURE_TRIAGE.tsv](TEST_FAILURE_TRIAGE.tsv). Five filtering tests now build tiny synthetic manifests in pytest temporary directories while retaining their algorithm assertions. Twelve frozen-data/run checks require genuine external artifacts and skip only if their declared files are absent. Invalid JSON, hash mismatches, leakage or other assertion failures still fail when artifacts are present. No blanket `FileNotFoundError` handler or silent pass was added.

The original Phase 3A/4A and counterfactual experiment integrity tests are retained as integration checks. Their historical origin does not make their recorded invariants obsolete; shared algorithm tests remain ordinary unit tests. No test was deleted or moved. The snapshot already contains the small transform helper under `reports/analysis/published_baseline_intervention_20261004T032525Z/scripts/` and the `shared/` modules reached by `src/vqg` symlinks; keep them when packaging the clone.

Required integration files are listed in [EXTERNAL_ARTIFACTS.md](EXTERNAL_ARTIFACTS.md). To run those checks, mount or link the external files at their recorded repository-relative locations locally. Do not add them to Git. The raw GQA source hash test also needs the file named by `datasets.gqa.splits.test.source_path` in the external manifest. Skip decisions never bypass missing repository source/config/schema files.

## Model and runtime paths

Python model defaults use `VVQG_MODELS_DIR`, with a default of `Path.home() / "models"`. Set it before starting a process. Existing explicit `--base-model`, `--base`, `--flat-model`, `--model` and `--embedding-model` options take precedence where provided. Shell launchers use the same environment variable.

```bash
export VVQG_MODELS_DIR="$HOME/model-storage"
PYTHONPATH=src python scripts/vvqg/generate.py --help
```

Current YAML configs use editable repository-relative `models/<model-name>` locations. YAML does not interpolate environment variables: edit the relevant config model fields to local absolute paths, or provide an ignored `models/` symlink to your model directory. When using a shell chain that also reads YAML, configure both consistently. Start config-driven commands from the repository root. Output/manifests/checkpoint fields remain explicit relative config paths; the referenced files are external prerequisites.

Generation/SFT and metrics/vLLM have separate environments; older orchestration scripts expect `.venv/bin/python` and `.venv-vllm-cu128-v011/bin/python` beneath the clone. These environments are not included and must be created locally for those workflows. Released LOVA3/Genixer additionally require their documented official code and specialized environments. None is needed for normal unit tests.

The frozen model registry and reports intentionally retain original absolute paths as provenance. They are not portable executable defaults. Archived scripts/configs are historical records; importing selected helpers does not make whole archived launchers runnable. See [PORTABILITY_AUDIT.md](PORTABILITY_AUDIT.md).

## Verification limits

The audit checks CPU logic, imports, Python/shell syntax and config/schema loading. It does not train, load model weights, call judging APIs, regenerate data or reproduce paper scores. The one recorded warning is an existing test's conversion of a gradient-bearing scalar tensor to `float`; it is neither hidden nor treated as a model failure. See [GIT_READINESS_FINAL.md](GIT_READINESS_FINAL.md) for the final counts and evidence links.
