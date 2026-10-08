# Snapshot audit — 2026-10-08

Copy-only code and lightweight report backup. The original workspace remains in place. No checkpoints were deleted in this stage; no Git initialization, add, commit or push was performed.

## Verification

Rechecked the existing snapshot on 2026-10-08: all 714 copied entries still match the source and manifest; both import smoke suites passed again; pytest repeated the same 138 passes and 17 missing-artifact failures in 125.74s. No original workspace file was written, moved or deleted during this recheck.

- Copied entries verified against original SHA256: 714 (includes required source dependencies).
- Original retained files verified unchanged: 44,135; original `.gitignore` unchanged.
- Import smoke: 15 current main scripts + 10 core modules PASS; 22 current evaluation scripts PASS. Imported vvqg/vqg code was asserted to resolve within this snapshot, not the original repo.
- `pytest tests/`: **138 passed, 17 failed, 1 warning**. All 17 failures are `FileNotFoundError` for intentionally excluded `data/` or `outputs/` artifacts. This is **not an all-tests-pass snapshot**.
- The initial collection failure exposed a dynamically imported transform helper under `reports/`; its byte-identical source was copied and collection now completes (155 tests). No test code was changed or skipped, and no dataset/raw generation was copied to make tests pass.
- Pure code tests run using the original workspace environments. We did not install dependencies, change environments, run training, inference, API calls or new evaluation.

## Sensitive information and portability

- Credential candidates: **6**, all reviewed as dummy literals in `tests/test_judge_roles.py` (lines 17, 23, 40, 49, 100) and `tests/test_phase4a4_audit.py` (line 105); **0 unresolved candidates / 0 provider-token or private-key matches**. Values are not logged. These are local environment-dictionary or monkeypatched request fixtures, not loaded credentials. Pattern scanning does not prove absence of all possible secrets.
- Private absolute HPC paths: **364 lines across 177 files**. File/line-only findings are in [PRIVATE_PATH_CANDIDATES.tsv](audits/snapshot_audit_20261008/PRIVATE_PATH_CANDIDATES.tsv). These include executable/config defaults and intentional provenance paths; they were flagged, not silently rewritten.
- The backup is ready for local Git review; absolute paths still need privacy/portability review before public release. No Git action has been taken.

## Scope

Included src/scripts/configs/docs/tests, lightweight archives, requested current indexes, final evaluation/freeze summaries, cleanup provenance, pyproject.toml, and required shared/helper source. `src/vqg` relative links resolve entirely inside the new copy.

Excluded original `.git`, environments, `data/`, `outputs/`, models and binary states, raw generations/API rows, images/archives, activation/detector/embedding caches, and the SFT image/question records under `src/`. Small split manifests contain only IDs, paths and split metadata. All original source files and active model/data artifacts remain in place.

External model/dataset/cache paths, sizes and known hashes are recorded in [EXTERNAL_ARTIFACTS.md](EXTERNAL_ARTIFACTS.md). Historical references in copied documents are preserved; renamed archive paths resolve through MOVE_LOG.tsv, and excluded result paths resolve in the original workspace.

## Audit files

- [COPY_MANIFEST.tsv](audits/snapshot_audit_20261008/COPY_MANIFEST.tsv), [EXCLUDED_FILES.tsv](audits/snapshot_audit_20261008/EXCLUDED_FILES.tsv)
- [VERIFICATION.json](audits/snapshot_audit_20261008/VERIFICATION.json), [SECRET_CANDIDATES.tsv](audits/snapshot_audit_20261008/SECRET_CANDIDATES.tsv)
- [PYTEST_RESULT.txt](audits/snapshot_audit_20261008/PYTEST_RESULT.txt), [TEST_EXTERNAL_DEPENDENCIES.tsv](audits/snapshot_audit_20261008/TEST_EXTERNAL_DEPENDENCIES.tsv)
- [IMPORT_MAIN.txt](audits/snapshot_audit_20261008/IMPORT_MAIN.txt), [IMPORT_EVAL.txt](audits/snapshot_audit_20261008/IMPORT_EVAL.txt)

Tests were run from the snapshot root with the original `.venv-vllm-cu128-v011/bin/python -B`, `sys.path.insert(0,"src")`, `pytest.main(["tests/","-q","-p","no:cacheprovider","--tb=short"])` and process-local `PYTHONDONTWRITEBYTECODE=1`, thread limits of 1. Main imports used the original `.venv/bin/python -B`; evaluation imports used the metrics environment.

## Final size and contents

```text
SNAPSHOT_SIZE_BYTES: 7122477
FILE_COUNT: 732
FILES_OVER_100MB: []
MODEL_BINARIES_INCLUDED: 0
RAW_DATA_INCLUDED: 0
VENV_INCLUDED: 0
SECRET_CANDIDATES: 6
UNRESOLVED_SECRET_CANDIDATES: 0
PRIVATE_HPC_PATH_LINES: 364
PYTEST_RESULT: 138 passed, 17 failed (excluded external data), 1 warning
IMPORT_SMOKE: PASS
```

Size = sum of regular file lengths and symlink target text lengths, without following directory symlinks; allocated filesystem space is not used.
