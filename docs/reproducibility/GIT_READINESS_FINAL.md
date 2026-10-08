# Final Git readiness — 2026-10-08

**READY_FOR_GIT_REVIEW.** Normal CPU tests have no failures. External frozen-artifact checks skip only for named absent files. No Git initialization, staging, commit or push was performed during this readiness stage. The earlier authorized repository binding remains in place.

```text
SNAPSHOT_SIZE_BYTES: 7399462
FILE_COUNT: 756
FILES_OVER_100MB: 0
MODEL_BINARIES_INCLUDED: 0
RAW_DATA_INCLUDED: 0
VENV_INCLUDED: 0
SECRET_CANDIDATES_UNRESOLVED: 0
HARD_CODED_RUNTIME_HPC_PATHS: 0
PYTEST_PASS: 150
PYTEST_SKIP: 12
PYTEST_FAIL: 0
WARNING: 1
```

Size and file count include regular files and symlink text, excluding `.git/`. Test temporary files have been removed. No model binaries, raw data, caches, environments or excluded large artifacts were copied into the repository. The original research workspace was not modified, moved or cleaned.

## Tests and imports

- Before fixes: 138 passed, 17 failed, 1 warning. All 17 were classified before patching in [TEST_FAILURE_TRIAGE.tsv](TEST_FAILURE_TRIAGE.tsv); every failure depended on intentionally excluded dataset/run artifacts.
- Five train-only gate tests now use a tiny temporary synthetic manifest and retain their algorithm assertions. Twelve official-data/frozen-run checks retain their original assertions and explicitly skip when declared artifacts are absent. Missing repository code/config/schema files are not a skip condition. No tests were deleted or disabled wholesale.
- Final clean-repo command: `python -B -m pytest tests/ -ra --tb=long -p no:cacheprovider --basetemp=.pytest_tmp`, from this repository root using the original metrics Python; GPUs hidden, Hugging Face offline, CPU thread limits set to one. **150 passed, 12 skipped, 0 failed, 1 warning in 62.97 seconds**. The warning is the pre-existing gradient-bearing scalar-to-float conversion in a trainer unit test.
- Import smoke: 15 main scripts + 10 core modules, 22 evaluation scripts, and the changed model-path entry points passed. Project imports were verified to resolve inside the snapshot. One legacy top-level GPU rollout script was syntax checked without importing/executing it.
- All 384 Python files compiled in memory; all 18 shell scripts passed `bash -n`. YAML configs and JSON config/schema files loaded successfully. Four CLI entry points passed `--help` without `PYTHONPATH`.

Evidence: [full baseline failure output](audits/git_readiness_20261008/PYTEST_BEFORE.txt), [final pytest output and skip reasons](audits/git_readiness_20261008/PYTEST_FINAL.txt), [main imports](audits/git_readiness_20261008/IMPORT_MAIN.txt), [evaluation imports](audits/git_readiness_20261008/IMPORT_EVAL.txt), [extended imports](audits/git_readiness_20261008/IMPORT_PORTABILITY.txt), [syntax/config invariants](audits/git_readiness_20261008/STATIC_CHECKS.json), and [config/schema loading](audits/git_readiness_20261008/CONFIG_LOADING.json).

## Paths, assets and repository consistency

Current runtime private HPC paths were replaced with configurable model locations. Final occurrence classification retains 283 provenance-only occurrences (including test scanner prefixes) and 105 stale archive occurrences; these are not current executable defaults. Original locations/hashes in historical reports and asset registries are deliberately retained. See [PORTABILITY_AUDIT.md](PORTABILITY_AUDIT.md) for scope and row-level inventories.

Six credential-shaped test literals were reviewed as dummy fixtures; zero unresolved credentials and zero provider-token/private-key matches were found. Values were not printed. The [scan record](audits/git_readiness_20261008/SECRET_SCAN.json) contains only locations and classifications. Pattern scans are not a guarantee against all possible secret formats.

README, CURRENT_PROJECT, MODEL_REGISTRY, EXTERNAL_ARTIFACTS, REPRODUCIBILITY and `.gitignore` are present. Navigation distinguishes Full Ours, optional Offline DPO refinement, published baselines and independent GQA experiments. All linked repository documents in the current navigation were checked. Existing archive documents/reports are unchanged; current YAML configurations differ only in model locations. The old snapshot audit and copy manifests describe the pre-portability snapshot and remain historical records.

No new environment was installed; verification used the original environments read-only. The fresh CPU setup is documented with observed package versions in [REPRODUCIBILITY.md](REPRODUCIBILITY.md), but was not independently installed during this audit. No training, model download, API judging or score reproduction was run.

Machine-readable result: [FINAL_CHECKS.json](audits/git_readiness_20261008/FINAL_CHECKS.json).
