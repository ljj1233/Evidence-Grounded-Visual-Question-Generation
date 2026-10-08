# Path portability audit — 2026-10-08

The current source tree has no hard-coded private HPC runtime paths. Python model defaults use `VVQG_MODELS_DIR`; existing CLI model arguments remain available. Current YAML model fields use editable `models/<name>` paths. Shell launchers read `VVQG_MODELS_DIR`, defaulting to the current user's model directory. Config files do not expand environment variables; see [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for configuring shell and YAML paths consistently.

The pre-edit inventory classified 462 occurrences: 76 `RUNTIME_REQUIRED`, 281 `PROVENANCE_ONLY`, and 105 `STALE`. Each row records its file, line, column, original path and classification in [PATH_OCCURRENCES_BEFORE.tsv](audits/git_readiness_20261008/PATH_OCCURRENCES_BEFORE.tsv). Runtime path edits cover 60 existing source/config/launcher files, plus a shared model-directory resolver. No model identifier, loss, selection threshold, seed, training schedule or final checkpoint definition changed.

| Classification | Meaning and disposition |
|---|---|
| `RUNTIME_REQUIRED` | Executed current source/defaults and current config values. Replaced private model paths with environment/CLI-configurable locations; the legacy rollout script's repository root now comes from its own file. Direct script imports were checked without `PYTHONPATH`. |
| `PROVENANCE_ONLY` | Original paths in model/asset registries, frozen reports, audit results and explanatory comments/docstrings. Retained as evidence of the producing workspace. They are not instructions to use another user's filesystem. |
| `STALE` | Paths in archived scripts/configs that are retained as historical source records. No current source or tests directly reference `scripts/archive` / `configs/archive` entry points. Archived launchers are outside the supported clean-clone runtime; they must be reviewed before replay. They are not reclassified as active methods. |

The [final occurrence inventory](audits/git_readiness_20261008/PATH_OCCURRENCES_FINAL.tsv) classifies remaining matches after editing. Scans cover source, scripts, configs, tests, shared modules, documentation, root Markdown files and retained reports. Git metadata, transient pytest files and this stage's generated audit inventories/logs are excluded from occurrence counts to avoid counting their own recorded findings recursively. Test logs contain machine-local traceback paths as provenance, not executable defaults.

All historical documents and archived report/config files were checked against pre-edit SHA256 records and left byte-identical. Current YAML configurations were checked by restoring only their original model-location scalars in memory and comparing hashes; all other configuration text is identical. Earlier snapshot manifests/hashes remain historical evidence, not hashes of the newly edited files.

Private provenance paths remain visible in retained records. `HARD_CODED_RUNTIME_HPC_PATHS: 0` does not mean those records were anonymized. No private key or credential value is printed in the sensitive-information scan; short test credentials are recorded only by file and line.

The standalone legacy `phase3a_flat_rollouts.py` executes model/GPU work at module scope, so it received syntax and path checks rather than an import that would start work. Ordinary tests and the active main/evaluation import checks do not execute it. Older orchestrators also require named local environments and external artifacts; their relative layout is documented, not bundled in this repository.
