# Publication snapshot pruning — 2026-10-08

STATUS: SNAPSHOT_PRUNING_COMPLETE

Only `/hpc2hdd/home/fye374/LJJ/look-before-you-ask/` was modified. The original research workspace, outputs, datasets and checkpoints were not modified. No Git initialization, staging, commit or push was performed.

| Metric | Value |
|---|---:|
| BEFORE_FILE_COUNT | 811 |
| AFTER_FILE_COUNT | 540 |
| FILES_DELETED | 282 |
| FILES_ADDED | 11 |
| BEFORE_SIZE (bytes) | 7757257 |
| AFTER_SIZE (bytes) | 5889458 |
| EXACT_DUPLICATES_DELETED | 18 |
| FILES_RELOCATED | 21 |
| CURRENT_PIPELINE_INTACT | YES |
| FINAL_EVAL_INTACT | YES |
| GQA_INTACT | YES |
| BASELINES_INTACT | YES |
| BROKEN_LINKS | 0 |
| IMPORT_FAILURES | 0 |
| COMPILE_FAILURES | 0 |
| PYTEST_PASS | 150 |
| PYTEST_SKIP | 12 |
| PYTEST_FAIL | 0 |

Counts and sizes cover regular files and logical bytes, excluding `.git` and symbolic links. FILES_DELETED counts pre-existing snapshot files; 38 additional temporary files created during this task were cleaned and logged with `existed_before=NO`. Eighteen pre-existing pytest temporary symlinks and newly generated temporary symlinks were cleaned separately; the four source symlinks remain intact. Files moved within the snapshot are not deletions. New indexes, navigation and final audit records are included in AFTER_SIZE and AFTER_FILE_COUNT.

| Directory category | Before | After |
|---|---:|---:|
| src files | 115 | 115 |
| scripts active | 109 | 109 |
| scripts archive | 154 | 4 |
| configs active | 13 | 13 |
| configs archive | 20 | 9 |
| docs current | 67 | 80 |
| docs archive | 25 | 22 |
| reports current | 66 | 53 |
| reports archive | 169 | 93 |
| tests | 26 | 26 |

The [deletion log](../../SNAPSHOT_DELETE_LOG.tsv) records SHA256, classification, original location, canonical replacement and A–F relevance answers for every removed regular file. A = final method; B = final evaluation; C = formal ablation; D = published baseline/GQA; E = necessary reproduction input; F = explicitly referenced mechanism. All six are NO for the deleted copy; uncertain items were retained. Eighteen duplicate copies were SHA256-equal to canonical files at deletion time; some historical canonical scripts were then independently pruned. No exact duplicate regular files remain.

The [historical experiment index](../archive/HISTORICAL_EXPERIMENT_INDEX.md) preserves branch status and negative-result conclusions. The [move log](../../SNAPSHOT_MOVE_LOG.tsv) records consolidation of historical results and relocation of final maintenance summaries/indexes into reproducibility audits.

All 268 protected runtime files (including source, tests, active scripts and required configs) are byte-identical. All hard-keep documents and reports remain present. Retained scientific prose is unchanged after normalizing Markdown destinations. The only code edits adjust the repository-root depth in two retained archive diagnostic scripts; both passed import smoke. The three retained diagnostic scripts support K4, parity and shortcut analysis. The two active Phase3A configs remain for test dependencies; G1 and the generic ablation contract remain for comparison/audit value.

[Validation](audits/snapshot_pruning_20261008/VALIDATION.json) checks Markdown relative paths and anchors, Python compilation (241 files), shell syntax (12 scripts), protected-file hashes and dependency references. [Main import smoke](audits/snapshot_pruning_20261008/IMPORT_MAIN.txt) covers 18 scripts and 11 modules; [evaluation import smoke](audits/snapshot_pruning_20261008/IMPORT_EVAL.txt) covers 22 scripts. Project imports resolve inside the snapshot. [Pytest](audits/snapshot_pruning_20261008/PYTEST.txt) reports 150 passed, 12 external-artifact skips, zero failures and one existing tensor-conversion warning in 14.64 seconds. No tests, skips or assertions were changed. Tests used the existing environments read-only, offline model settings, hidden GPUs and one CPU thread.

The final [three-level file tree](audits/snapshot_pruning_20261008/DIRECTORY_TREE.txt) is available in full. Directory overview:

```text
docs/
├── archive/
│   ├── diagnostics/
│   ├── evaluation/
│   └── historical/
├── data/
├── evaluation/
├── method/
└── reproducibility/
    ├── audits/
    ├── environments/
    └── vllm/

reports/
├── ablations/
├── analysis/
│   ├── documentation_cleanup_20261008/
│   ├── dpo_refinement_selection/
│   ├── judge_robustness/
│   ├── offline_dpo_k4_diversity_20261008/
│   ├── offline_dpo_parity_audit_20261008/
│   ├── offline_dpo_same_type_pair_audit_20261008/
│   └── published_baseline_intervention_20261004T032525Z/
├── archive/
│   ├── diagnostics/
│   ├── dpo/
│   └── historical/
├── baselines/
│   └── development/
├── cross_dataset/
│   └── gqa/
└── main/
    ├── final_evaluation_20261008/
    └── offline_dpo_freeze_20261008/

scripts/
├── archive/
│   └── diagnostics/
├── new_vqg/
└── vvqg/

configs/
├── archive/
│   └── historical/
├── baselines/
├── gqa_native/
├── new_vqg/
├── phase3a/
└── phase5c/
```
