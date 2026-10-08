# Documentation architecture cleanup — 2026-10-08

Completed inside the Git-ready snapshot only. All 752 pre-existing regular files retain their original file identity at the recorded destination. No file was deleted; the original research workspace and its outputs were not touched. No Git initialization, staging, commit or push was performed.

```text
DOCS_MOVED: 68
REPORTS_MOVED: 85
DUPLICATES_RELOCATED: 3
BROKEN_LINKS: 0
PYTEST_PASS: 150
PYTEST_SKIP: 12
PYTEST_FAIL: 0
WARNING: 1
IMPORT_SMOKE: PASS
```

Counts are per moved file: DOCS_MOVED includes root documentation/registry records and original docs; REPORTS_MOVED includes original reports moved into the reproduction audit directory. The three duplicates are included in the move counts, not added again. [DOCUMENT_MOVE_LOG.tsv](../../../DOCUMENT_MOVE_LOG.tsv) records every old/new path, category and reason.

## Navigation

- [Documentation guide](../../../docs/README.md): method, data, evaluation, reproduction and history.
- [Report guide](../../README.md): final results, formal component comparisons, diagnostics, published baselines, GQA and historical experiments.
- [Current output registry](../../../docs/reproducibility/OUTPUT_REGISTRY.md) contains only the seven requested logical asset families and their original paths.
- [Historical output index](../../../docs/reproducibility/HISTORICAL_OUTPUT_INDEX.md) links old output families to retained archive reports. No actual output was copied or moved.

Classification follows the existing historical phase map, CURRENT_METHOD_CODEMAP, BRANCH_ARCHIVE_PLAN and report content. Full Ours/step207 frozen results are under reports/main; early pilots and superseded methods remain archived. The formal ablation index links to the original QSE/D1–D2/G1–G2 tables without copying or rewriting results. GQA and published baselines retain their reported limitations and are not presented as VQAv2 Full Ours ancestors.

## Verification and preservation

[Markdown link audit](../../../docs/archive/HISTORICAL_EXPERIMENT_INDEX.md#removed-snapshot-files) checks local targets and Markdown anchors. Twenty-two previously absent output/model/report targets now link to explicit [external-reference records](../../../docs/reproducibility/EXTERNAL_REFERENCE_INDEX.md), not to fabricated local artifacts. External content itself was not fetched or verified.

[Preservation audit](../../../docs/archive/HISTORICAL_EXPERIMENT_INDEX.md#removed-snapshot-files) checks every original regular-file inode and path. Historical Markdown content matches after normalizing only link destinations; the reproduction guide additionally has its relocated helper-directory literal updated. The root README has new navigation. Three byte-identical duplicate records were moved without deletion; their canonical copies are listed in the [duplicate index](../../../SNAPSHOT_DELETE_LOG.tsv). No duplicate report groups remain outside the duplicate archive.

Six source/helper files required path-only updates for moved documentation or the image-intervention helper. Reversing those exact path substitutions reproduces their original SHA256; algorithmic content is unchanged. The relocated helper's root-depth adjustment was verified by import and unit tests. Existing symlinks continue to resolve within the snapshot.

[Main import smoke](../../../docs/archive/HISTORICAL_EXPERIMENT_INDEX.md#removed-snapshot-files): 15 scripts, 10 modules and the relocated transform dependency passed. [Evaluation import smoke](../../../docs/archive/HISTORICAL_EXPERIMENT_INDEX.md#removed-snapshot-files): 22 scripts passed. All project imports resolve within this snapshot.

[Final pytest run](../../../docs/archive/HISTORICAL_EXPERIMENT_INDEX.md#removed-snapshot-files): 150 passed, 12 explicitly documented external-artifact skips, 0 failed, 1 existing scalar-conversion warning, 12.33 seconds. Tests used the original metrics environment read-only, hidden GPUs, offline model settings, and one CPU thread. Main imports used the original generation environment.

The [first pytest attempt](../../../docs/archive/HISTORICAL_EXPERIMENT_INDEX.md#removed-snapshot-files) encountered 18 setup errors because the newly selected nested temporary directory lacked its parent. Creating that parent fixed the invocation; no tests or assertions were changed. Test fixtures remain under the ignored .pytest_tmp directory; no pre-existing file was deleted to clean them.

[Directory tree](../../../docs/archive/HISTORICAL_EXPERIMENT_INDEX.md#removed-snapshot-files) shows two levels beneath the requested roots, within the three-level limit.
