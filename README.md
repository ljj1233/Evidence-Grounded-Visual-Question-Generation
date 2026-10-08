# Evidence-Grounded Visual Question Generation

Image-only visual question generation: research code and lightweight paper records.

Start with the [documentation guide](docs/README.md) for the current method and reproduction, or the [report guide](reports/README.md) for final results, component ablations, baselines and GQA. No phase-history knowledge is required.

| Direction | Role | Start here |
|---|---|---|
| Full Ours | Final VQAv2 paper main model: G2 step256, initialized from D1 | [Current project](CURRENT_PROJECT.md) |
| Full Ours + Offline DPO refinement | Optional post-training refinement: selected step207 | [Model registry](docs/reproducibility/MODEL_REGISTRY.md) and [freeze record](reports/main/offline_dpo_freeze_20261008/FREEZE_RECORD.md) |
| Published baselines | Independent released-model / same-backbone comparisons | [Published baseline results](reports/baselines/development/PUBLISHED_VLM_VQG_BASELINES.md) |
| GQA cross-dataset experiments | Independent GQA-native training and comparisons | [GQA method contract](docs/method/GQA_NATIVE_FULL_OURS_METHOD_CONTRACT.md) |

The published baselines and GQA experiments are not ancestors of VQAv2 Full Ours. The [final evaluation report](reports/main/final_evaluation_20261008/FINAL_HOLDOUT_REPORT.md) records the frozen main-model comparison.

For a fresh clone, follow [REPRODUCIBILITY.md](docs/reproducibility/REPRODUCIBILITY.md). From the repository root, `python -m pytest tests/ -ra` runs CPU unit tests and explicitly skips integration checks whose named external artifacts are absent. Models, raw datasets, generations, API caches and environments are not included.

See [external assets](docs/reproducibility/EXTERNAL_ARTIFACTS.md), [path portability](docs/reproducibility/PORTABILITY_AUDIT.md), [failure triage](docs/reproducibility/TEST_FAILURE_TRIAGE.tsv), and the [current readiness audit](docs/reproducibility/GIT_READINESS_FINAL.md). [SNAPSHOT_AUDIT.md](docs/reproducibility/SNAPSHOT_AUDIT.md) and the archive reports preserve earlier results; their original failure counts and copy hashes describe the snapshot before portability work.

Historical material is indexed in [ARCHIVE_INDEX.md](ARCHIVE_INDEX.md) and the [phase map](docs/archive/HISTORICAL_PHASE_MAP.md).
