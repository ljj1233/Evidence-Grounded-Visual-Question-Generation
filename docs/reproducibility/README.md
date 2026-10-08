# Reproducibility

Start with [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the supported clean-clone setup and CPU test policy.

- [Model registry](MODEL_REGISTRY.md) records the selected models and weight identities.
- [External artifacts](EXTERNAL_ARTIFACTS.md), [current outputs](OUTPUT_REGISTRY.md), and [historical outputs](HISTORICAL_OUTPUT_INDEX.md) locate assets kept outside this repository.
- [Current method specification](../method/CURRENT_METHOD_TECHNICAL_SPEC.md) explains the historical checkpoint versus current-source reproduction boundary.
- [Portability audit](PORTABILITY_AUDIT.md) and [clean-repository test audit](GIT_READINESS_FINAL.md) record the preceding snapshot checks. [Documentation cleanup](../../reports/analysis/documentation_cleanup_20261008/FINAL_REPORT.md) records the latest layout verification.
- [CUDA environment record](environments/training_environment.md) and [vLLM runtime record](vllm/SAME_BACKBONE_BASELINES_RUNTIME.md) preserve older environment recipes. Their original bootstrap/launcher references are historical provenance; this snapshot does not claim to include all referenced tooling. Use the main reproduction guide for a fresh clone.

Frozen implementation and checkpoint provenance documents in this directory retain their original experimental content.
