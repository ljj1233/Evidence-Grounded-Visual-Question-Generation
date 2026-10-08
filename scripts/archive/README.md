# Historical script pointers

Abandoned training launchers and one-off diagnostic helpers are excluded from this publication snapshot. Their source locations and deletion decisions are recorded in the [deletion log](../../SNAPSHOT_DELETE_LOG.tsv); conclusions and original research locations are in the [historical experiment index](../../docs/archive/HISTORICAL_EXPERIMENT_INDEX.md).

The following mechanisms remain available for the retained analysis reports:

| Script | Analysis |
|---|---|
| [K4 diversity](diagnostics/overnight_k4_diversity.py) | [K4 report](../../reports/analysis/offline_dpo_k4_diversity_20261008/SUMMARY.md) |
| [Parity audit](diagnostics/overnight_parity_audit.py) | [Parity report](../../reports/analysis/offline_dpo_parity_audit_20261008/PARITY_AUDIT.md) |
| [Same-type shortcut audit](overnight_same_type_audit.py) | [Shortcut report](../../reports/analysis/offline_dpo_same_type_pair_audit_20261008/REPORT.md) |

For the current pipeline use the [method code map](../../docs/method/CURRENT_METHOD_CODEMAP.md). Historical core modules under `src/` and their tests remain intact. Diagnostic scripts require the external artifacts named in their arguments/source.
