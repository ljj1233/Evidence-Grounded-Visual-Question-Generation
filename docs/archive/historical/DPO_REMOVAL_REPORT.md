# DPO REMOVAL REPORT (dpo_cleanup_v1)

The project has one current pipeline:

```text
VQAv2
  ↓
Qwen3-VL SFT
  ↓
Reward Calibration
  ↓
GRPO
```

The DPO / preference-construction direction has been permanently removed.
Deleted artifacts were **not** archived (no `archive/`, `legacy/` or
`outputs/archive/` copy was created). The deletion-log metadata directory
`outputs/dpo_cleanup_v1/` was later removed during the `outputs/`
reorganization; this report is the retained record. It originally contained
`deletion_inventory.json`, `deleted_paths.json`, `kept_reviewed_paths.json`,
`active_dependency_check.json`, `stale_reference_scan.json`.

## Removed

- DPO-family datasets / candidate pools / preference pairs (chosen/rejected, q+/q−).
- DPO predictions and evaluation results for `mdpo`, `pdpo`, `random`, `vanilla`.
- Old DPO/P-DPO/mDPO/V-DPO/RC-DPO/CADPO/Boundary-DPO/ECQ/CrossQG configs, scripts, source modules, tests.
- Old preference-era reports, docs and process notes.
- Obsolete DPO-era archives and stale symlinks.

Deleted: **85 paths, ~340.9 GB** (deletion log; see note above), including
`active/`, `baseline/`, `artifacts/`, `experiments/`, `historical/`, `runs/`,
`reports/`, `eval/`, `_archive/`, `archive/`, `tests/premise_dpo/`,
`tests/test_preference_labels.py`, `shared/preference_labels.py`,
`configs/active/`, `configs/baselines/`, `data/final_holdout_v1/`,
`data/plan_selection_v2/`, the old relation vocabularies, and the preference/DPO docs
(including the obsolete root `PLAN.md`, `RESEARCH_STATE.md`, `MAINLINE_STATUS.md`,
`PROGRESS.md`, `MIGRATION_REPORT.md`).

> Note: `_archive/` and `archive/` were already absent at deletion time (observed
> before this cleanup ran); they are recorded as `already_absent` in the inventory.

## Preserved

- Raw source data: `data/vqav2/` and `data/train_resources/image_split_manifest.json`.
- Current SFT dataset `src/vvqg/data/sft/` (`{image, question}` only).
- Running/current SFT checkpoints and logs: `outputs/checkpoints/` (protected, untouched).
- Current reward infrastructure: `src/vvqg/rewards/`, `src/vvqg/data/calibration/`.
- Active evaluation infrastructure: `src/vvqg/evaluation/`, `scripts/vvqg/evaluate.py`,
  `outputs/evaluation/evaluation_audit_v1/`, `docs/EVALUATION_*.md`.
- Generic future GRPO scaffolding: `src/vvqg/training/train_grpo.py`,
  `configs/new_vqg/grpo.yaml`, `src/vvqg/data/rollout/schema.json`.
- Generic shared infrastructure used by current tests: `shared/dataset/`,
  `shared/utils/`, `shared/evaluation/judge_roles.py` (docstring de-DPO-ed).

## Validation

- SFT data validator: `OK: schema, questions, images, splits, no eval leakage`.
- `pytest`: 83 passed.
- Import smoke: `import vvqg, vvqg.rewards, vvqg.evaluation` OK.
- Dependency check: no active code/config/data depends on a deleted DPO artifact.
- Stale-reference scan: every remaining term hit is an audit log, an explicitly
  historical/negation statement, or a common English word; no `review` hits.

## Remaining legacy references (intentional)

| Location | Why it remains |
|---|---|
| `README.md`, `CURRENT_STATUS.md`, `REPO_MAP.md` | state that DPO is **not** part of the pipeline (negation) |
| `docs/EVALUATION_ACTIVE_PROTOCOL.md`, `EVALUATION_METRIC_CONTRACT_V1.md`, `EVALUATION_IMPLEMENTATION_AUDIT_V1.md` | historical note that DPO-system data/archives were purged; active evaluation reference |
| `docs/archive/evaluation/*.md` | textual historical evaluation observations, explicitly archived |
| `outputs/evaluation/evaluation_audit_v1/*` | evaluation audit metadata/logs (pre/post tree, hashes, old-vs-recomputed) |
| `outputs/dpo_cleanup_v1/*` (later removed) | this phase's deletion audit logs; the directory was removed during the `outputs/` reorganization and this report is the retained record |
| `docs/DATA_PROTOCOL.md`, `docs/SFT_TRAINING_CONFIG.md`, `docs/REWARD_FRAMEWORK.md`, `PHASE1_DATA_REPORT.md`, `docs/active/training_environment.md` | contain negations ("no preference pairs") or the common word "chosen"/"rejected" |
| `shared/evaluation/judge_roles.py` | generic construction-vs-evaluation judge-role separation (kept) |
| `tests/test_reward.py`, `tests/test_judge_roles.py`, `src/vvqg/rewards/...` | unrelated uses of "rejected"/"chosen" |

These create **no active data dependency** on any DPO artifact.

## Repository state

The tree now has a single, unambiguous pipeline (SFT → Reward Calibration →
GRPO). Current trainable data is `{image, question}`; future reward-calibration
data is `image + generated question + evaluation/reward metadata`. There are no
q+/q−, chosen/rejected or preference-pair training datasets in the repository.
