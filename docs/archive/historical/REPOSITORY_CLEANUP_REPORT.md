# Repository Cleanup Report (Phase 4A.5)

```
Cleanup decision:              CLEANUP_PASS_WITH_ARCHIVED_LEGACY
Disk saved:                    3.75 GB (7 intermediate Phase 4A checkpoints)
Canonical experiments preserved: yes (Final SFT, Pure RFT 42/43/44, Replay 42/43/44, audit)
Superseded experiments archived: pre-fix random-LoRA Phase 4A (record only; no longer present)
Regenerable artifacts deleted: 7 checkpoint-250 optimizer directories
Code duplication reduced:      2 compare scripts (-120 LOC) + 2 evaluators (-6 LOC)
Tests:                         38 passed
Active Phase4B touched:        NO
```

## 1. Cleanup scope

Phase 4A family hygiene: `outputs/phase4a_uvfc_rft`,
`outputs/phase4a_uvfc_rft_multiseed`, `outputs/phase4a2_rft_replay`,
`outputs/phase4a3_rft_replay_multiseed`, `outputs/phase4a4_integrity_audit`, plus
Phase 4A scripts/tests. `outputs/phase4b_online_uvfc/**` and every module it
imports were treated as FROZEN/READ-ONLY. `outputs/same_backbone_baselines`,
`outputs/phase26`, and other phases were inventoried but not cleaned.

Phase 4B status: the Dynamic GRPO job was writing during the early window
(checkpoint `step_0200` at 00:08, train_log step 205 at 00:11) then stopped; no
Phase 4A.5 action touched it. Recorded in
`outputs/repo_cleanup/phase4b_running_job_dependencies.json`
(`phase4a5_actions_on_phase4b = NONE`).

## 2. Artifact classification

`outputs/repo_cleanup/ARTIFACT_STATUS.json`:

| Path | Status |
|---|---|
| `outputs/checkpoints/qwen3vl8b_lora_r16_vqav2_50k` | CANONICAL |
| `outputs/phase4a_uvfc_rft` / `_multiseed` | CANONICAL |
| `outputs/phase4a2_rft_replay` / `phase4a3_rft_replay_multiseed` | CANONICAL |
| `outputs/phase4a4_integrity_audit` | CANONICAL |
| `outputs/phase4b_online_uvfc` | ACTIVE_DO_NOT_TOUCH |
| Phase 4A `checkpoint-250` intermediates | REGENERABLE |
| pre-fix random-LoRA Phase 4A | SUPERSEDED (no longer present) |
| `seeq_style/round2_generation_invalid` | SUPERSEDED (retained) |
| `outputs/same_backbone_baselines` | UNKNOWN / out of scope (not cleaned) |

## 3. Archived experiments

`archive/superseded/phase4a_random_lora_init/README.md` records the invalid
pre-fix Phase 4A lineage (`get_peft_model` fresh random LoRA instead of Final SFT
continuation, status INVALID_SUPERSEDED). The invalid artifacts themselves no
longer exist — the run directories were re-run in place and their manifests now
record `init_adapter = .../checkpoint-final`. Invalid SeeQ round-2 generation
(`outputs/same_backbone_baselines/seeq_style/round2_generation_invalid`) is
retained in place with its `invalid_reason.json`.

## 4. Deleted smoke/debug payloads

None. Phase 4A smoke runs only produced compact metadata
(`smoke_report.json` for Replay42/43/44), which was **kept** as evidence
(init adapter, finite loss, 43,646,976 trainable params, image-affects-loss).
No bulky Phase 4A smoke model/prediction payloads existed.

## 5. Duplicate datasets removed

None deleted. `duplicate_files.json` found 37 byte-identical groups; almost all
are `checkpoint-250` copies removed with the checkpoint directories. Remaining
duplicates are canonical/required or raw evaluator outputs:
`*_per_image.jsonl` zero-shot files, support `system.jsonl` == variant output,
`phase4a2` `comparison/paired_bootstrap.json` == `evaluation/paired_bootstrap.json`
(both are Phase 4A.2 deliverables), and root vs `train/` config/manifest copies
(the Phase 4A.3 required layout). Per policy, raw evaluator outputs and canonical
manifests were not deleted.

## 6. Checkpoints removed

7 intermediate `checkpoint-250` directories (Pure42 uvfc_violation/uvfc_only,
Pure43, Pure44, Replay42, Replay43, Replay44). Each contained an
`adapter_model.safetensors` **byte-identical to its `checkpoint-final`** (verified
SHA256) plus `optimizer.pt`/`scheduler.pt`/`rng_state.pth`. Runs are complete and
audited; the protocol uses `checkpoint-final`; no report references the
intermediates. Saved 3,750,160,645 bytes. Every action logged to
`cleanup_actions.jsonl`. `checkpoint-final` adapters and all other artifacts kept.

## 7. Code refactors

- `src/vvqg/evaluation/common/reporting.py` (new): shared `norm_text`,
  `read_jsonl`, `read_json`, `dim_vector`, `support_vector`, `system_row`,
  `bootstrap_pair`. `phase4a2_compare.py` and `phase4a3_compare.py` now import it;
  re-run outputs are **byte-identical** to the pre-refactor copies.
- `src/vvqg/evaluation/common/cache.py` (new): `needs_request` centralizes
  cache-resume/retry selection; `seeq.py` and `visual_support/api.py` use it.
- `src/vvqg/training/train_sft.py`: added `require_rft_sft_checkpoint` — a
  `stage: rft` run without `sft_checkpoint` now fails loudly instead of silently
  building a fresh random LoRA. `stage: sft` (fresh adapter) is unchanged.
- Not touched (Phase 4B dependencies / frozen semantics): `online_uvfc_grpo.py`,
  `fixed_group_relative.py`, `policy_logprobs.py`, `online_flat.py`, `uvfc.py`,
  `generation/*`, UVFC/Flat/reward/metric/prompt code.

## 8. Test results

Before cleanup: 34 passed. After cleanup + new tests: **38 passed**
(`tests/test_repo_cleanup.py`: RFT guard, reporting helpers, retry policy).

## 9. Numerical regression

- Local metrics (BLEU-4, CIDEr, Q-BLEU4-v, Distinct-2, Dup) recomputed from frozen
  Replay42 predictions → 0 mismatches; identical to pre-cleanup.
- Replay42 deterministic 50-image greedy regeneration → **50/50 exact normalized
  match** with stored predictions (re-verified after the refactor).
- Canonical dataset SHA256 unchanged (`canonical_hashes_post.json`: zero drift).

## 10. Reproducibility status

Canonical predictions, evaluator caches (SeeQ GPT-5.6-Luna, GPT-4o support),
manifests, final adapters, and audit outputs are intact. `final_holdout` untouched.
The cleanup is reproducible and reversible for the deleted intermediates:
deterministic seed/config + `train_log.jsonl` + `checkpoint-final` remain.

## Outputs

`outputs/repo_cleanup/`: `phase4b_running_job_dependencies.json`,
`pre_cleanup_git_status.txt`, `pre_cleanup_git_diff.patch`,
`pre_cleanup_inventory.json`, `cleanup_plan.{json,md}`, `cleanup_actions.jsonl`,
`duplicate_files.json`, `canonical_hashes_{pre,post}.json`,
`code_duplication_audit.json`, `ARTIFACT_STATUS.json`, `disk_usage.json`,
`decision.json`. Docs: `docs/EXPERIMENT_INDEX.md`, this file.
