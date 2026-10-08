# Final evaluation freeze (20261008)

- Systems: Full Ours `step_final`; Full Ours + DPO `step_00207` (adapter sha256 `32384eea…`, global_step 207).
- DPO pair manifest `UNIFIED_PAIRS.jsonl` sha256 `ae27e2ee…` (6633 pairs).
- Primary: GPT-4o Supported/Unsupported/Uncertain; SeeQ Coverage/Search/Context/Spatial.
- Reference: BLEU-4, CIDEr, Q-BLEU4-variant.
- Statistics: image-paired bootstrap, 10000 resamples, seed 42, 95% percentile CI.
- Frozen evaluators: SeeQ `gpt-5.6-luna` (prompt_hash 57a357e0…), GPT-4o `visual-support-phase23-v1.0`.
- Generation: greedy, T=0, max_new_tokens=32, seed 42, batch size 8, identical batching for both systems.
- Hard stop: no changes after final_holdout is opened.
