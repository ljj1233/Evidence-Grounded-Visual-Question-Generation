# Reward Calibration Corpus Report

## Frozen inputs and seeds

The canonical reward-calibration directory is outputs/reward_calibration/. Existing output files were moved from the legacy directory with SHA256 preservation recorded in outputs/reward_calibration/path_migration_manifest.json. The frozen reward-calibration image manifest contains 1,000 images. The final-holdout manifest contains 2,000 images and was not regenerated. All split and generation seeds are 42; 20260926 is not used.

## Dataset roles

SFT train/dev/test are the VQAv2 train2014 partitions. val_debug is an evaluation-development manifest. reward_calibration is the held-out VQAv2 val2014 image set used for candidate generation. final_holdout is a separate held-out val2014 set reserved for later evaluation. GQA remains future cross-domain evaluation.

## Sampling comparison

The comparison reuses greedy predictions and the existing 1x sampled predictions. Active metrics are BLEU-4, CIDEr, Q-BLEU4-variant, Distinct-2, and normalized excess duplicate rate. Q-BLEU4-variant is reported under that exact name because the registry documents it as a local variant, not standard Q-BLEU4.

| decoding | BLEU-4 | CIDEr | Q-BLEU4-variant | Distinct-2 | Dup | Unique | Unique rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| greedy | 0.317225 | 0.454971 | 0.310714 | 0.139524 | 0.570000 | 430 | 0.430000 |
| sampling 1x | 0.199243 | 0.313951 | 0.192300 | 0.252336 | 0.417000 | 583 | 0.583000 |

Unique is exact string uniqueness. These are diagnostic lexical/reference metrics and do not establish semantic or information-demand diversity. See outputs/reward_calibration/sampling_comparison.json.

## Candidate pool

The existing 4x file contains 4000 candidates for 1000 images, with exactly four records and generation IDs 0, 1, 2, 3 per image. There are 2621 exact unique questions and 2619 unique questions after strip+lower normalization; the normalized duplicate rate is 0.345250. Distinct-2 is 0.189762, empty rate is 0, structural malformed-question rate is 0, and mean question length is 5.620 tokens. The exact per-image unique probabilities are P(1)=0.001, P(2)=0.019, P(3)=0.136, P(4)=0.844. Deterministic examples for 100 images selected with seed 42 are in candidate_examples.jsonl.

## Viability

rollout_unique_rate = 0.955750. P(all four identical) = 0.001000; P(at least two unique) = 0.999000; P(at least three unique) = 0.980000; P(all four unique) = 0.843000. All images have the required four generation IDs.

## Provenance and runtime

generation_manifest.json records the model, LoRA adapter, processor, prompt, decoding parameters, seeds, hashes, counts, and recoverability notes. runtime_report.json is derived from the preserved generation and vLLM logs. The logged 4x throughput was 2.331 questions/s at client concurrency 64 on an NVIDIA A800-SXM4-80GB; this task did not rerun or tune generation.

## Disjointness and scope guards

disjointness_matrix.json records image-ID intersections among SFT train/dev/test, val_debug, reward calibration, and final_holdout. holdout_generation_guard.json records that no final-holdout generation was started and that no DPO/preference artifact was added. Phase 2.1/GRPO remains unstarted.

## Limitations

The active metrics are lexical/reference diagnostics. Unique-question counts, Distinct-2, and rollout viability measure surface-form candidate availability; they do not establish semantic or information-demand diversity. The comparison does not prove image grounding or explain model behavior. Runtime values are log-derived, and unrecoverable provenance fields remain marked unknown. GQA is reserved for future cross-domain evaluation.
