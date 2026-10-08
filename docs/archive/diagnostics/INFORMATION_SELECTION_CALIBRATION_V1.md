# Phase 2.2A — Visual Information Selection Calibration

## Decision

**NO-GO for the current GPT pairwise protocol/local selector combination.**
The information-selection labels are not saturated, but 100-pair A/B swap
consistency is only 0.69, below the preregistered 0.80 stability threshold.
The exploratory local selector results do not establish superiority over the
trivial baselines. **The requested strict protocol was not fully completed:**
test was evaluated twice and the revised local prompt shortened the rubric.
No reward scalar, training objective, DPO data, or GRPO run was created.

## Frozen benchmark

The benchmark uses only `outputs/reward_calibration/sft_candidates_4x.jsonl`.
It samples two distinct unordered question pairs per image from 600 images,
with seed 42 and image-disjoint splits: 400 images/800 pairs for dev, 100/200
for val, and 100/200 for test. There are no reference answers, final-holdout
images, GQA examples, or preference-training fields. Twenty source images were
excluded because they did not provide two distinct normalized pairs. The pair
file and source hashes are in `pair_manifest.json`.

GPT-4o first judged visual support independently for A and B. Information
selection was compared only when both were supported. Overall, 862/1,200 pairs
(71.83%) were comparable; support failures were never forced into a preference.

The 50-pair pilot passed the schema/saturation/position gate: 43 both-supported
pairs, A/B/tie = 8/8/27, and first-position win rate 0.50. The pilot also
reported a longer-question win rate of 0.833 among unequal-length wins; this
was retained as a diagnostic and did not change the frozen rubric.

## GPT label audit

Across all 1,200 pairs:

| quantity | value |
|---|---:|
| both supported | 862 (71.83%) |
| at least one unsupported | 27.25% |
| uncertain support | 1.00% |
| A wins / B wins / ties | 149 / 130 / 583 |
| tie rate among comparable | 67.63% |
| largest comparison label share | 67.63% (not saturated) |
| longer-question win rate among unequal wins | 63.18% |

Non-tie reason codes were dominated by `more_substantive_visual_information`
(259/279), followed by `less_trivial` (16), `clearer_information_target`
(3), and an invalid `less_specific` reason (1) incorrectly normalized as
`less_generic`. This mapping is not semantically equivalent; the frozen record
and original raw response are retained for audit, not endorsed as a valid
less-generic attribution. Reason codes are diagnostic, not a reward.

The 100-pair position swap audit found preference consistency 0.69, support-A
identity consistency 1.00, support-B identity consistency 0.90, and 90/100
pairs remaining jointly supported. The full transition table is in
`position_consistency.json`.

## Local selector and baselines

The local model is Qwen2.5-VL-7B-Instruct, PIL backend, bfloat16, batch size 8,
without training. The initial full-rubric run copied schema placeholders on
690/862 items. Its test metrics were inspected before changing the prompt.
The second run uses a shortened rubric and has 862/862 parse-valid outputs.
Both runs are preserved, but the second run is exploratory: it does not satisfy
the requested same-rubric, test-once protocol. Calling the first run merely a
debugging failure does not restore a clean test boundary.

On the already-inspected test comparable set (143 pairs; exploratory only):

| system | macro-F1 | balanced accuracy | exact agreement | non-tie accuracy | coverage |
|---|---:|---:|---:|---:|---:|
| Local selector | 0.379 | 0.405 | 0.399 | 0.412 | 1.000 |
| Random | 0.299 | 0.341 | 0.329 | 0.333 | 1.000 |
| Length | 0.365 | 0.441 | 0.378 | 0.510 | 1.000 |
| Question-type | 0.430 | 0.425 | 0.671 | 0.157 | 1.000 |
| Always tie | 0.261 | 0.333 | 0.643 | 0.000 | 1.000 |

Image-cluster bootstrap (2,000 resamples, seed 42) gave local-minus-baseline
test macro-F1 intervals of −0.037 to 0.198 (random), −0.104 to 0.138
(length), −0.175 to 0.076 (question type), and 0.025 to 0.209 (always tie).
Thus local selection is not meaningfully above every trivial baseline.
Confusion matrices and dev/val metrics are stored in the output directory.

## Runtime and cost projection

GPT execution used direct HTTP, cleared HTTP/HTTPS/ALL_PROXY, concurrency 32,
temperature 0, cache/resume, incremental JSONL writes, retries, and failed-item
logging. It completed 50 pilot, 1,200 full-label, and 100 swap records; the
private endpoint tariff is unavailable, so no invented dollar estimate is
reported. Recorded token usage and completed-session timing are in
`runtime_report.json`; they are lower bounds, not full costs, because the
first interrupted full-label session lacked a runtime summary and unsuccessful
responses have incomplete usage accounting.

The final local run used an NVIDIA A800-SXM4-80GB, batch size 8, peak allocated
memory about 16.5 GiB, and 0.117 s/pair (8.52 pairs/s). A workload of 10,000
images with four rollout questions/image would require an estimated 20,000
pairs and 0.65 GPU-hours if all pairs were already support-comparable; six
pairs/image would be 60,000 pairs and 1.96 GPU-hours. This is only a projection
and was not executed.

## Interpretation and next action

The construct is operationally meaningful enough to produce a non-saturated
distribution and legitimate ties, but the judge is not stable under A/B swap.
The local selector has full coverage after prompt revision, yet its exploratory
macro-F1 does not separate it from all simple baselines. This phase therefore
does not authorize a selector reward, information-selection training, QRG, or
GRPO. A future attempt should first improve swap-stable judging and re-freeze a
new benchmark; the current test labels remain evaluation-only.
