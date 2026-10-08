# Phase 2.1D — Anchor-First Controlled Visual-Support Benchmark

## Decision

**STILL UNDERPOWERED.** The frozen benchmark contains 49 pairs, while the
protocol requires at least 50 pairs in each of five failure categories. Counts
are missing entity 34, wrong attribute 3, wrong relation 8, ambiguous reference
0, and unobservable target 4. Therefore these results do not support a
verifier-superiority claim or a downstream training decision.

## Construction

The anchor inventory was restricted to the existing Phase 2.1 GPT-supported
labels and their calibration questions. It contains 667 supported anchors.
Eligibility was computed before mutation: missing entity 220, wrong attribute
50, wrong relation 50, ambiguous reference 2, and unobservable target 225.
The anchor-first proposal generator produced 384 unique proposals. The
independent GPT-4o validator was run with `temperature=0`, `max_tokens=300`,
16 concurrent requests, direct HTTP (`trust_env=False`, proxy variables
removed), and image-only evidence. Strict validation retained 49 proposals
(12.76%); 335 were rejected. The proposal pool was below the requested
250/category target because the available anchors and deterministic edit
patterns did not yield that many distinct candidates. The frozen pair file,
manifest hash, raw validator records, and rejection records are in
`outputs/visual_support_controlled_v2/`.

For comparison, Phase 2.1C retained 6/384 (1.56%). The higher anchor-first
acceptance rate improves construction efficiency, but it does not solve the
per-category sample-size requirement.

## Frozen verifier protocol

Both verifiers used the exact Phase 2.1 prompts from
`scripts/vvqg/run_local_visual_verifiers.py`, the same local
`Qwen2.5-VL-7B-Instruct` checkpoint, `bfloat16`, left padding, deterministic
generation (`do_sample=false`, `max_new_tokens=256`), PIL image objects, and
batch size 8. No prompt, parsing, or threshold tuning was performed after
the pair file was frozen. Holistic and Flat each scored 98 questions. One Flat
question was malformed; it is reported as a parse failure and excluded from
label denominators rather than treated as unsupported.

## Overall metrics

| verifier | positive coverage | positive preservation | negative coverage | negative detection | pair coverage | strict pair accuracy | ordered win/tie/loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Holistic | 1.000 | 1.000 | 1.000 | 0.878 | 1.000 | 0.878 | 43 / 6 / 0 |
| Flat requirement-aware | 0.980 | 0.875 | 1.000 | 0.980 | 0.980 | 0.854 | 41 / 7 / 0 |

Flat minus Holistic bootstrap deltas (5,000 resamples, seed 42) were −0.125
(95% CI −0.229, −0.041) for positive preservation, +0.102 (0.020, 0.204)
for negative detection, and −0.023 (−0.147, 0.100) for strict pair accuracy.
The paired strict-accuracy McNemar p-value is 1.0. These pilot estimates are
descriptive only because every category is underpowered.

## Error transitions and runtime

Among the 48 pairs scored by both verifiers, Flat fixed 4 Holistic pair errors,
introduced 5 regressions, was correct on both for 37, and wrong on both for 2.
Flat fixes were triggered by presupposition (1), reference ambiguity (3), and
query target (1) in the independent Flat evidence; trigger counts need not
equal pair transitions because one pair had a positive-side parse failure.

Holistic runtime was 123.97 s (1.265 s/question); Flat runtime was 153.97 s
(1.571 s/question). Both used an NVIDIA A800-SXM4-80GB, with peak allocated
memory 16.51 and 16.54 GiB respectively. Full runtime/config fingerprints are
in `runtime_report.json` and `verifier_manifest.json`.

## Reproducibility and next action

`controlled_pairs.jsonl` is frozen and hash-checked by the evaluator. The
construction logs, validation prompt hash, category counts, bootstrap,
McNemar, transition, trigger, and efficiency artifacts are all stored beside
it. The next run must improve anchor-first proposal coverage and produce a
new frozen benchmark; this result must not be used to tune either verifier or
to start GRPO.
