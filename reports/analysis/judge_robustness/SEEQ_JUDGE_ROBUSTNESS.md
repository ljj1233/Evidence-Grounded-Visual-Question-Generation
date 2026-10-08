# SeeQ-rubric judge robustness audit

This audit reuses the exact 200-record Phase 2.4 pilot: 100 Zero-shot and 100 final-SFT questions on matched images. Both judges receive the released SeeQ prompts, four 0–5 dimensions, the same image and question, and no system identity. GPT-5.6 Luna is the primary scalable judge; GPT-4o is a secondary judge only for robustness analysis. Neither is called an official SeeQ score: the paper uses GPT-5.4 and five questions per image.

The existing GPT-4o pilot outputs were reused. The source pilot had five GPT-4o timeouts; only those five records were retried, with no Luna rerun. Luna had 199 valid records and GPT-4o has 200 valid records after retry; 199 records are valid under both judges. The raw pairing is in `outputs/phase24/evaluation/seeq_judge_robustness/matched_luna_gpt4o.jsonl`.

| Dimension | Spearman | Pearson | Exact | Within 1 | MAE | Luna − GPT-4o |
|---|---:|---:|---:|---:|---:|---:|
| Search | 0.499 | 0.533 | 0.548 | 0.879 | 0.573 | 0.080 |
| Evidence coverage | 0.512 | 0.558 | 0.618 | 0.714 | 0.754 | -0.362 |
| Context | 0.707 | 0.675 | 0.734 | 0.925 | 0.357 | -0.176 |
| Spatial | 0.327 | 0.334 | 0.422 | 0.734 | 0.920 | -0.769 |

The primary scientific result is system ordering. SFT is below Zero-shot under both judges on every dimension in this pilot:

| Dimension | Luna Zero-shot | Luna SFT | Luna Δ | GPT-4o Zero-shot | GPT-4o SFT | GPT-4o Δ | Direction agrees |
|---|---:|---:|---:|---:|---:|---:|:---:|
| Search | 2.020 | 1.420 | -0.600 | 1.810 | 1.460 | -0.350 | yes |
| Evidence coverage | 2.111 | 1.280 | -0.831 | 2.460 | 1.660 | -0.800 | yes |
| Context | 1.889 | 1.310 | -0.579 | 2.110 | 1.460 | -0.650 | yes |
| Spatial | 0.525 | 0.300 | -0.225 | 1.510 | 0.860 | -0.650 | yes |

Bootstrap intervals use 5,000 record resamples with seed 42. The Search and Coverage direction agreement is stable in this pilot, while absolute calibration and spatial rank agreement differ materially. The paired SFT-minus-Zero-shot image-vector correlations are The paired SFT-minus-Zero-shot image-vector correlations after retry are 0.566 (Search), 0.386 (Coverage), 0.600 (Context), and 0.225 (Spatial) on 99 common images. This supports the direction comparison without implying judge equivalence.

No third judge was run: neither Search nor Coverage had a direction reversal. The primary baseline matrix therefore keeps GPT-5.6 Luna values and reports GPT-4o only here. Raw scores, bootstrap intervals, cache/runtime metadata, prompt hash, and input manifest hash are in `outputs/phase24/evaluation/seeq_judge_robustness/`.

Conclusion: `CALIBRATION_SHIFT_ONLY`. The relative Zero-shot versus SFT direction is preserved, but the agreement is not strong enough to claim judge equivalence, especially for Spatial and Coverage calibration.
