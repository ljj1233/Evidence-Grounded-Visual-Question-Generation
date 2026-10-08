# Phase 2.2C — Visual Information Contribution V2

This phase tested whether answer-token likelihood can measure visual information contribution while remaining separate from the Flat visual-violation detector. The result is **NO-GO for downstream reward use**. Stage A passed its controlled complementary-switch test, but Stage B did not show higher contribution for questions judged to require image information. Stage C was therefore not run.

## Stage A

The official validation complementary-pair archive was reused from the local acquisition and verified against the [official VQA download page](https://visualqa.org/download.html). The v2 output preserves the archive, raw mapping, SHA256 values, and extraction manifest in `outputs/visual_information_contribution_v2/official_pair_source_manifest.json`. The mapping contains 95,144 official pairs. After question/annotation/image joins, exact normalized question matching, distinct image and canonical answers, duplicate removal, and exclusion of final-holdout images, 77,122 pairs were eligible. Seed 42 sampled exactly 1,000 pairs with zero holdout overlap.

Canonical answers use `multiple_choice_answer` with the official VQA punctuation, digit, article, and contraction normalization. The frozen scorer is `/hpc2hdd/home/fye374/models/Qwen2.5-VL-7B-Instruct`. It computes mean causal log probability over answer tokens only under real-image, text-only, and same-size blank-image conditions. Prompt tokens do not enter the normalization. The implementation audit passed on 20 pairs before the remaining run; `stageA_debug20.jsonl` records token IDs, masks, offsets, image grids, and likelihood values.

The 1,000-pair results are:

| statistic | estimate | paired bootstrap 95% CI |
| --- | ---: | ---: |
| switch accuracy, both directions | 0.942 | [0.931, 0.953] |
| mean switch margin | 5.547 | [5.360, 5.738] |
| median switch margin | 5.439 | [5.167, 5.700] |
| positive G_text rate | 0.994 | [0.991, 0.997] |
| positive G_blank rate | 0.959 | [0.950, 0.968] |
| mean G_text | 6.860 | [6.681, 7.042] |
| mean G_blank | 3.736 | [3.611, 3.860] |
| Spearman(G_text, G_blank) | 0.804 | [0.785, 0.822] |
| sign agreement | 0.965 | [0.957, 0.973] |

The Stage-A decision is **GO**. These values are a controlled measurement sanity check, not exact mutual information and not a reward definition.

## Stage B

The sample contains 1,000 questions from 1,000 unique images, balanced exactly at 250 questions for each generation ID. The separate Qwen3-VL-4B-Instruct proposer produced usable short answers for all 1,000 questions. The primary scorer then computed `L_real`, `L_text`, `L_blank`, `G_text`, and `G_blank` for all 1,000 answers.

GPT-4o saw question text only, at temperature 0 and concurrency 32. Labels were 985 `image_information_required`, 11 `language_prior_sufficient`, and 4 `uncertain`; retained coverage after excluding uncertain was 99.6%.

| construct metric | G_text | G_blank |
| --- | ---: | ---: |
| required − prior mean difference | -0.725 [-2.330, 0.899] | -0.078 [-0.890, 0.825] |
| required − prior median difference | -0.707 [-2.170, 1.703] | -0.107 [-0.789, 0.358] |
| AUROC, required vs prior | 0.441 [0.286, 0.597] | 0.462 [0.347, 0.589] |

The language-prior group is very small, so this is exploratory. It does not support the intended interpretation that higher G identifies image-information-required generated questions.

Exact matches to existing Flat outputs were found for 979 questions. After excluding uncertain Flat states and uncertain language-prior labels, groups were: A (no violation + prior sufficient), 8; B (no violation + image required), 557; C (confirmed visual violation), 368. Mean G_text/G_blank were A 6.765/3.788, B 6.570/4.128, and C 7.357/3.821. The groups do not establish a complementary downstream signal.

## Runtime and decision

Stage A took 503.8 seconds with batch size 16 and 17.1 GiB peak allocated GPU memory. Stage B answer proposal took 72.3 seconds; scoring took 189.9 seconds with 17.0 GiB peak allocated memory; GPT labeling took 81.5 seconds. The measured linear estimate for 10,000 images × 4 questions is about 1.9 hours for proposal + real/text likelihoods, or 2.9 hours including the optional blank pass, excluding model-load and cache effects.

The final decision is **NO-GO**. Do not run the offline four-rollout audit, define final reward weights, train a reward model, or start GRPO from this result. All raw values and bootstrap samples are in `outputs/visual_information_contribution_v2/`.
