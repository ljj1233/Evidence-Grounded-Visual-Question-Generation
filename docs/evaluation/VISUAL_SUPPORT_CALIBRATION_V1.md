# Visual Support Calibration v1

## 1. Goal

This phase measures whether generated questions are supported by visible image evidence. It does not measure usefulness, difficulty, novelty, or information selection.

## 2. Calibration data

The frozen source is `outputs/reward_calibration/sft_candidates_4x.jsonl`. One candidate per sorted image was selected by balanced round-robin generation ID (250 from each of 0, 1, 2, 3). Seed 42 produces 1,000 unique image-question pairs split into development 600, validation 200, and test 200; the split files are image-disjoint.

## 3. GPT-4o pseudo-label protocol

GPT-4o received the image and exact generated question and returned premise consistency, reference clarity, visual answerability, failure type, and a short evidence reason. The prompt explicitly handles valid No answers, zero counts, non-presupposed query properties, external knowledge, and uncertain evidence. Requests used bounded concurrency 32, deterministic cache keys, incremental JSONL writes, and resume. All 1,000 labels completed with zero failed items.

## 4. Label distribution

Overall labels are supported 667 (66.7%), unsupported 178 (17.8%), and uncertain 155 (15.5%); overall entropy is 1.249815 bits. Premise labels are supported 670, unsupported 43, uncertain 287. Reference labels are clear 969, ambiguous 29, not applicable 2. Answerability labels are answerable 741, unanswerable 143, uncertain 116. No overall saturation above 95% occurred.

## 5. Holistic verifier

The frozen local evaluator is `/hpc2hdd/home/fye374/models/Qwen2.5-VL-7B-Instruct` (`qwen2_5_vl`). It directly predicts the three dimensions from image and question. The same checkpoint and decoding settings are used for both verifier variants.

## 6. Flat requirement-aware verifier

The question is decomposed into flat presuppositions, reference constraints, and a query target. Each requirement is checked against the image and aggregated with the deterministic three-dimension rule. No dependency graph, QRG, learned weight, or reward-model fitting is used.

## 7. Main comparison

On the frozen 200-example test split, holistic overall macro-F1 is 0.412458 with balanced accuracy 0.435084. Requirement-aware overall macro-F1 is 0.501979 with balanced accuracy 0.526518. Premise macro-F1 is 0.431253 vs 0.405513; reference macro-F1 is 0.231183 vs 0.229227; answerability macro-F1 is 0.407407 vs 0.552310 (holistic vs requirement-aware). Complete metrics and confusion matrices are in `outputs/visual_support_calibration_v1/test_metrics.json`.

## 8. Failure-type analysis

Test failure-type counts and verifier diagnostics are in `failure_analysis.json`. The requirement-aware system improves overall and answerability performance, while reference macro-F1 remains low; this is evidence for a structured-verification follow-up, not evidence that decomposition solves reference grounding.

## 9. Runtime / computational cost

GPT-4o completed 1,000 calls with concurrency 32. On the A800, the frozen local evaluator processed the 200-example test split in 211.6 s holistic (1.06 s/question) and 275.7 s requirement-aware (1.38 s/question), batch size 8. Runtime provenance is in `runtime_report.json`.

## 10. Go / No-Go decision

**GO.** Requirement-aware verification exceeds holistic overall macro-F1 by 0.089521 on the frozen test split, so a dependency-aware verification ablation is justified. This decision does not authorize GRPO or QRG implementation; the next experiment must remain a separate ablation.

## 11. Limitations

GPT-4o labels are pseudo-labels, not human ground truth. The calibration test is a development-stage verifier benchmark; final paper evaluation requires independent validation. Several local validation outputs were malformed and are excluded by coverage-aware metrics; the frozen test outputs have full coverage. No QRG contribution is claimed at this stage.
