# Phase 2.5B — VIC Failure Attribution

## Decision

**`VIC_REFORMULATE`**. The answer-side contribution signal is real, but its ranking failure is not explained by hallucinated questions or only by the Qwen answer proposer. The strongest evidence is answer-space/difficulty bias plus residual construct mismatch with visual exploration. VIC was not modified and GRPO was not started.

## Coverage and protocol

The frozen Phase 2.5 pool was reused: 4,000 candidates, 1,000 images, seed 42. GPT-4o Task A produced 4,000/4,000 question-structure labels. Combined image-conditioned Task B+C produced 3,958 successful labels and 42 explicit failures; all 4,000 records remain in the output. The independent answer-proposer subset contains 1,000 records, 995 usable GPT-4o answers, and 5 explicit failures. Those 995 answers were rescored with the unchanged Qwen2.5-VL-7B VIC scorer and original likelihood prompts.

## Decomposition

| Variable | vs Search | vs Coverage |
|---|---:|---:|
| `L_real` | −0.151 [−0.181, −0.121] | −0.121 [−0.152, −0.090] |
| `L_text` | +0.169 [+0.137, +0.201] | +0.100 [+0.068, +0.132] |
| `G_text` | −0.230 [−0.261, −0.198] | −0.151 [−0.183, −0.120] |
| `G_blank` | −0.273 [−0.303, −0.243] | −0.218 [−0.248, −0.188] |

All intervals are 5,000-resample bootstrap intervals, seed 42, excluding missing evaluator scores. The negative gap is not driven only by unusually high image-conditioned likelihood: the text-only baseline is positively associated with Search/Coverage, and `L_real` is itself negative.

## Question-type and answer-space evidence

Small-closed questions have mean `G_text` 8.584, Search 1.362, Coverage 1.507. Medium-closed questions have mean `G_text` 4.904, Search 1.923, Coverage 1.929. Open questions have mean `G_text` 5.800, Search 1.503, Coverage 1.526. This supports an answer-difficulty/answer-space confound: high-G questions are disproportionately short closed-answer questions, while harder visual search is not higher for that group.

Within-type correlations do not explain the global reversal. Among types with at least 50 examples, Search correlations are negative for color (−0.050), count (−0.097), yes/no (−0.038), object identity (−0.149), scene context (−0.055), and text reading (−0.272); action (+0.128) is positive. Coverage correlations are mixed, with a weighted positive fraction of 0.56 across eligible types. The global Search relation therefore is not only a Simpson reversal.

## Evidence scope and support controls

| Evidence scope | n | mean G | Search | Coverage |
|---|---:|---:|---:|---:|
| Single local cue | 1,880 | 7.038 | 1.297 | 1.201 |
| Single object / multiple cues | 964 | 6.452 | 1.700 | 1.778 |
| Multi-object relation | 522 | 6.466 | 2.256 | 2.713 |
| Multi-region | 44 | 6.738 | 2.326 | 2.349 |
| Global scene | 524 | 8.228 | 1.566 | 1.934 |

The local-cue pattern is present for single-local versus multi-object questions, but global-scene questions also have high G; this is not a complete explanation. Support filtering does not rescue VIC: `rho(G,Search)` is −0.301 among independently Supported questions and −0.249 among Flat no-violation questions. The top-vs-bottom Search difference among Supported candidates is −0.485 [−0.566, −0.403]. Support/hallucination confounding is therefore **not supported**.

## Proposer substitution

| VIC version | Search correlation | Coverage correlation |
|---|---:|---:|
| Original Qwen3-VL-4B answer | −0.217 [−0.279, −0.155] | −0.151 [−0.217, −0.086] |
| GPT-4o answer | −0.074 [−0.134, −0.015] | −0.085 [−0.144, −0.027] |

The two G rankings have Spearman correlation −0.006 on 995 matched answers. GPT-4o substitution weakens the negative association but does not remove it, so proposer artifact is at most a weak contributor, not the dominant mechanism.

## Partial association and Flat failures

After controlling for question tokens, answer tokens, question type, answer space, and support state, the standardized G coefficient is −0.052 [−0.084, −0.019] for Search and −0.004 [−0.038, +0.028] for Coverage. The residual Search effect remains negative; Coverage becomes compatible with zero.

The first Flat pass reported 1,604 `FAILED` records, but that was not a completed inference result: the process stopped partway through, and the old cache logic treated every existing non-`ok` row as done while synthesizing missing rows as failures. The rerun retries all non-success rows and uses the frozen verifier limit (`max_new_tokens=256`, rather than the stale 64-token setting). It now has exactly 4,000 predictions and 4,000 states: 3,934 successful predictions and 66 genuine extraction/parse failures. The 66 are model-output format failures; no verifier prompt, threshold, or aggregation was altered, so this is not a behavior redefinition.

## Attribution table

| Hypothesis | Evidence |
|---|---|
| Question-type confound | MODERATE |
| Answer-difficulty confound | STRONG |
| Answer-proposer artifact | WEAK |
| Support/hallucination confound | NOT_SUPPORTED |
| Local-cue bias | MODERATE |
| Fundamental construct mismatch | MODERATE |

The allowed next step is reformulation research under a separately approved phase. This phase does not define a new reward, normalize G, tune coefficients, or authorize GRPO.

Artifacts are under `outputs/phase25b_vic_failure_attribution/`; the machine-readable decision is `final_attribution.json`.
