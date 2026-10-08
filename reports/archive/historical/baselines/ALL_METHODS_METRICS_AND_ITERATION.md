# All methods — consolidated metrics and iteration analysis

Development1000 (frozen 1000 images), greedy 1 question/image. SeeQ dims are
image means in [0,1] (`gpt-5.6-luna` variant); support are GPT-4o rates; BLEU /
CIDEr / Q-BLEU4 / Dup are corpus; SemDiv is within-image semantic diversity
(Qwen3-Embedding-4B, sampled4x). Raw values: `outputs/analysis/all_methods_metrics.json`.

## Table

| Method | Init | Training | Reward anchor | Search | Coverage | Context | Spatial | Supported | Unsupported | BLEU | CIDEr | Q-BLEU4 | Dup | Distinct-2 | SemDiv |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot Qwen3-VL | base | none | — | 0.368 | 0.402 | 0.366 | 0.125 | 0.634 | 0.230 | 0.090 | 0.126 | 0.087 | 0.121 | 0.276 | 0.394 |
| SFT | SFT | SFT | — | 0.275 | 0.260 | 0.270 | 0.073 | 0.725 | 0.159 | 0.317 | 0.455 | 0.311 | 0.570 | 0.140 | 0.536 |
| Fixed-rollout P1strong | SFT | offline GRPO | SFT | 0.320 | 0.326 | 0.285 | 0.062 | 0.734 | 0.114 | 0.292 | 0.470 | 0.284 | 0.461 | 0.186 | 0.542 |
| Static UVFC GRPO | SFT | online GRPO | SFT | 0.332 | 0.332 | 0.255 | 0.071 | 0.814 | 0.094 | 0.326 | 0.519 | 0.319 | 0.499 | 0.174 | 0.537 |
| Dynamic UVFC GRPO | SFT | online GRPO | SFT-dyn | 0.292 | 0.279 | 0.250 | 0.076 | 0.758 | 0.118 | 0.312 | 0.467 | 0.306 | 0.516 | 0.152 | 0.535 |
| Base UVFC GRPO (SFT anchor) | base | online GRPO (91/1000) | SFT | 0.399 | 0.440 | 0.388 | 0.145 | 0.587 | 0.268 | 0.075 | 0.097 | 0.073 | 0.065 | 0.296 | 0.431 |
| Base UVFC GRPO (base anchor) | base | online GRPO (249/1000) | base | 0.382 | 0.406 | 0.367 | 0.130 | 0.622 | 0.240 | 0.088 | 0.128 | 0.086 | 0.110 | 0.282 | 0.412 |
| Prompt-CoT Qwen3-VL | base | none (CoT prompt) | — | **0.503** | **0.467** | 0.249 | 0.223 | 0.883 | 0.069 | 0.074 | 0.073 | 0.071 | **0.021** | 0.263 | — |
| GPT-5.6-Luna Direct | external | none | — | 0.421 | 0.445 | 0.271 | **0.257** | **0.930** | **0.036** | 0.080 | 0.119 | 0.077 | 0.006 | **0.352** | — |

## What the constructs separate

- **Reference alignment**: BLEU / CIDEr / Q-BLEU4 / Dup / SemDiv. The SFT family
  (SFT, Static, P1strong, Dynamic) dominates; it is trained on the VQAv2
  reference distribution and collapses (Dup 0.46–0.57, Distinct-2 0.14–0.19).
- **Visual coverage**: Search / Coverage / Spatial. The base family and the
  external/prompt baselines dominate; these were never tuned to references.
- **Grounding**: Supported / Unsupported. Luna > Prompt-CoT > Static > SFT >
  zero-shot > base-GRPO runs.
- **Diversity**: Luna/Prompt-CoT/base runs ≫ SFT family.

No method wins on all constructs; the frontier is a **reference-alignment ↔
visual-coverage** tradeoff, with grounding and diversity mostly tracking whether
the policy was allowed to stay diverse (base/prompt) or collapsed to short
VQAv2 questions (SFT).

## Prompt-CoT implications

1. **The behaviour is latent, not missing.** A zero-shot base model with a
   single CoT instruction reaches the best Search (0.503) and Coverage (0.467)
   of any local method, best local grounding (0.883 / 0.069), and near-best
   diversity (Dup 0.021) — with **no training and no reward**. So the coverage
   limitation is not a model-capability limit; it is an objective/instruction
   limit.
2. **The prompt is doing what the reward fails to do.** The CoT prompt states
   the target behaviour explicitly (objects/attributes/relations/spatial/
   non-trivial details + "answerable from the image alone, avoid external
   knowledge"). That last clause is exactly the grounding term UVFC lacks:
   UVFC raised Search/Coverage but *lowered* Supported; the prompt raises both
   together.
3. **The UVFC reward is coverage-only.** Its violation-first ranking is a
   tie-break, not a grounding penalty, so it can trade grounding for coverage
   (Base UVFC: Coverage 0.44, Supported 0.587). Prompt-CoT shows coverage and
   grounding are jointly achievable, so the reward — not the data — is the
   binding constraint.
4. **The anchor is a second-order effect.** Rebuilding the reward in base space
   improved grounding (Supported +0.035, Unsupported −0.028) and stability
   (91→249 steps) but barely moved Search/Coverage (−0.014…−0.034). The base
   family's coverage edge comes from the *policy behaviour*, not the anchor.

## Iteration options

Ordered by expected value / cost, all requiring explicit approval where they
change the frozen pipeline.

1. **Prompt-conditioned rollouts (prompt + reward).** Run Static UVFC GRPO with
   the CoT instruction for rollouts and evaluation. This starts the policy at
   the best-known local behaviour and tests whether the reward adds anything on
   top of it. Cheapest meaningful test; isolates prompt vs reward.
2. **Grounding-aware reward.** Add an explicit grounding term (support/Flat
   penalty or rejection) instead of a lexicographic tie-break. Directly targets
   the failure mode (Unsupported 0.24–0.27 for base GRPO). Use Prompt-CoT's
   clause as the spec.
3. **Distil the CoT behaviour into the SFT target.** Generate image-only CoT
   questions on the train manifest and train on `{image, question}` (the repo's
   canonical SFT schema). This removes prompt dependence and fixes the SFT
   mode collapse at its source (short VQAv2 targets). Changes the training
   direction — needs approval.
4. **Re-calibrate guards/anchors per initialization.** The degeneration guards
   were calibrated on SFT (initial dup ≈ 0.06); the base policy starts at 0.31
   and self-stops at 91/249 steps. A guard expressed relative to the init's own
   distribution (or a diversity floor) would let base-init runs be measured.
5. **Combine reference alignment + coverage.** SFT init gives alignment and
   grounding; base/prompt gives coverage. A short SFT-alignment pass on
   CoT-style questions (option 3) is the most direct way to get both.
6. **Keep the anchor correct if base init is used.** Base-anchored reward is
   the right construction; reuse `outputs/phase4b_online_uvfc/base_vfc` rather
   than the SFT anchor.

## Recommendation

The evidence points to the **prompt/objective**, not the model or the anchor, as
the bottleneck: the base model already produces coverage+grounding under a CoT
instruction, while UVFC GRPO trades grounding away for coverage. Start with
option 1 (prompt-conditioned rollouts) as a low-cost test, and pair it with
option 2 (grounding term) if the goal is to make reward-driven training match
the prompt's joint coverage+grounding.
