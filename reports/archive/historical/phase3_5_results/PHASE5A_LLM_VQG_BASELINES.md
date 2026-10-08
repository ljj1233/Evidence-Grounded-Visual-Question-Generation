# Phase 5A — LLM/VLM generation baselines

Two inference-only, image-only VQG baselines evaluated on the frozen
development1000 manifest to test whether the visual-information-coverage
limitation observed for the SFT model is addressed by either (A) a stronger
external multimodal model or (B) better prompting/reasoning.

These are evaluation baselines only. No model was trained, no RFT data was
built, GRPO was not run, UVFC/Flat were not modified, and `final_holdout` was
not touched. No reference questions, answers, captions, candidates, or UVFC
scores were provided to either generator.

## 1. Setting

- Manifest: `outputs/evaluation/sft_greedy/val2014_sample1000.jsonl`
  (sha256 `36501b7f…9c97`; 1,000 unique image ids, 5,330 references).
- Same images, image loading, evaluation pipeline, and metrics as
  SFT/RFT/GRPO. No train images used; `final_holdout` untouched.
- Greedy, one question per image (1,000 questions per system).

## 2. Baseline A — GPT-5.6-Luna direct

- Model `gpt-5.6-luna` via the `LLM_JUDGE_BASE_URL` endpoint, temperature 0,
  max_tokens 1,024, one image + instruction per request.
- Instruction (verbatim):
  `Generate one high-quality visual question about this image. The question should be answerable from visible image evidence.`
- Output: `outputs/phase5a_baselines/luna_direct/{predictions.jsonl,manifest.json,runtime.json,cache.jsonl,integrity.json}`.
- Provenance: each response cached content-addressed (model + endpoint +
  instruction + image sha256); request id, returned model, usage, and timestamp
  retained. Failed requests were retried only for failed keys; no successful
  response was regenerated internally (the token-cap constant was changed once
  and is a non-binding safety bound, so all 1,000 final questions come from the
  1,024-token configuration).

## 3. Baseline B — Prompt-CoT VQG

- Model `Qwen3-VL-8B-Instruct` (`…/models/Qwen3-VL-8B-Instruct`), zero-shot,
  inference only (no LoRA, no fine-tuning), greedy, max_new_tokens 64.
- The exact chain-of-thought instruction (objects → attributes → relations →
  spatial → non-trivial details, then return only the final question) is stored
  in `outputs/phase5a_baselines/prompt_cot/manifest.json`.
- Output: `outputs/phase5a_baselines/prompt_cot/{predictions.jsonl,manifest.json,runtime.json,generation_cache.jsonl,integrity.json}`.
- Provenance: prompt hash `a999c7aa…`, model checkpoint hash `70bd1497…`
  (sha256 of `config.json` and `model.safetensors.index.json`), decoding
  parameters, seed 42, PIL backend.

## 4. Generation integrity

`outputs/phase5a_baselines/integrity_audit.json` (`all_passed = true`):

| check | luna_direct | prompt_cot |
|---|---:|---:|
| predictions | 1000 | 1000 |
| unique image ids | 1000 | 1000 |
| matches manifest image ids | yes | yes |
| empty questions | 0 | 0 |
| malformed (no `?`) | 0 | 0 |
| records with forbidden fields | 0 | 0 |
| exact reference-question matches | 0 | 0 |

## 5. Results

Corpus-level reference metrics plus image-mean SeeQ (gpt-5.6-luna variant) and
GPT-4o visual-support rates, 1,000 images each. SeeQ/support have full coverage
(1000/1000 valid after retry).

| Model | BLEU | CIDEr | Q-BLEU4-variant | Search | Coverage | Context | Spatial | Supported | Unsupported | Distinct-2 | Dup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT | 0.317 | 0.455 | 0.311 | 0.275 | 0.260 | 0.270 | 0.073 | 0.725 | 0.159 | 0.140 | 0.570 |
| GPT-5.6-Luna Direct | 0.080 | 0.119 | 0.077 | 0.421 | 0.445 | 0.271 | 0.257 | 0.930 | 0.036 | 0.352 | 0.006 |
| Prompt-CoT Qwen3-VL | 0.074 | 0.073 | 0.071 | 0.503 | 0.467 | 0.249 | 0.223 | 0.883 | 0.069 | 0.263 | 0.021 |

Image-paired bootstrap deltas vs SFT (5,000 resamples, seed 42; improvement
requires 95% CI lower bound > 0), from
`outputs/phase5a_baselines/comparison/baseline_table.json`:

| delta vs SFT | Search | Coverage | Context | Spatial | Supported | Unsupported |
|---|---:|---:|---:|---:|---:|---:|
| Luna | +0.146 [+0.134, +0.158] | +0.185 [+0.172, +0.199] | +0.001 [-0.007, +0.009] | +0.184 [+0.171, +0.197] | +0.205 [+0.174, +0.237] | -0.123 [-0.149, -0.098] |
| Prompt-CoT | +0.229 [+0.217, +0.240] | +0.207 [+0.192, +0.222] | -0.021 [-0.031, -0.013] | +0.150 [+0.135, +0.164] | +0.158 [+0.126, +0.190] | -0.090 [-0.116, -0.064] |

## 6. Interpretation

The SFT model was optimized for VQAv2 reference overlap: it has the highest
BLEU/CIDEr but also heavy mode collapse (Dup 0.570, Distinct-2 0.140) and the
lowest visual-centric scores. Both image-only baselines trade reference overlap
for visual coverage and groundedness, and do not optimize the reference
distribution, so their BLEU/CIDEr are low by construction and are not a
like-for-like quality deficit.

**Q1 — Does a stronger external VLM automatically solve visual coverage?**
Partially. Luna significantly raises Search (+0.146), Coverage (+0.185), and
Spatial (+0.184) over SFT and cuts unsupported questions from 15.9% to 3.6%,
but Context does not improve (+0.001, CI includes 0) and absolute Coverage is
0.445 — under half the rubric range. The limitation is reduced, not solved.

**Q2 — Does prompting improve Search/Coverage over SFT?**
Yes, significantly: Prompt-CoT Search +0.229 and Coverage +0.207 (both CIs
exclude 0), the highest Coverage of the three systems. But Context slightly
decreases (-0.021) and absolute Coverage is 0.467, so prompting improves
coverage dimensions without removing under-coverage.

**Q3 — Do these methods improve groundedness?**
Yes. Supported rises significantly for both (Luna +0.205, Prompt-CoT +0.158)
and Unsupported falls (Luna -0.123, Prompt-CoT -0.090). Luna is the most
grounded (93.0% supported, 3.6% unsupported, unobservable-target 3.2%);
Prompt-CoT retains a 6.4% unobservable-target rate. Ambiguous-reference errors,
which dominate SFT (6.4%), fall to 0.0% (Luna) and 0.8% (Prompt-CoT).

**Q4 — Does UVFC still provide additional value beyond stronger generation?**
Within this evaluation the motivation is not eliminated: neither external
baseline fully restores Coverage (max 0.467) and neither improves Context
reasoning, so image-only stronger generation/prompting does not fully address
visual behaviour coverage. Because these baselines are untrained and not
reference-matched, this does not by itself establish that a UVFC-style reward
adds value inside training; that comparison remains a pipeline question.

## 7. Decision

`outputs/phase5a_baselines/comparison/decision.json` (no ranking):

- `LUNA_HIGH_QUALITY_LANGUAGE` = **false** — Luna did not improve reference
  overlap (BLEU 0.080, CIDEr 0.119, both below SFT); it improved visual-centric
  coverage instead.
- `PROMPT_LIMITED_IMPROVEMENT` = **true** — Prompt-CoT improves Search,
  Coverage, Spatial, and Supported, but not Context, and absolute Coverage
  remains 0.467.
- `UVFC_MOTIVATION_SUPPORTED` = **true** — max external Coverage 0.467 < 0.60
  threshold and Context not improved.

## 8. Limitations

- SeeQ dims use the documented `gpt-5.6-luna` judge substitution, not paper
  GPT-5.4/five-questions; support uses GPT-4o. Both are independent judges over
  visible evidence only.
- Baseline questions are not trained on references, so reference-overlap and
  visual-centric metrics measure different objectives; no master score is
  defined and methods are not ranked.
- One question per image (greedy); diversity is lexical (Distinct-2, duplicate
  rate) with no sampled rollouts for these untrained baselines.
