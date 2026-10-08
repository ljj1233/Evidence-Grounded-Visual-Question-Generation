# SeeQ Baseline Faithfulness + Collapse Attribution Audit

SeeQ baseline decision:
    SEEQ_STYLE_VALID_WITH_MAJOR_CAVEATS

Exact reproduction:
    NO

Core self-evolution loop preserved:
    PARTIAL

High-severity divergences:
    5

Rerun required:
    NO

Diversity collapse first becomes substantial at:
    PROPOSAL

Current paper-safe name:
    SeeQ-style Self-Evolution Adaptation (same backbone)

This audit is read-only with respect to the SeeQ pipeline, the M1/M2 weights, the
generation artifacts, Phase4B, and `final_holdout`. No model was trained, no round
was regenerated, no existing SeeQ file was modified. Only audit scripts, audit
JSON, and this document were created.

---

## 0. What was audited and how

| Item | Value |
|---|---|
| Official repo | `https://github.com/tianyi-lab/SeeQ` |
| Official commit | `18e795293808c5cf9f69f1e980577ab40b3faa95` (verified on the local clone `/tmp/SeeQ.qjyM`, `git log -1`) |
| Official paper | "Self-Evolving Visual Questioner", arXiv 2606.13929 (PAPER) |
| Local implementation | `src/vvqg/baselines/seeq_style/{pipeline,vllm_backend,training}.py`, `scripts/vvqg/{run_seeq,prepare_seeq_round,train_seeq}.py` (untracked in git) |
| Local artifacts | `outputs/same_backbone_baselines/seeq_style/` |
| New audit artifacts | `outputs/seeq_baseline_audit/` (+ this document) |

Evidence sources are tagged in the JSON artifacts as `PAPER`, `OFFICIAL_CODE`,
`README`, or `UNKNOWN`. Nothing is inferred from memory.

---

## 1. Official pipeline (reconstructed from code/paper)

The official repository is generation + evaluation only. It ships **no**
checkpoints, datasets, caches, retention code, or trainer.

Propose → (simplify → answer → self-judge) → evolve → filter, per image:

1. **Seed proposal** (`generation_pipeline.py`): **5 boosters × 2 prompt
   versions** = up to 10 seed draws per image, `do_sample=True`.
2. **Simplify** (`selfjudge_utils.simplify_prompt`) then **answer**
   (`gene_answer`) then **self_judge hard gate**.
3. **Standard evolution** (`evol_utils.generate_instructions_v2`):
   multi-strategy prompts (add-object, add-task, comparison, multi-hop), then
   `is_multi_question` reject, simplify, dedup, **lexical_overlap ≥ 0.82
   reject**, self-judge, **difficulty-improvement gate**.
4. **RAG evolution** (`retrieval_utils`): md5-cached **image facts** →
   `broader_question_prompts`; **sample bank** exemplars →
   `harder_question_prompts`; the rewrite must return structured JSON
   (`visible_evidence`, `intermediate_inference`, `reasoning_steps`) validated by
   `validate_rag_question(require_structured=True)`; **model-following gate**;
   accepted questions are added back to the bank.
5. **Stage 2** on `rank_seed_pairs_for_stage2(top_k=8)` (`NUM_TURN`).
6. **Training formatting** (`gen/create_training/utils.py`): four formats
   `IQ` (image→question), `IQA` (image→question+answer), `VQG` (image+answer→
   question), `VQA` (image+question→answer), random instruction per example.
7. **Training**: PAPER says questioner + answerer on retained synthetic QA;
   optimizer/epochs/LoRA are `UNKNOWN` (unreleased). **Retention is `UNKNOWN`.**
8. **Evaluation** (`eval/question/`): GPT judge, 4 visual dimensions scored
   0–5, plus `Qwen3-Embedding-4B` mean pairwise cosine distance.

Full stage table: `outputs/seeq_baseline_audit/official_pipeline.json`.

---

## 2. Local pipeline (traced from code + artifacts)

Same-backbone Qwen3-VL-8B + LoRA adaptation (`SEEQ_STYLE_POST_SFT_ADAPTATION`):

```
M0 = final VQAv2 SFT LoRA
R1: M0 proposer | frozen M0 reference | 4 proposals/image (2 turns x 2 candidates)
    -> answer -> rewrite -> hard-constraint filter -> round-robin 10,000 -> qg+qa 20,000 -> M1
R2: M1 proposer | frozen M0 reference | same stages -> M2
```

| Stage | Local |
|---|---|
| Proposal | one `PROPOSAL_PROMPT` + `" Task family: {task}"`, **greedy**, 4/image |
| Answer | `ANSWER_PROMPT` on the raw proposal, greedy |
| Rewrite | one free-form `REWRITE_PROMPT`, reference M0, greedy |
| Filter | `FILTER_PROMPT`; accept iff `Factuality.count==0 and Ambiguity.count==0 and Validity.answerable` |
| Retention | deterministic image round-robin to exactly 10,000 |
| Training | `qg` = image→`question + " " + answer`; `qa` = image+question→answer |
| Round roles | R1 M0/M0; R2 M1/M0 |

Absent: simplify, boosters/prompt versions, difficulty-improvement, compound
rejection, lexical-overlap, dedup, model-following, structured rewrite, RAG fact
cache, sample bank, `IQ`/`VQG` formats.

Full table: `local_pipeline.json`.

---

## 3. Fidelity matrix

Counts: `ADAPTED` 6, `MISSING` 6, `FUNCTIONALLY_EQUIVALENT` 3, `EXACT` 1,
`UNKNOWN` 1. The six `MISSING` stages are the official core of the "harder"
mechanism.

| Stage | Match | Severity |
|---|---|---|
| Proposal (10 seed draws, sampled) | ADAPTED | HIGH |
| Intent conditioning | ADAPTED | MEDIUM |
| Answer generation | FUNCTIONALLY_EQUIVALENT | LOW |
| Rewrite (multi-strategy + structured) | ADAPTED | HIGH |
| Simplify | MISSING | MEDIUM |
| **Difficulty comparison** | **MISSING** | **CRITICAL** |
| Validity filtering | FUNCTIONALLY_EQUIVALENT | LOW |
| Compound rejection | MISSING | MEDIUM |
| Lexical / model-following filter | MISSING | HIGH |
| Structured rewrite validation | MISSING | HIGH |
| Deduplication | MISSING | HIGH |
| RAG sample bank | MISSING | HIGH |
| Fact cache | MISSING | MEDIUM |
| Retention | UNKNOWN | MEDIUM |
| Training formatting (IQ/IQA/VQG/VQA) | ADAPTED | HIGH |
| Training objective | ADAPTED | MEDIUM |
| Round progression | ADAPTED | MEDIUM |
| Evaluation dataset / judge / diversity | ADAPTED / FUNCTIONALLY_EQUIVALENT / EXACT | LOW |

Full table: `fidelity_matrix.json`.

---

## 4. Model-role audit (verified by hash)

| Model | Path | Adapter SHA-256 |
|---|---|---|
| M0 | `outputs/checkpoints/qwen3vl8b_lora_r16_vqav2_50k/checkpoint-final` | `5fb66c36…934e0e` |
| M1 | `…/seeq_style/round1_m1` | `55545b7f…bc0e135` |
| M2 | `…/seeq_style/round2_m2` | `9d2f0055…db28af2` |

* **R1**: proposer M0, reference M0 (recorded command `logs/run_b4_after_b3.sh`
  passes `--adapter M0` and no `--reference-adapter`; the backend then defaults
  reference to the proposer). Reference = the same M0 adapter, not the base
  model.
* **R2 (valid)**: proposer **M1**, reference **M0** — `round2_generation_m1/generation_manifest.json`.
  Independent confirmation: 61.7% of R2 proposal strings carry the
  `question? answer` signature that only the `qg`-trained M1 emits, while R1
  proposals do not.
* `round2_generation_invalid/` (M0 proposer, written before M1 existed,
  timeline 09-29 21:22 vs M1 09-29 23:18) is **not** used.
* No stale adapter, no wrong model loaded in the valid path.

Details: `model_role_audit.json`.

---

## 5. Proposal intent and proposal-stage diversity

Intent = the 17 `task_definition` names in `config/task_definition.json`,
assigned round-robin as `TASKS[(index+seed+round)%17]`, 4 per image. All 17 are
present and balanced (≈2,161 proposals each). Intent metadata is carried in
`stage_a`/`stage_b`/`stage_d` but **dropped before training** (retained records
keep only `image_id/image/question/answer`) → `retained_per_task =
NOT_TRACEABLE`.

**Proposal diversity is already collapsed before any rewrite/filter** (this is
the key finding):

| | R1 proposal (question-part) | R2 proposal (question-part) |
|---|---:|---:|
| proposals | 36,736 | 36,736 |
| unique normalized | 2,874 | 1,902 |
| unique rate | **7.82%** | **5.18%** |
| normalized excess dup rate | 0.922 | 0.948 |
| distinct-2 (canonical) | **0.0159** | **0.0118** |
| top question | "What is the man doing?" ×2,890 (7.9%) | "What is the man doing?" ×5,823 (15.9%) |
| images with only 1 unique of 4 | 5,820 / 9,184 (63.4%) | 7,707 / 9,184 (83.9%) |

The proposer runs **greedy** with a **single prompt**, whereas official seed
generation **samples** with 10 diverse prompts. The M0 model, sampled with 4
questions/image on the evaluation set, produces 65.4% unique questions — the
collapse is injected at generation time by the local decoding/prompt choice, not
by the base model's capacity.

Artifacts: `proposal_diversity.json`, `intent_audit.json`.

---

## 6. Rewrite audit

| | R1 | R2 |
|---|---:|---:|
| proposal unique | 2,874 | 1,902 |
| rewrite unique | 2,823 | 2,088 |
| exact proposal→rewrite copy rate | **65.6%** | **81.6%** |
| mean lexical overlap (Jaccard, official rule) | 0.819 | 0.915 |
| mean token-edit fraction | 0.095 | 0.040 |
| distinct-2 proposal → rewrite | 0.0159 → 0.0163 | 0.0118 → 0.0131 |

The local rewrite is a free-form paraphrase that is **mostly an identity**
(67% of R1 rewrites and 82% of R2 rewrites are the proposal verbatim). The
official rewrite is multi-strategy, RAG-grounded, and must return structured
evidence/reasoning fields; official would reject any rewrite with lexical
overlap ≥ 0.82, i.e. `67%` (R1) / `43%` (R2) of the local rewrite stream.
Artifact: `rewrite_audit.json`.

---

## 7. Filter audit and acceptance funnel

Local funnel (identical shape both rounds):

```
proposal 36,736 -> answer 36,736 -> rewrite 36,736 -> [hard gate] -> accepted -> retained 10,000
R1: accepted 23,618 (64.3%); rejections: hard_constraint_failed 13,056, missing_hard_constraints 57, invalid_answerability 5
R2: accepted 25,789 (70.2%); rejections: hard_constraint_failed 10,889, missing_hard_constraints 55, invalid_answerability 3
```

The gate is **validity only**. It does **not** compare proposal vs rewrite, does
**not** require difficulty improvement, and the judge's
`soft_constraints.perception_difficulty/reasoning_difficulty` values are ignored
(they are `(0,0)` for ~87–90% and `(1,1)` for the rest of sampled judge outputs).
Artifact: `filter_audit.json`, `acceptance_funnel.json`.

### 7.1 Counterfactual deterministic official-filter replay (offline)

Applying the official **deterministic** filters in record order
(`is_multi_question`, `_is_compound_question`, `lexical_overlap ≥ 0.82`, global
simplified-question dedup) to the local stream:

| | R1 | R2 |
|---|---:|---:|
| local accepted | 23,618 | 25,789 |
| surviving accepted after official deterministic filters | **1,287 (−94.6%)** | **1,026 (−96.0%)** |
| of these, rejected by lexical_overlap ≥ 0.82 | 12,586 (53.3%) | 12,191 (47.3%) |
| of these, rejected by duplicate simplified question | 9,743 (41.3%) | 12,570 (48.7%) |
| distinct-2 of the surviving accepted set | 0.239 | 0.247 |

The deterministic official filters alone discard ~95% of the local accepted
stream; the survivors are ~10× more lexically diverse (distinct-2 0.24 vs 0.02)
than the local accepted set. The structured-rewrite and difficulty gates are
`NOT_REPRODUCIBLE_OFFLINE` (the fields are not generated).

---

## 8. Uniqueness and diversity funnels

| Stage | R1 N | R1 unique | R2 N | R2 unique |
|---|---:|---:|---:|---:|
| Proposal (question-part) | 36,736 | 2,874 | 36,736 | 1,902 |
| Rewrite | 36,736 | 2,823 | 36,736 | 2,088 |
| Accepted | 23,618 | 2,219 | 25,789 | 1,771 |
| Retained | 10,000 | 1,889 | 10,000 | 1,587 |
| M1 sampled output (4,000) | 4,000 | 756 | — | — |
| M2 sampled output (4,000) | — | — | 4,000 | 573 |
| M0 sampled output (reference) | 4,000 | 2,614 | 4,000 | 2,614 |

Distinct-2 / dup rate:

| Stage | R1 | R2 |
|---|---:|---:|
| Proposal | 0.0159 / 0.922 | 0.0118 / 0.948 |
| Rewrite | 0.0163 / — | 0.0131 / — |
| Accepted | 0.0221 / — | 0.0167 / — |
| Retained | 0.0462 / 0.811 | 0.0395 / 0.841 |
| M0 sampled4 | 0.1916 / 0.347 | 0.1916 / 0.347 |
| M1 sampled4 | 0.0532 / 0.811 | — |
| M2 sampled4 | — | 0.0417 / 0.857 |

The M0/M1/M2 `sampled4` values equal the published evaluation summary exactly
(canonical `vvqg.evaluation.metrics.diversity.distinct_2`), so the audit funnel
and the reported numbers use one implementation.

The three M-output series show a monotone loss of diversity M0 → M1 → M2, while
the retained set is already ~85–92% duplicates. Artifacts:
`uniqueness_funnel.json`, `diversity_funnel.json`, `retention_audit.json`.

---

## 9. Retention audit

* Rule: deterministic image round-robin, one accepted triple per image per pass,
  stop at exactly 10,000. Official retention code is **not released** →
  `UNKNOWN`; the local rule is project-defined.
* R1 accepted 23,618 (6,572 images) → retained 10,000 (6,572 images; 1,889
  unique questions). R2 accepted 25,789 (6,924 images) → retained 10,000 (6,924
  images; 1,587 unique).
* Round-robin *raises* the unique **rate** (it spreads over more images), but it
  **does not deduplicate**: the top retained question appears 1,080× (R1) /
  1,250× (R2), i.e. 10.8% / 12.5% of the training set is one template.
* It inherits the upstream collapse; it is not the origin of it.

Artifact: `retention_audit.json`.

---

## 10. Training format and budget

* Formats: `qg` 10,000 (`question + " " + answer`, unstructured ≈ IQA) and `qa`
  10,000 (≈ VQA). **`IQ` (image→question only) and `VQG` are absent.** The model
  is never trained on the pure image→question format used at evaluation.
* Budget (both rounds): LoRA r16/α32/dropout0.05, 1 epoch, per-device batch 2,
  no accumulation, lr 2e-5, cosine, warmup 1,000, 10,000 steps → 20,000
  exposures = 1 effective epoch. Official optimizer/epochs/LoRA are `UNKNOWN`.
* Overtraining check: effective epochs = 1, so this is **not** epoch
  overtraining, but each unique question is seen **5.3× (R1) / 6.3× (R2)** per
  epoch (data duplication). M1 loss 2.89 → 0.263; M2 loss starts already at
  0.192 (M1 nearly assigns its own outputs) and ends at 0.054. This is a
  **moderate** collapse contributor, not a distinct epoch-overfitting failure.

Artifacts: `training_format_audit.json`, `training_budget_audit.json`.

---

## 11. Inference consistency and evaluation protocol

* M0/M1/M2 evaluated identically: `development1000`, greedy 1 Q/image and
  sampled 4 Q/image (`T=0.8, top_p=0.9, max_new_tokens=32`), same image
  preprocessing.
* SeeQ adapters emit `question? answer`; predictions truncated at the first `?`
  uniformly (published summary). M0 output is unaffected. The raw outputs are
  retained.
* Judge is `gpt-5.6-luna` (documented variant, not official `gpt-5-mini`/GPT-5.4);
  the 4-dimension 0–5 rubric is functionally equivalent. Diversity uses the same
  `Qwen3-Embedding-4B`. Absolute official/local numbers are not comparable;
  only the within-local M0/M1/M2 trend is interpretable.

Artifact: `evaluation_protocol_audit.json`.

---

## 12. Are BLEU/CIDEr decisive?

No. The reproduction question is whether the official self-evolution trend is
replicated on the SeeQ-aligned dimensions. On the local result:

| Dimension | M0 → M1 → M2 | Trend vs official |
|---|---|---|
| Search | 0.2824 → 0.2894 → 0.2908 | ↑ (qualitatively matches) |
| Coverage | 0.2672 → 0.2722 → 0.2741 | ↑ (matches) |
| Context | 0.2694 → 0.2832 → 0.2804 | ↑ then flat (partial) |
| Spatial | 0.0808 → 0.0736 → 0.0680 | ↓ (does not match) |
| Supported (GPT-4o) | 0.737 → 0.701 → 0.701 | ↓ (does not match) |
| Diversity | 0.524 → 0.166 → 0.097 | severe collapse (does not match) |

So Search/Coverage improve slightly, Context is unstable, Spatial and
grounding-support degrade, and diversity collapses. BLEU-4/CIDEr also decline
but are reference metrics and are not the deciding signal.

---

## 13. Collapse attribution

Earliest substantial contraction: **PROPOSAL**.

| Source | Classification | Key numbers |
|---|---|---|
| A. Proposal collapse | **STRONG_EVIDENCE** | unique 5–8%, distinct-2 0.012–0.016, 63–84% of images yield one unique of four; greedy + single prompt |
| B. Rewrite homogenization | MODERATE_EVIDENCE | 66–82% exact copies; rewrite unique ≤ proposal unique |
| C. Filter homogenization | MODERATE_EVIDENCE | official deterministic replay keeps 1,287/23,618 (R1) and 1,026/25,789 (R2); hard-only gate admits generic/duplicate questions |
| D. Retention homogenization | MODERATE_EVIDENCE | 10k retained contain 1,889 (R1) / 1,587 (R2) unique; top template 10.8–12.5% |
| E. Training overfitting | MODERATE_EVIDENCE | 1 epoch but 5.3–6.3 unique-question exposures/epoch; M2 initial loss 0.192 → final 0.054 |
| F. Evaluation artifact | WEAK_EVIDENCE | collapse present in both greedy and sampled; normalization uniform |

The dominant contraction is at the proposal stage and is an implementation
divergence (greedy decoding + single prompt + no seed boosters), **not** expected
SeeQ behavior. Artifact: `collapse_attribution.json`.

---

## 14. Severity and decision

| Divergence | Severity |
|---|---|
| D1 difficulty-improvement acceptance missing | CRITICAL |
| D2 greedy + single-prompt proposal | HIGH |
| D3 lexical/dedup/compound/model-following filters missing | HIGH |
| D4 unstructured, mostly-identity rewrite | HIGH |
| D5 RAG fact cache + sample bank absent | HIGH |
| D6 no `IQ` (image→question) training format | MEDIUM |
| D7 simplify step omitted | MEDIUM |
| D8 no dedup in retention | MEDIUM |
| D9 rewrite-schema drift / untracked code | LOW |

**Decision: `SEEQ_STYLE_VALID_WITH_MAJOR_CAVEATS`.** The self-evolution loop is
preserved (propose → answer → rewrite → filter → retain → train, iterated, R2
generated by the M1 proposer against a frozen M0 reference, questioner+answerer
SFT on model-generated data), but the acceptance gate and rewrite no longer
implement SeeQ's difficulty-driven, grounded, structured core. The result
represents self-evolution but is **not** a fair direct comparison to SeeQ's
headline "harder questions" claim.

Artifacts: `divergences.json`, `decision.json`.

---

## 15. Rerun recommendation

A rerun is **not mandatory**: no wrong model was loaded, the valid R2 uses the
correct M1/M0 roles, and retention/training/evaluation followed the documented
spec. The current weights and results are a valid *SeeQ-style adaptation* and may
remain as an auxiliary baseline with explicit caveats.

If a fairer main-table SeeQ comparison becomes necessary, the **minimal**
correction (reuses all existing infrastructure) is:

1. **Sample instead of greedy at proposal/rewrite** (`do_sample=True`,
   temperature ≈ 0.8), and use multiple seed prompts/boosters — reason: proposal
   collapse is the earliest divergence; affected stage: proposal. *Reuses* the
   pipeline; only decoding changes.
2. **Re-enable the deterministic official filters** in `_accepted`
   (multi-question, compound, lexical-overlap ≥ 0.82, global dedup) — reason:
   these alone reject ~95% of the current accepted stream; affected stage:
   filter. *Deterministic, no new model.*
3. **Require difficulty improvement** using the judge's existing
   perception/reasoning scores (`has_difficulty_improved`), and make the rewrite
   answer the structured fields when `require_structured=True` — reason: this is
   the CRITICAL missing mechanism; affected stage: rewrite + filter.

Items 1–3 require regenerating R1 (and optionally R2) generation, but reuse the
manifests, retention, training, and evaluation code unchanged. Adding the RAG
fact cache/sample bank (D5) and the `IQ` training format (D6) are the next-highest
value but are larger changes.

---

## 16. Direct answers

**Q1. Can the current baseline be called "SeeQ"?** NO.

**Q2. Can it be called "SeeQ-style Self-Evolution Adaptation"?** YES.

**Q3. Did the local implementation preserve iterative M_t proposer → M0
reference → filtering → training?** PARTIAL. The loop, the M0→M1→M2 proposer
update, and the frozen M0 reference are present; the filtering step is a
validity-only gate, not the official difficulty/model-following filter.

**Q4. Is the low BLEU/CIDEr itself evidence of implementation failure?** NO.
They are reference metrics and decline only slightly; the relevant evidence is
the SeeQ-aligned dimensions and the diversity collapse.

**Q5. Did local M1/M2 reproduce the official qualitative improvement pattern?**
- Search: YES (slight, 0.2824→0.2894→0.2908).
- Coverage: YES (slight, 0.2672→0.2722→0.2741).
- Context: PARTIAL (0.2694→0.2832→0.2804, up then flat).
- Spatial: NO (0.0808→0.0736→0.0680, decreases).
- Diversity: NO (0.524→0.166→0.097, severe collapse).

**Q6. Earliest stage where diversity substantially contracts?** PROPOSAL.

**Q7. Strongest candidate missing/altered mechanism?** The difficulty-improvement
acceptance is CRITICAL, but the observed collapse is best explained by the
missing proposal-stage sampling/diversity (greedy + single prompt), compounded by
the absent lexical/dedup filters and RAG machinery.

**Q8. Is a rerun scientifically necessary before putting this baseline in the
main table?** NO, provided it is labelled and treated as an auxiliary
SeeQ-style adaptation. YES if it is to be presented as a fair SeeQ reimplementation.

**Q9. If a rerun is necessary, the minimum correction?** Items 1–3 of §15
(sampled proposals, deterministic official filters, difficulty-improvement +
structured rewrite). Reuse all manifests, retention, training, and evaluation
code.

---

## 17. Completion checklist

1. Official commit verified locally as `18e7952…` — done.
2. Official pipeline reconstructed from code/paper — `official_pipeline.json`.
3. Local pipeline traced from code/artifacts — `local_pipeline.json`.
4. Model roles M0/M1/M2 verified by SHA-256 — `model_role_audit.json`.
5. Proposal intents audited — `intent_audit.json`.
6. Rewrite audited — `rewrite_audit.json`.
7. Difficulty-improvement filter audited — `filter_audit.json`.
8. Validity/model-following/lexical/compound filters audited — `filter_audit.json`.
9. RAG/sample-bank/fact-cache audited — `rag_factcache_audit.json`.
10. Retention audited — `retention_audit.json`.
11. Training format audited — `training_format_audit.json`.
12. Training budget audited — `training_budget_audit.json`.
13. Proposal→rewrite→accepted→retained uniqueness funnel — `uniqueness_funnel.json`.
14. Diversity funnel — `diversity_funnel.json`.
15. Collapse attribution — `collapse_attribution.json`.
16. Evaluation-protocol differences — `evaluation_protocol_audit.json`.
17. Single decision — `decision.json`.
18. No existing SeeQ model/artifact modified.
19. Phase4B untouched.
20. `final_holdout` untouched.
