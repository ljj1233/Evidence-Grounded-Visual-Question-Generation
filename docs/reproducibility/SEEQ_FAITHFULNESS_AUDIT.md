# SeeQ Method-Faithfulness Audit

Status: audit only. No SeeQ pipeline, training code, dataset, checkpoint, or
retention strategy was modified to produce this document.

Scope: decide whether the same-backbone SeeQ implementation in this repository
can be called **SeeQ**, **SeeQ reproduction**, **SeeQ-style adaptation**, or
**SeeQ-inspired baseline**, by comparing it against the authoritative official
source at the recorded commit.

---

## 1. Authoritative sources

| Source | Reference |
|---|---|
| Official repository | `https://github.com/tianyi-lab/SeeQ` |
| Repository commit | `18e795293808c5cf9f69f1e980577ab40b3faa95` |
| Paper | "Self-Evolving Visual Questioner", `https://arxiv.org/abs/2606.13929` |
| Project source manifest | `outputs/phase24/evaluation/seeq_luna/official_source_manifest.json` |
| Project feasibility record | `outputs/phase24/external_baselines/feasibility.json` (`system_id: seeq`) |

Files inspected from the official repository at the recorded commit (fetched
from `raw.githubusercontent.com` / GitHub tree API):

* `README.md`
* `config/task_definition.json`
* `gen/scripts/generation_pipeline.py`
* `gen/run_cmd_sat_3b.sh`
* `utils/gen/selfjudge_utils.py`
* `utils/gen/evol_utils.py`
* `utils/gen/retrieval_utils.py`
* `gen/create_training/utils.py`
* full repository tree at the commit (31 tracked paths)

Additional official files already collected by this project (evaluation only):
`eval/question/valid_difficulty_eval.py`,
`eval/question/question_gene_embedding_diversity.py`,
`eval/run_qs_eval.sh`, `eval/run_qs_diversity.sh`
(see `official_source_manifest.json`).

A local clone recorded earlier (`/tmp/seeq_phase24.8BNk`) no longer exists; the
official source was re-fetched at the same commit. The official repository
ships **no checkpoints, datasets, caches, or generated outputs**
(`README.md`, "Inputs").

Official behavior that could **not** be recovered from the inspected files is
marked `UNKNOWN`; nothing below is inferred from memory.

`UNKNOWN` items:

* the retention/selection code that turns generation output into the training
  set (no such file exists in the released repository);
* the training loop `M0 → M1 → M2` (the repository contains generation +
  evaluation only, plus a conversation-formatting helper);
* optimizer, epochs, batch size, and whether official training uses LoRA or
  full fine-tuning.

---

## 2. Method comparison table

| Component | Official SeeQ | Current implementation | Status |
|---|---|---|---|
| Backbone | Qwen2.5-3B / Qwen2.5-7B / Qwen3-4B (`README.md`, `MODEL_TYPE`) | Qwen3-VL-8B-Instruct | ADAPTATION |
| Initial model M0 | `MODEL_PATH` optional; base model otherwise (`run_cmd_sat_3b.sh`) | frozen final VQAv2 SFT LoRA (`qwen3vl8b_lora_r16_vqav2_50k/checkpoint-final`) | ADAPTATION |
| Proposer model | generation model (`MODEL_PATH`/`MODEL_TYPE`) | M0 for R1; M1 intended for R2 | ADAPTATION |
| Answer generation | `selfjudge_utils.gene_answer()` | identical `ANSWER_PROMPT` | SAME |
| Rewrite model/reference | separate filter/rewrite model (`--filter_model_type`, default base `qwen25_3b`); multi-strategy RAG instructions (`evol_utils.generate_instructions_v2`, `retrieval_utils.broader/harder_question_prompts`) | frozen M0 adapter; one hand-written `REWRITE_PROMPT` | DIFFERENT |
| Filter/judge | `selfjudge_utils.self_judge`: reject compound questions, score entity easy/hard perception, count reasoning steps, check model-following; acceptance also requires difficulty improvement, `lexical_overlap < 0.82`, no duplicate simplified question | `FILTER_PROMPT` + `_accepted`: hard-constraint gate only (`Factuality.count==0`, `Ambiguity.count==0`, `Validity.answerable`) | DIFFERENT |
| Self-evolution loop | seed generation (5 boosters × 2 prompt versions) → standard evolution + RAG evolution → stage-2 on top-8 (`rank_seed_pairs_for_stage2`) → accepted questions added back to a growing sample bank | schedule `turns=2 × candidates=2` per image → propose → answer → rewrite → filter | DIFFERENT (simplified) |
| Round count | `NUM_TURN` (default 2) within a run; bank "bootstraps over multiple rounds" | 2 rounds (R1, R2) with adapter update between them | ADAPTATION |
| R1 generation | generation model proposes | M0 proposes | ADAPTATION |
| R2 generation | generation model may be swapped via `MODEL_PATH`; filter stays the base reference | intended M1 proposer; **the current `round2_generation` artifact was produced with the M0 proposer** | INVALID_FOR_SEEQ_EVOLUTION (current artifact) |
| Stable reference model | separate filter/rewrite model (default released base), not the round's proposer | frozen M0 adapter (`--reference-adapter`) | ADAPTATION |
| Synthetic QA triples | accepted rewrites with `accepted=true`, rejection labeled | accepted rows, `rejection_reason` recorded | SAME (concept) |
| QG training format | `create_training/utils.py`: `IQ` (image→question), `IQA` (image→question+answer), `VQG` (image+answer→question), `VQA` (image+question→answer) | `qg` (image→question+" "+answer), `qa` (image+question→answer) | ADAPTATION |
| QA training format | `VQA` / `IQA` | `qa` | ADAPTATION |
| Fact cache | `retrieval_utils.get_image_facts`, JSON cache keyed by `md5(image_path)`, feeds RAG instructions | scaffolding only; `fact_cache` empty in `run_seeq.py`; `rag_used: false` | DIFFERENT (absent) |
| Sample bank / RAG | `load_sample_bank` / `try_add_to_bank` / `sample_from_bank` with concept-tag novelty; drives `add_example_harder_v1` | scaffolding only; `sample_bank` empty; `rag_used: false` | DIFFERENT (absent) |
| Retention strategy | `UNKNOWN` (no retention code in the released repository) | deterministic image round-robin to exactly 10,000 accepted triples | UNKNOWN → project-defined |
| Deduplication | per-run `dict_ans` / `dict_qs_met` dedup; bank novelty filter (`is_novel`, Jaccard ≥ 0.6) | none across candidates; retained set may contain exact duplicates | DIFFERENT |
| Training objective | paper: train questioner + answerer on retained synthetic QA; released repo ships only the format helper | QG+QA SFT (LoRA) on 20,000 formatted records per round | ADAPTATION (trainer UNKNOWN) |
| Backbone update | `M_t` updated per evolution round (paper); repo has no trainer | `M0 → M1 → M2` LoRA adaptation | ADAPTATION |

---

## 3. Core algorithmic identity

These are the elements that decide whether the method is fundamentally SeeQ.
Reference: official `README.md` ("VLM acts as both the proposer and filter …
continuously bootstraps harder, more visually grounded questions") and
`generation_pipeline.py`.

| Core element | Official | Current | Verdict |
|---|---|---|---|
| Iterative self-evolution | yes | yes (2 rounds) | preserved |
| Proposer improvement across rounds | yes (paper; `M_t`) | R1 M0→M1 yes; R2 not yet exercised with M1 | partially preserved |
| Stable reference filtering | yes (separate base filter model) | yes (frozen M0 adapter) | partially preserved (reference role and criteria differ) |
| Difficulty-improvement acceptance | yes (`has_difficulty_improved`) | absent (hard-constraint gate only) | absent |
| Model-following acceptance | yes (`has_required_model_following`) | absent | absent |
| RAG fact cache + sample bank bootstrapping | yes (core of "harder" evolution) | absent in generation | absent |
| Structured rewrite (visible evidence / reasoning steps / intermediate inference) | yes (`validate_rag_question`, `require_structured`) | absent (free-form rewrite) | absent |
| Generated QA training (questioner + answerer) | yes | yes (QG + QA) | preserved |
| Multiple evolution rounds | yes (`NUM_TURN`, bank across runs) | yes (2 rounds) | preserved |

The **loop skeleton** (propose → answer → rewrite → filter → retain → train,
iterated) is preserved. The **difficulty-driven acceptance signal** and the
**RAG/fact/bank machinery** that the official README presents as the mechanism
for "harder" questions are absent or replaced by a hard-constraint gate.

---

## 4. Implementation adaptations

| Adaptation | Changes mechanism or implementation only |
|---|---|
| Qwen3-VL-8B instead of Qwen2.5-3B/7B / Qwen3-4B | implementation (backbone substitution) |
| LoRA instead of full fine-tuning | implementation (official trainer `UNKNOWN`) |
| Optimizer / scheduler / epochs (cosine, 1 epoch/round) | implementation |
| Data universe: VQAv2 50k train images vs official SAT JSON input | implementation / data |
| Retention implemented by project (official retention `UNKNOWN`) | implementation |
| Hard-constraint filter replaces `self_judge` + difficulty improvement | **mechanism** |
| IQA/VQA/QG format subset instead of full `IQ/IQA/VQG/VQA` | implementation (format coverage) |
| Simplify step defined but not called | **mechanism** (official calls `simplify_prompt()` before answer/score) |
| RAG fact cache / sample bank removed | **mechanism** |
| Task-family assignment via 17 `task_definition` names instead of 5 boosters × 2 prompt versions | implementation / prompt |
| Hardware (A800, cgroup quota) | implementation |

---

## 5. Critical question: retention strategy

Current change: `accepted[:10000]` → deterministic image round-robin to 10,000.

Verdict: **A — a bug fix preserving the intended objective, not a new
algorithmic component.**

Reasoning:

* The released official repository contains **no retention code** (`UNKNOWN`);
  there is therefore no official selection rule being violated. The project
  specification only requires `exactly 10,000 retained triples per round` and a
  `deterministic candidate order` (`docs/SAME_BACKBONE_BASELINES_SPEC.md`).
* The accept/reject mechanism (proposal → answer → rewrite → filter) is
  unchanged. Round-robin only re-orders **already-accepted** triples before the
  10,000 cut; it cannot add or remove an algorithmic step.
* `accepted[:10000]` in image-major order covered 2,789 / 9,184 images
  (1,257 unique questions, R1); round-robin covers 6,572 images (1,889 unique
  questions). The objective — 10,000 accepted synthetic QA triples — is intact.
* Because official retention is unpublished, the rule is **project-defined**,
  and it remains `UNKNOWN` whether it matches official practice.

Caveat: since it determines training-data composition and coverage, a reviewer
may treat it as a design choice rather than a reproduction detail. It does not,
by itself, change the method identity.

---

## 6. Critical question: filtering / rewrite logic

Official (`selfjudge_utils.self_judge`, `retrieval_utils`):

* reject compound questions (`_is_compound_question`);
* score perception difficulty from entity easy/hard categorization
  (`hard*2 + easy*1`);
* count reasoning steps;
* when constraints exist, require model-following compliance on the latest
  constraint;
* in the evolution loop, additionally require difficulty improvement
  (`has_difficulty_improved`), `lexical_overlap_ratio < 0.82`, non-duplicate
  simplified question.

Current (`pipeline.py`):

* `FILTER_PROMPT` returns a single JSON object with `hard_constraints`
  (Factuality/Ambiguity/Validity) and empty soft-constraint placeholders;
* `_accepted` accepts iff `Factuality.count == 0 and Ambiguity.count == 0 and
  Validity.answerable`;
* no compound-question rejection, no perception/reasoning scoring, no
  difficulty improvement, no model-following, no lexical-overlap gate, no
  dedup.

Comparison:

* Prompts: **not identical** (official `self_judge` uses
  `soft_constraints_extracted` + `reasoning_steps_and_count`; the project's
  `FILTER_PROMPT` resembles the disjoint `hard_constraints()` /
  `question_judge()` schema and uses only the hard part).
* Criteria: **not identical**.
* Failure labels: **not identical** (`hard_constraint_failed`,
  `invalid_answerability`, `missing_hard_constraints` vs official
  `multi_question`, `lexical_overlap_too_high`, `selfjudge_failed`,
  `difficulty_not_improved_enough`, `model_following_failed`,
  `insufficient_visible_evidence`, `low_specificity`, …).
* Acceptance decisions: **not equivalent** (the project gates on static
  answerability; official gates on difficulty *increase*).

Severity: **HIGH.** Filtering and rewrite are core identity components
("stable reference filtering", "continuously bootstraps harder questions"). The
project keeps a VLM-as-judge but changes *what is judged*.

The project's rewrite prompt (`REWRITE_PROMPT`) is a paraphrase of the official
`question_condition` block from `evol_utils.py` and drops the structured
`visible_evidence` / `reasoning_steps` / `intermediate_inference` output
required by `validate_rag_question(require_structured=True)`.

---

## 7. Critical question: training objective

Official:

* the released repository contains a conversation formatter
  (`gen/create_training/utils.py`) with four task types — `IQ` (image→question),
  `IQA` (image→question+answer), `VQG` (image+answer→question), `VQA`
  (image+question→answer);
* the trainer and its objective are `UNKNOWN` (not released);
* the paper states the model is trained as both questioner and answerer on the
  retained synthetic QA.

Current:

* `training.py.build_messages`: `qg` = image → target `question + " " + answer`
  (≈ official `IQA`); `qa` = image+question → target `answer` (official `VQA`);
* LoRA SFT, 1 epoch, cosine, lr 2e-5, batch 2;
* 20,000 formatted records per round (10,000 triples × 2 formats).

Verdict: **objective preserved in spirit (questioner + answerer on synthetic
QA), format is a subset adaptation.** The `VQG` form (answer→question) is not
used. Because the official trainer is unreleased, exact equivalence is
`UNKNOWN`.

---

## 8. Critical question: R2 evolution

Official: `M0` proposes R1 → train `M0 → M1` → R2 proposes with `M1` while the
reference filter/rewrite stays the released base model
(`README.md`, `run_cmd_sat_3b.sh`; paper `M_t`).

Current: the persisted `round2_generation` artifact was generated **before** M1
existed and therefore used the **M0 proposer** (command recorded in
`/tmp/seeq_round2_vllm.log`, `--adapter .../qwen3vl8b_lora_r16_vqav2_50k/checkpoint-final`
for both proposer and reference).

Verdict: **INVALID_FOR_SEEQ_EVOLUTION** for the current R2 artifact. The
`M0→M1→M2` sequence is only faithful once R2 is regenerated with the M1
adapter as proposer and the frozen M0 adapter as reference/filter. (M1 was
trained after R2 and is available at
`outputs/same_backbone_baselines/seeq_style/round1_m1`.)

---

## 9. Naming recommendation

### OPTION C — "SeeQ-style adaptation"

The iterative self-evolution skeleton, proposer update, stable reference
filtering, generated-QA training, and multi-round structure are present, but
the backbone, data, training, and — critically — the filter/rewrite criteria
and RAG/fact/sample-bank components are substantially adapted or absent.

* **Not** OPTION A ("SeeQ"): the core acceptance mechanism differs and a core
  component (RAG + difficulty-gated filtering) is missing.
* **Not** OPTION B ("SeeQ reproduction"): several official components are
  absent and official details (retention, trainer) are unreleased.
* **Not** OPTION D ("SeeQ-inspired baseline"): the algorithm is not
  *substantially different* — the evolution loop and training signal are the
  same family; the differences are component-level.

If a reviewer treats the RAG/fact/bank machinery and the
difficulty-improvement acceptance as essential to SeeQ's identity, the
classification degrades to OPTION D.

This is consistent with the project's own label:
`SEEQ_STYLE_POST_SFT_ADAPTATION` (`configs/baselines/seeq_style.yaml`,
`docs/SAME_BACKBONE_BASELINES_SPEC.md`).

---

## 10. Reviewer risk

### HIGH

* **Filter/acceptance criteria differ.** Issue: the project accepts on static
  hard constraints, official accepts on *difficulty increase* + model-following.
  Why it matters: the claimed mechanism ("self-evolves to harder questions") is
  not enforced. Mitigation wording: "VLM self-judging is retained but the
  acceptance gate is a documented hard-constraint adaptation; difficulty gain
  is not asserted."
* **RAG / fact cache / sample bank absent.** Issue: official "harder" evolution
  is driven by extracted image facts and a quality-weighted sample bank. Why it
  matters: a headline official component is missing. Mitigation: "RAG/fact/bank
  are not used in this adaptation; only the proposal/answer/rewrite/filter loop
  is retained."
* **R2 artifact used the M0 proposer.** Issue: the persisted R2 is not the
  `M1`-proposer evolution. Why it matters: an "M0→M1→M2" claim would be false.
  Mitigation: regenerate R2 with M1 before any claim, or drop the R2 claim.

### MEDIUM

* **Rewrite output is free-form**, not the structured
  `visible_evidence`/`reasoning_steps`/`intermediate_inference` object official
  validation requires.
* **Simplify step omitted** even though official calls `simplify_prompt()`
  before answer/scoring (identical prompt text is present but unused).
* **Backbone/scale and LoRA vs full-FT** differ; official trainer is
  unreleased.
* **Retention and coverage are project-defined** (official retention
  `UNKNOWN`); no deduplication across candidates.
* **Training format subset** (`IQ`/`VQG` absent).

### LOW

* **Answer prompt identical** and seed `42`; these are safely reusable.
* **Hardware** and evaluation protocol differences are already documented as a
  variant (`docs/SEEQ_JUDGE_ROBUSTNESS.md`).

---

## 11. Reproduction status summary

* Core loop: preserved.
* Acceptance signal and RAG/fact/bank: adapted/absent.
* Trainer and retention: official code unreleased (`UNKNOWN`).
* Current R2: `INVALID_FOR_SEEQ_EVOLUTION` until regenerated with M1.
* Recommended label: **SeeQ-style adaptation** (`SEEQ_STYLE_POST_SFT_ADAPTATION`).
