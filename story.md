# What Does the Answer Depend On?
# Modality Dependence in Image-Text Question-Answer Pair Generation

> **Protocol version note (2026-11-13):** The numbers and conclusions in this document come from the old protocol (generation prompt `plain-v1`, Qwen3-VL-8B, transformers 4.57.6). Results from the new protocol `plain-v2` (a fixed neutral English prompt, transformers 5.22.2) are written to `outputs/<run_id>/probe_report.md`; after final integration, a "Probe" section will be added to this document without changing the existing definitions and numbers.

Updated: 2026-11-13. This document is the main paper-narrative file: the research questions, existing observations, the relationship to prior work, the measures, and the methodological direction. Research definitions are in [VQG_Proposal.md](VQG_Proposal.md), the mathematical argument is in [docs/necessity_formal_argument.md](docs/necessity_formal_argument.md), and the literature index is in [docs/references.md](docs/references.md).

## Overview

- **Task:** Given text $T$ and an accompanying image $I$, generate a question $q$ (an answer $a$ may be generated at the same time).
- **Problem:** Generated image-text questions are often assumed by default to examine joint reasoning as long as they are relevant to the image and text and answerable under the full input. The authors test this default: relevance and answerability do not imply that both modalities are indispensable.
- **Observation:** On 1257 material groups from ScienceQA, about 93% of the valid questions generated in a single pass by Qwen3-VL-8B require only one modality; the verified lower bound on questions for which both modalities are necessary is only 19 questions (1.5%).
- **Prior work:** Existing approaches to judging which modality a question needs rely on instructions, responder performance, or uncertainty, rather than on the information structure of the answer.
- **Measure:** Modality necessity is determined by the identifiability of the answer, the dependence structure is characterized by partial information decomposition (PID), and a checkable certification is given per question.
- **Method:** a closed loop built out of the task's own criterion — a local answerability gate selects candidates, a repaired local three-condition verifier certifies them, and the loop is licensed only where $s>\ell$; generation-side circuit intervention is closed as load-bearing (negative, with mechanism).

## Task Setting

Given text $T$ and an accompanying image $I$, generate a question $q$ (an answer $a$ may be generated at the same time). Success for $q$ has three requirements:

- **Valid ($V$):** $q$ is clear, its presuppositions hold, and the answer is unique under the full material;
- **Text-necessary ($D_T$):** after removing $T$, the answer is no longer determined;
- **Image-necessary ($D_I$):** after removing $I$, the answer is no longer determined.

The three holding simultaneously is denoted $B$ (dual-necessary). "Necessary" means that after removing that modality the answer is no longer determined; "redundant" means that after removal the answer is still determined. If $a$ is also generated, it is further required that $a$ be equivalent to an independently established reference answer. The above judgments depend only on the answer to $q$, so the study examines $q$ first.

## Problem

> **Whether an image-text question can be used to examine joint reasoning depends on what information determines its answer, not on what it mentions.**

Given an image and an accompanying text, there are usually many reasonable questions: they are all relevant to the image and text, and all can be answered from the full input.

For example (a real case, `scienceqa_9369`): the text says two aquariums are placed by an open window, one of them has a plastic cover, and they are observed for 1.5 hours; the image is a table that itself labels the two aquarium conditions "uncovered/covered", with both rows starting at 23.7°C and ending at 17.7°C.

The model can ask:

> According to the table, after 1.5 hours, by how many degrees Celsius did the temperature of the uncovered and covered aquariums each drop?

This question is correct, relevant, and answerable; the answer is that both dropped by 6°C. But the question and the image alone already determine the answer: computing 23.7−17.7 row by row, the window and the plastic cover in the text play no part.

Contrast a type of question (a schematic of the type, not an original question): the text gives the correspondence between the dominance/recessiveness of a trait and genotype, and the image is a Punnett square that gives only the genotypes. When asking "the probability that the offspring exhibits the trait", without the text one does not know which genotypes exhibit the trait; without the image one does not know which genotypes are present.

Both questions are valid and both are relevant to the image and text; the difference is not in difficulty but in the source of information on which the answer depends:

- The first: image → numerical values → subtraction
- The second: text → correspondence between the trait and genotypes; image → genotype distribution; both together → probability

Thus a finer property than "relevance" emerges:

> **After removing one of the modalities, is the answer still determined?**

Relevance and answerability also hold for a question with a redundant modality; they do not imply "neither can be missing".

### Why It Matters

Image-text question-answer pairs are used to evaluate joint understanding of multimodal models, to construct training data, and to generate educational questions. If a question needs only one modality, it examines image reading or text reading; in evaluation it reflects single-modality ability, and in training it provides single-modality supervision. Distinguishing "a question involves two modalities" from "both modalities are necessary" is a prerequisite for using generated image-text questions for joint reasoning.

### Observation: Valid, but Rarely Dual-Necessary

On 1257 hint+image material groups from ScienceQA, Qwen3-VL-8B generated questions directly (a single generation pass, a single run). Evaluation was carried out by two model groups plus an arbitration model (the answer-identifiability evaluator). The evaluator automatically confirmed 28 questions; after question-by-question review by Claude, 19 were retained; this review is an AI review commissioned by the researchers, not independent human annotation.

| Result (Qwen3-VL-8B, N=1257) | Questions | Proportion |
|---|---:|---|
| Valid (clear question, presuppositions hold, grounded reference) | 962 | 76.5% of N |
| Of these, a single modality already suffices to determine the answer | 896 | 93% of valid questions |
| Verified dual-necessary (lower bound) | 19 | 1.5% of N, about 2% of valid questions |
| Remaining valid questions (16 with no counterexample in either direction, 22 pending in the evaluator, 9 accidental invariance and false passes) | 47 | — |

Source: `outputs/qwen3vl8b-necessity-v2/probe_report.md`. luna test reference batch: of 1100 answerable questions, 976 are single-modality-sufficient, and about 23 (1.8%) are verified dual-necessary. Spot-check results for the single-modality screening are given in E2.

> **Most generated questions are valid and answerable, but very few require both modalities.**

Redundancy is bidirectional: among the 894 questions judged single-modality-sufficient, 329 are sufficient with the image alone, 264 with the text alone, and 301 with each alone. The redundant side can be the text or the image.

### Requiring Only "Dual-Necessity" Is Also Not Enough

If, when generating questions, one only pursues making each modality "necessary", two kinds of deviation have appeared in the evaluation records:

- **Beyond the material:** in order to make every modality indispensable, the question presupposes content that the material does not support. Among the 268 invalid questions from the Qwen baseline, the common causes are unfounded presuppositions, missing decisive premises, and an ambiguous scope of what is asked.
- **Formally necessary, substantively concatenated:** juxtaposing two unrelated sub-questions, or restating the key information of the other modality in the question stem.

Making a single modality "fail" is easy: one need only ask a question that no material supports. Then neither single modality can answer it, which looks like "dual-necessary", but in fact the question itself is invalid. Therefore modality necessity must be built on "the question is valid and the answer is grounded", not on "cannot answer".

The goal is:

> **Each of the two modalities provides information needed for the answer, and the question does not exceed the evidence that the two together can provide.**

-------------------------------------------------------------------------

# Prior Work

Read in terms of "how to judge which modalities a generated question needs". All of the following are works whose methods and key sections have been read; works seen only as abstracts or search summaries are listed at the end. Citation details and links are in [docs/references.md](docs/references.md).

## I. Multimodal Question-Answer Pair Generation (Direct Competitors)

**SMMQG** (Wu et al., Findings EMNLP 2024; arXiv 2407.02233)

- Approach: first specify the modality combination M=(number of texts, number of tables, number of images), a retriever and an LLM select sources, and GPT-4-Turbo generates question-answer pairs. The generation prompt contains "The question should not be answerable if any chosen source passage is removed".
- Verification: Step 5 of the main text claims that three checks are performed, in which an LLM/LMM single call verifies "(a) the answer pair (b) each source in Z is necessary for answering".
- Actual situation: the verification prompt published in Appendix D (the ACL version is consistent with arXiv v2) has only two criteria—"the answer can be inferred from the provided material" and "the question matches the specified style"—with no criterion that "each source is necessary"; the paper does not report the pass rate of this verification step; the search found no public code.
- Human study measurements: fluency, style faithfulness, source relevance (93.0%, MMQA 85.8%), answerability (94.7%, MMQA 85.8%), answer correctness (92.7%, MMQA 80.0%). "Source relevance" asks "whether all sources are relevant to answering the question", not "whether it would be unanswerable if one were missing".
- The paper itself notes: the model may select sources yet not generate the question based on them ("One source of error arises when the question-generation model selects question sources that the generated question is not based on").
- Conclusion: in the public materials, necessity is required through the **instruction at generation time**; the description of the verification step is inconsistent with the published prompt; no measure assesses "whether the question really needs each source".

**FM2DS** (Abaskohi et al., Findings EMNLP 2025; code github.com/ServiceNow/FM2DS)

- Approach: GPT-4-turbo generates multi-hop multimodal questions according to a prompt (the prompt contains "unanswerable if any one of the documents is missing"); verification follows.
- Single-modality check (paper): GPT-4o answers using only the text, only the image, and only the table respectively; if it can answer, the question is discarded, and if it cannot, it is retained.
- Single-modality check (public code, `src/question/validate.py`): `is_question_multimodal` calls `generate_answer` separately for each modality, and as long as the returned string is non-empty it is judged "not multimodal"; the code has no refusal recognition and no comparison with the reference answer. `get_gpt_response` calls `gpt-4o-mini` and sends plain-text messages, with the image appearing in the prompt only as text in the form of a URL, file path, or Base64 placeholder, so the model cannot see the image content. Whether the public code is consistent with what was used in the paper's experiments cannot be determined from the public materials.
- Reported effects: among 1000 GPT-4o-generated samples, "question validation" rejected 131 (11.58%); this is a combined figure for the multi-hop check and the multimodal check, with no separate precision. Human annotation involved three annotators and 1200 samples, Fleiss' kappa 0.83, with each of four items—"correctness, multi-hop reasoning, multimodality, answer accuracy"—scored 0/1, and about 5% with a mean score below 0.75 were removed; no separate report is given for the necessity check.
- The paper itself notes: in the zero-shot setting the model "occasionally circumvents the validation step", the example being "Looking at the image of the Eiffel Tower, what engineering innovation...?"—the question uses the image, but the answer is in the page text, i.e., the image is redundant.
- Conclusion: the necessity check does exist in the design and in the code, with the criterion "can a single modality produce an answer"; the public materials give no precision for this check.

## II. Question-Generation Self-Play with Responder Behavior as the Signal

**VisPlay** (He et al., CVPR 2026; arXiv 2511.15661)

- The same model acts as questioner and reasoner, generating questions from unlabeled images (Vision-47K, about 47K images), trained with GRPO.
- Question-generation reward: `r = 1_valid · ReLU(r_unc − r_div)`, `r_unc = 1 − |2c−1|`, where c is the majority-vote confidence of the reasoner over m samples (maximal at c=0.5); `r_div` penalizes repetition by the BLEU cluster size of questions on the same image; `1_valid` only checks whether the output is wrapped in `<question>` tags. The pseudo-label is the reasoner's own majority vote; samples with 0.25≤c≤0.75 are kept.
- The prompt only requires the question to be "directly related to the image" and "must require analysis or reasoning, not just description", and the paper does not verify whether these requirements are met.
- There is no check of "whether the question needs the image"; the quality of generated questions is not evaluated directly, only downstream benchmark accuracy; the authors state "Our framework lacks a definitive verification method for the self-generated data."

**V-Zero** (arXiv 2601.10094)

- The questioner writes a four-option multiple-choice question and gives an intuitive answer at the same time; reward: `min(c,1−c)` when the intuitive answer agrees with the solver's majority vote, and `0.5·c` when it disagrees.
- The authors claim that the design "naturally prevents reward hacking": meaningless questions lead to inconsistent reasoning, a low c, and a low reward.
- There is no modality-dependence check; question quality was assessed only with Qwen3-VL-32B for format validity rate and difficulty, with no human validation.

**SPELL** (Yang et al., ICLR 2026; arXiv 2509.23863)

- Three roles—question generation, answering, and verification—targeting long documents, with difficulty-adaptive rewards.
- Appendix F.6 itself notes: without a curriculum and historical memory, the questioner "repeatedly propose similar, trivial questions about the same document to maximize the responder's success rate"; a verifier that updates only on its own majority vote exhibits "verifier self-delusion", requiring consistency checks against rule-based rewards.

**RGX** (Luo et al., arXiv 2103.07449)

- Three agents (text): question generation, answer identification, and answering; generated questions are divided into low-, medium-, and high-loss groups by answering loss, and only the low- and medium-loss ones are used for training.
- The authors explain: "If we only reward questions that are correctly answered, the generator will converge to a trivial local optima."

## III. Measures and Diagnostics of Modality Dependence

**Park et al.** (arXiv 2408.12763, video QA; the arXiv page lists no venue)

- Modality importance score `MIS = perf(q|M⁺) − perf(q|M⁻)`, i.e., the difference in accuracy between the subset containing a modality and the subset without it; only one MLLM, GPT-4 Turbo, is used.
- Complementary-question proportions: TVQA 2.1%, LifeQA 2.4%, AVQA 0.6%; humans (4 people, 197 questions) found no complementary questions; the kappa between humans and the MLLM is about 0.3 (Appendix A.3, for confidence-weighted classification); the authors state that the main limitation is the use of only a single MLLM.

**Multi-modal Data Spectrum** (Madaan et al., arXiv 2509.23499)

- "Shuffling" is done by replacing a modality with that of another sample, and dependence is estimated by majority vote of three Cambrian-1 models; 23 VQA benchmarks are covered; when only the question text is used, ScienceQA is 17.5% above random; the authors state that the limitation is reliance on multiple-choice benchmarks. The conclusions are at the dataset and subcategory level, with no per-question labels.

**M3IRT** (Uebayashi et al., ICLR 2026; arXiv 2603.02663)

- Model ability and question difficulty are decomposed into three components—image, text, and cross-modal—fitted from the response records of 24 VLMs on three benchmarks; used to screen out genuinely cross-modal questions.
- The authors state limitations and directions: multiple-choice only; "developing question-generation methods that control cross-modal difficulty" is a future direction.
- The components are estimated from the response records of a group of models and are statistics relative to that model population.

**Liang et al.** (NeurIPS 2023; arXiv 2302.12247)

- Uses partial information decomposition (PID) to characterize the redundancy, uniqueness, and synergy of multimodal **datasets**, proposes an estimator scalable to high dimensions, and validates it with synthetic data and human annotations.

## IV. Common Structure and Consequences for This Task

When judging, per question, which modality a question needs, these works all use the **behavior of the responder or the generator** (Liang et al. are at the dataset level and do not judge per question):

| Signal | Work | What it actually measures |
|---|---|---|
| Instruction at generation time | SMMQG, FM2DS | Whether the question text meets the requirement |
| Whether an answer can be produced under a single modality | FM2DS | Whether a model can answer at this moment |
| Responder uncertainty | VisPlay, V-Zero, SPELL, RGX | The difficulty of the question for the current responder |
| Accuracy difference with and without a modality | Park, Madaan | Model ability plus prior |
| Response records of a model population | M3IRT | Difficulty components relative to that population |

Information structure (what determines the answer) and behavioral signals are not in one-to-one correspondence. The following three points are consequences derived from the above approaches; the next section gives a formal account, and experiments are used to test them:

1. **Difficulty rewards cannot separate "difficult" from "necessary".** VisPlay gives the highest reward at c≈0.5. A question whose answer can be determined by a single modality but which is merely hard to compute, and an ambiguous question whose answer the material cannot determine at all, may both obtain c≈0.5. The responder always sees the full input, so the reward cannot see whether a modality is redundant; moreover, VisPlay and V-Zero both have only images as input, so "which modality bears the answer" is undefined for them.
2. **"A single modality cannot answer" does not equal "necessary".** The FM2DS-style retention rule would retain invalid questions: about 21% (268/1257) of the questions in the Qwen baseline are themselves invalid, and they cannot be answered under any single modality.
3. **Trivial questions are an attractor for the questioner.** Both SPELL and RGX document that the questioner converges to trivial questions; the 93% single-modality sufficiency observed in this task points in the same direction. This baseline is one-pass generation rather than self-play, and whether the two mechanisms are the same has not yet been tested.

The M3IRT authors explicitly list "question-generation methods that control cross-modal difficulty" as an open direction, which is consistent with the goal of this study.

## Not Yet Read in Full

CaptionQA (blind-answer screening), VeriSciQA, MedVLSynther, MuMuQA, CRIT, QGEval, See First, Answer Later (blind reader as reward). These have so far been seen only as search abstracts or excerpts and are not used as prior conclusions.

-------------------------------------------------------------------------

# Core Research Questions

> **How can generated image-text question-answer pairs be made necessary in both modalities without exceeding what the material can support?**

## Challenge 1: How to Certify Which Modalities a Question Needs?

Behavioral signals (instructions, single-modality answerability, uncertainty, accuracy differences) are indirect estimates of information structure, and their evidential strength is asymmetric (see Proposition 15 in the next section). What is needed is a judgment that can give a "necessary/redundant" certification, is auditable, and does not depend on a single responder. The next section provides this judgment.

## Challenge 2: How to Increase the Dual-Necessary Proportion Without Exceeding the Material Evidence?

The questioner's attractor lies in single modalities; difficulty pressure alone pushes questions toward harder questions on the same modality; and "making a single modality fail" alone rewards invalid questions. What is needed is a signal that both identifies redundancy and does not reward invalidity. The methodological direction is given in the last section.

-------------------------------------------------------------------------

# Measure: Modality Necessity Based on Partial Information Decomposition

This section answers Challenge 1. The full setup and proofs are in the mathematical reference [Section 10](docs/necessity_formal_argument.md); the conclusions are given here. The criterion itself is consistent with Sections 1–7 of the mathematical reference; this section merely restates it in information-theoretic language and adds partial information decomposition (Partial Information Decomposition, PID; Williams & Beer 2010; Bertschinger et al. 2014).

**Per-question level.** Fix a question $q$, take any full-support distribution on the legitimate worlds $\mathcal W(q)$, and denote the residual uncertainty when the text is removed and the actual image $i$ is retained as $U_T(q,i)=H(A\mid I'=i)$.

> **Proposition 13.** $D_T=0\iff U_T(q,i)=0$, and zero versus non-zero is independent of the distribution chosen.

Equivalently, the text is redundant if and only if, given the question and the image, the text no longer brings any information about the answer.

**Question level.** The two-source PID decomposes $I(A;T',I')$ into four atoms—redundancy, text-unique, image-unique, and synergy—and the consistency equations give

$$
I(A;T'\mid I')=\mathrm{Unq}_T+\mathrm{Syn},\qquad I(A;I'\mid T')=\mathrm{Unq}_I+\mathrm{Syn},
$$

and $I(A;T'\mid I')=\mathbb E_{i'}\,U_T(q,i')$.

> **Proposition 14.** (a) $\mathrm{Unq}_T+\mathrm{Syn}=0$ if and only if, for any legitimate image, the text brings no information about the answer. (b) If $D_T=1$ in the actual material, then $\mathrm{Unq}_T+\mathrm{Syn}>0$. (c) The converse does not hold.

The evaluator's evidence (sufficient solution, witness pair) acts on the actually retained material, corresponding to the per-question $U_T$; the PID atoms describe the dependence structure of the question over the entire legitimate world. The binary judgment of $D$ uses only conditional mutual information and does not depend on which redundancy measure is chosen.

| Question type (question level) | PID condition | Example |
|---|---|---|
| Text-redundant | $\mathrm{Unq}_T+\mathrm{Syn}=0$ | The aquarium question whose table is self-labeled |
| Image-redundant | $\mathrm{Unq}_I+\mathrm{Syn}=0$ | The text already gives all the information determining the answer, and the image is merely illustrative |
| Synergistic dual-necessary | $\mathrm{Syn}>0$ | The Punnett-square question: the text gives the rule and the image gives the genotypes, so either modality alone carries almost no information about the answer |
| Concatenated dual-necessary | $\mathrm{Unq}_T>0,\ \mathrm{Unq}_I>0,\ \mathrm{Syn}=0$ | Two independent sub-questions juxtaposed |

Which type a question specifically belongs to requires selecting a PID measure and estimating it; the study uses only the binary necessity, and type attribution serves only for interpretation.

**Evidential strength of responder signals.** Suppose the responder cannot see the text and outputs $\hat A$. Fano's inequality gives $P_e\ge (U_T-1)/\log_2|\mathfrak A_T|$.

> **Proposition 15.** (a) If $D_T=1$, the probability that any responder who cannot see the text answers correctly is at most $1-(U_T-1)/\log_2|\mathfrak A_T|$: the more spread out the answer distribution, the less likely a correct guess. (b) If $D_T=0$, the probability of a correct answer depends only on the responder's ability. (c) There is no analogous bound for "failure".

Therefore a single match that agrees with an independent reference answer and whose derivation can be checked is strong evidence for $D_T=0$; a single failure has limited evidential strength. Certification of necessity comes from witness pairs, not from failures. This explains the common structure of prior work: using "cannot answer" as "necessary" relies on the side that lacks evidential strength.

**Relation to existing PID work.** Liang et al. compute PID statistics with an estimator at the dataset level; here the decomposition is at the level of a single question, and the per-question zero versus non-zero is certified by a sufficient solution or a witness pair rather than estimated.

**Aggregate quantities.** For a fixed $N$ questions, with $n_1$ confirmed dual-necessary and $n_\bot$ pending, the true count $m$ satisfies $n_1\le m\le n_1+n_\bot$. Let $p$ be the precision of the single-modality screening on the $n_c$ valid questions that it removes, and let the number of valid questions be $n_V$; then

$$
\#\{\text{at least one modality redundant}\}\ \ge\ p\,n_c,\qquad \#\{B=1\}\ \le\ (n_V-n_c)+(1-p)\,n_c .
$$

-------------------------------------------------------------------------

# Question-Validation Experiment: Are Generated Image-Text Question-Answer Pairs "Valid but Rarely Dual-Necessary"?

## E1. Scale (existing)

Qwen3-VL-8B, $N=1257$, $n_V=962$: confirmed dual-necessary $n_1=19$ (1.5% of N); a single modality already suffices to determine the answer for 896 questions, 93% of $n_V$; removed by the pre-screening $n_c=894$. Control (luna, test reference): 1100 answerable, 976 single-modality-sufficient, about 23 verified dual-necessary (1.8%).

## E2. Screening Precision (existing)

From the $n_c=894$ questions, 40 were randomly sampled with stratification, and Claude Haiku answered them blind (without seeing the evaluator's conclusion); inconsistent cases were adjudicated by Claude looking at the image: 40/40 indeed had at least one redundant modality, $\hat p=1$. The Wilson 95% lower bound is $p_L\approx 0.912$.

Limitations: the reviewer is Claude (an AI review commissioned by the researchers, not independent human annotation); the sampling is stratified by (judgment type × material template) and the Wilson interval is approximate; the check is for "at least one modality is redundant" and does not verify one by one which specific modality.

## E3. Substituting into the Aggregate Bounds (directly computable)

$$
\text{redundant share of valid questions}\ \ge\ \frac{p_L\,n_c}{n_V}=\frac{0.912\times 894}{962}\approx 84.8\%,
$$

$$
\#\{B=1\}\ \le\ (962-894)+(1-0.912)\times 894\approx 68+78=146,
\quad\text{i.e.}\ \le 11.6\%\ \text{of}\ N .
$$

In the 95% Wilson sense, at least about 85% of valid questions have one redundant modality; even in the worst case, dual-necessary does not exceed about 12%; the verified lower bound is 1.5%. The interval is wide because the upper bound counts all the worst-case false removals among the questions removed by screening as dual-necessary, whereas none was found in the audit sample.

Limitation: validity $V$ (962) is given by the evaluator and has not been audited separately; another 27 questions whose validity awaits adjudication are not counted.

## E4. Direction of Redundancy (counts existing, direction labels not checked one by one)

Among the 894 questions removed by screening (evaluator screening judgments): image alone sufficient 329; text alone sufficient 264; each alone sufficient 301. Text-redundant at least $329+301=630$ questions, image-redundant at least $264+301=565$ questions.

## E5. Structure of Dual-Necessary Questions (existing, small sample)

Of the 19 questions, 14 are Punnett-square questions (formally compatible with the synergistic type), and 5 are yes/no exchange questions of the form "the text gives what is wanted, the image gives what each has"; no per-question PID estimation was performed. The sample comes from one dataset and is descriptive only.

## Follow-up

- Measure reliability: stratified sampling of the "remaining" questions to estimate the miss rate; the review precision for confirmed items is currently 19/28.
- Stability across more generators, datasets, and repeated runs.
- Contrast the default prompt with a prompt that explicitly requires both the image and the text to be necessary, testing "whether instructions are sufficient to change the dependence structure".
- Multiple candidate samples on the same material, to distinguish "the material does not support it" from "generation tendency".

## Conclusion

> $\Pr(V=1)$ is high, but $\Pr(B=1\mid V=1)$ is small.

In the generation examined (a single generator, a single dataset, a single run), most questions are valid and answerable, but very few require both modalities; this is of the same order of magnitude on the luna test reference batch.

-------------------------------------------------------------------------

# Technical Route: A Closed Loop Built from the Task's Own Criterion

The route is not "apply a self-improvement recipe"; it follows from what the target property is. This section states the
argument, the loop it induces, the law that decides whether the loop may run, and the evidence status of each link. The
mathematics is in [docs/necessity_formal_argument.md](docs/necessity_formal_argument.md) §11 and §10.7.

## Why this technique fits this task

"Both modalities are necessary" is a **counterfactual**, not a surface property: remove one side and ask whether the
answer is still determined. Every term of the definition is a statement about what a subject can determine under an
intervention. Three consequences:

1. **An observational scorer is a proxy for an interventional target.** A judge reading the question, a reward model
   trained on labels, and a difficulty statistic all score the *observed* pair; none removes a side. The only instrument
   that measures the definition without approximation is one that performs the removal. The answerability gate is not a
   heuristic correlated with necessity; it is the definition made executable, and criterion and evaluator are the same
   object.
2. **No reference answer is needed.** Necessity is a difference of determinability across conditions, so the criterion
   reads only the pattern $(TI,T,I,0)$ and never compares against a gold answer. This is why the loop can close locally,
   and it is a structural consequence of (1), not a cost saving.
3. **Two modalities give a constant-size criterion.** Four conditions, so one instrument serves four jobs: measuring a
   corpus, diagnosing a generator, filtering candidates, certifying a round.

## The loop

```
material m ─► generator G ─► candidates Q
                              │
                              ▼
                answerer A under TI / T / I / 0
                              │
                              ▼
       criterion C_A: a single side answered? → reject
                              │
                              ▼
              kept set ─► fine-tune G' ─► next round
```

Because $C_A$ is built from $A$, the loop needs no external auditor to produce a label — and for the same reason it can
drift without one noticing. The drift has two names: $\ell=P(\text{side answers}\mid\text{dual})$ (a dual is rejected
because the answerer guessed) and $1-s=P(\text{side silent}\mid\text{redundant})$ (a redundant question is kept because
the answerer could not use the side). What makes the loop auditable is that **the same device measures its own
reliability**: the blind condition estimates the prior part of $\ell$, the single-side conditions estimate $s$ on a
labelled pilot.

## Where the loop may run: the law and the stop rule

With dual prevalence $\pi$, the gate has recall $1-\ell$ and precision
$\pi(1-\ell)/[\pi(1-\ell)+(1-\pi)(1-s)]$; it beats the base rate **iff $s>\ell$**. As a loop, $s>\ell$ says the filter
keeps duals preferentially; if $s\le\ell$ each round preferentially keeps single-modality questions and trains the model
to produce more of them, with the drift invisible from the model's confidence. So the inequality is both the deployment
test on a new corpus and the stop rule for the iteration.

Measured: all four side rules satisfy it (MMQA text 0.930 vs 0.333; MMQA image 0.211 vs 0.095; ScienceQA text 0.409 vs
0.292; ScienceQA image 0.617 vs 0.250), the two-sided independence prediction matches the observed dual rejection
(0.397 vs 0.429 MMQA; 0.531 vs 0.583 ScienceQA), and the recall ceiling holds exactly. ScienceQA's weakness is one
number, $s_T=0.409$: the answerer cannot read the lecture enough to notice it already answers the question.

## Where the loop can be amplified

Because $C_A$ is built from $A$, raising $A$ raises certified supply: the verifier's recall is capped by full-material
accuracy, and the per-item repair raised that accuracy 0.517 → 1.000 and certified recall 0.483 → 0.983 with the
criterion untouched. The amplifier is on the **answering** side.

The generating side offers no comparable lever on this stack. E4 measured that the copied span is **load-bearing**:
ablating the attributed heads leaves the question byte-identical (the copy is redundant across heads), and blocking
attention to the source span removes the copy but makes the question confabulate a threshold or lose its criterion.
A generation-side intervention is inadmissible not because it is hard but because the copy is what makes the question
well posed. Supply is further bounded by mode collapse: best-of-8 produced no new dual shape. Hence the gradient is
asymmetric — answering-procedure changes move $s$ without touching the criterion, generation-side changes require new
data or a diversity term.

## The four layers actually used

| layer | instrument | what it fixes | evidence |
|---|---|---|---|
| measurement | three-valued identifiability label (valid, redundant, direction; undetermined kept) | the yardstick: a question is judged on $(q,m)$, not on $q$ | 1,257 materials, verified dual lower bound 19 (1.5%) |
| diagnosis | E1 leak/anchor decomposition + per-side answer rates | which side is redundant, and whether the redundancy is a copy or an anchor | leak 100% synthetic vs 4–13% real; MMQA text-anchored, ScienceQA image-anchored |
| selection | local answerability gate | reject what one side already answers; portable two-sided rule when the direction is unknown | MMQA 0.571/0.800, ScienceQA 0.458/0.367, 0 API calls |
| certification | repaired local three-condition verifier | confirm a survivor needs both sides, and which repair tier is enough | per-item 0.983 = API PVS 0.983 |

Generation-side remains framed generation: the schema is the working lever (refer prompt reaches dual 1.000 on the
synthetic set), and the circuit route is closed with mechanism, not with a failed attempt.

## Non-claims

- The gate is not a universal replacement for the judge: on ScienceQA the judge is significantly better on recall, and
  the union rule is the high-recall fallback.
- The generation-side repair is untested beyond inference-time intervention; no fine-tuning round was attempted.
- All labels are AI consensus; the human review packs are still owed.
- The schema effect is nested in skill and is not a causal lever.

-------------------------------------------------------------------------

# Paper Structure and Claims (Draft)

## Structure

| Section | Content |
|---|---|
| 1 Introduction | Task and goal → distinction between relevance and necessity → observation → gap → contributions |
| 2 Related Work | Multimodal question-answer pair generation, question-generation self-play, measures of modality dependence (organized by "how to judge which modalities are needed") |
| 3 Task and Measure | Task setting, answer identifiability, reading of PID, evidence types |
| 4 Diagnosis of the Status Quo | E1–E5 |
| 5 Method | Why the technique is the definition made executable; the loop; the law $s>\ell$ as licensing and stop rule; the gate, the repaired verifier, the closed circuit route |
| 6 Experiments | E1 source decomposition; E2 grid and interval; E3 repair ladder; selector tables against API PVS on two corpora; E4 attribution/ablation/blocking; cost |
| 7 Discussion and Limitations | Single dataset, AI review, evaluator false passes and missed judgments |

## Claims and Evidence Status

| ID | Claim | Evidence | Status |
|---|---|---|---|
| C1 | In the generation examined, about nine in ten valid questions have one redundant modality, and about 2% are dual-necessary | E1–E3; two generators; 40/40 spot check | Supported (scope: one dataset, a single run, AI review) |
| C2 | The evidential strength of responder signals is asymmetric: a match that agrees with an independent reference and whose derivation can be checked is strong evidence, while failure is limited evidence | Proposition 15 (Fano) | Supported (theory); experimental test on this task not done |
| C3 | Prior behavioral signals (instructions, single-modality answerability, uncertainty, accuracy differences) cannot certify modality necessity | Sections I–III of Prior Work; the derivation in Section IV | Partially supported (approaches read; consequences on this task await experiments) |
| C4 | The questioner's attractor lies in single modalities | Records in SPELL and RGX; 93% single-modality sufficiency | Partially supported (this baseline is not self-play) |
| C5 | A local answerer can select dual-necessary questions without references, and its usefulness is decided by $s>\ell$ | MMQA 0.571/0.800, ScienceQA 0.458/0.367; API PVS control; blind control; law checked on both corpora | Supported (labels AI consensus; 60 materials; two corpora) |
| C6 | The verifier's recall is capped by full-material accuracy and is repairable at the answering interface | per-item repair 0.517 → 1.000, recall 0.483 → 0.983, = API PVS | Supported (pool B is template-designed) |
| C7 | Attention-level intervention on the copied span cannot manufacture dual questions | E4: head ablation byte-identical; span blocking confabulates the threshold, fidelity 0.000 | Supported (one generator, one material family) |
| C8 | One self-play round with this filter increases the dual-necessary supply of the generator | None | **Not tested** (needs a training round, `peft`, ~tens of GPU·h) |
