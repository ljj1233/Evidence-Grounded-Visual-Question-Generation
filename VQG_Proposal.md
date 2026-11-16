# Research Proposal: Modality Necessity in Image-Text Question-Answer Pair Generation

> **Protocol version note (2026-11-13):** The numbers and conclusions in this document come from the old protocol (generation prompt `plain-v1`, Qwen3-VL-8B, transformers 4.57.6). Results for the new protocol `plain-v2` (fixed neutral English prompt, transformers 5.22.2) are written in `outputs/<run_id>/probe_report.md`; after final integration a "Probe" section will be added to this document without changing the existing definitions and numbers.

 
## Abstract

The goal of image-text question generation is to have text and image jointly participate in solving. A question being related to both text and image, and being answerable under the complete input, does not mean that both modalities are indispensable: a question whose answer can be determined from only one modality cannot examine joint reasoning.

This study focuses on **answer-level modality necessity**: after removing the text or removing the image, is the question's answer still determined? The study has three parts:

1. **Measurement.** Using answer identifiability as the criterion, give a checkable certification for each question, and report it separately from the four-condition answering behavior (BDR) of fixed questions.
2. **Diagnosis.** Evaluate baseline-generated questions on 1,257 material groups from ScienceQA: among valid questions, about 93% need only one modality, and verified dual-necessary questions account for only about 1.5%.
3. **Method.** A closed loop built from the task's own criterion: a local answerability gate selects candidates (reject when one side alone answers), a repaired local three-condition verifier certifies survivors, and the loop is licensed only where $s>\ell$, with $s$ and $\ell$ measured on a labelled pilot. Generation-side circuit intervention is measured to be inadmissible because the copied span is load-bearing. Details and evidence status: "Technical Route" in [story.md](story.md) and [docs/necessity_formal_argument.md](docs/necessity_formal_argument.md) §11.

## 1. Task Definition and Scope

### 1.1 Inputs and Outputs

Given text $T$ and a companion image $I$, generate a question:

$$Q=G(T,I),$$

An answer $a$ may also be generated at the same time. The judgment depends only on the answer to $Q$, so the study first examines $Q$; when $a$ is also generated, it is additionally required that $a$ be equivalent to an independently established reference answer.

- **Text T:** background explanation, rules, conditions, or other text that can be used independently.
- **Image I:** a diagram, structural figure, chart, or example image accompanying the text.
- **Output:** one question; the generator does not receive a pre-specified answer and does not obtain evaluation labels.

All generation methods receive the same $T,I$. ScienceQA is the current empirical vehicle; the question definition depends only on "whether the text, image, and answer are determined", and is not tied to subject, image type, or dataset fields.

### 1.2 Properties of an Ideal Question

1. **Valid ($V$):** the intent is clear, references are clear, presuppositions hold, and the material supports a unique complete answer.
2. **Text-necessary ($D_T$) and image-necessary ($D_I$):** after removing the corresponding modality, the answer is no longer determined. When all three hold simultaneously, it is recorded as dual-necessary $B$.
3. **No stitching, no leakage:** the two kinds of information jointly serve the same reasoning goal; the question stem does not restate key information from the other modality, nor juxtapose two independent sub-questions.

"Necessary" means that after removing that modality the answer is no longer determined; "redundant" means that after removing it the answer is still determined. Reasonable overlap between the two modalities is allowed, as long as each still has information that the other cannot replace.

### 1.3 Answers and Evidence Are Part of the Evaluation Object

After generation, the evaluation process independently establishes a reference answer, scoring points, and acceptable formulations for the current question, and binds them to the question. Answer establishment is separate from generation: an independent model provides candidates, which are then confirmed via rules, programmatic derivation, or review. The original ScienceQA answer is reused only when the new question is semantically consistent with the original question's answer.

## 2. Research Questions

| Research question | Corresponding design | Main evidence |
|---|---|---|
| RQ1: Among baseline-generated questions, how many genuinely require both modalities? | One generation over all materials, evaluated per question | Dual-necessary proportion, direction of redundancy, valid rate |
| RQ2: Can behavioral signals (correct or incorrect under a single modality, uncertainty) certify modality necessity? | Compare identifiability certification with four-condition answering | Agreement and divergence of the two types of evidence, spot-check error |
| RQ3: What kind of question-posing or training signal can raise the dual-necessary proportion without exceeding the material evidence? | Candidate methods compared under the same base model and the same materials | Dual-necessary proportion, valid rate, cost |
| RQ4: What structure do dual-necessary questions have? | Classify verified questions | Distribution of types and material templates |

RQ1, RQ2, and RQ4 already have preliminary results (Section 7); RQ3 belongs to the method stage and has not yet been experimented on.

## 3. Metrics

### 3.1 Answer Identifiability

Fix a question $q$ and common knowledge $K$. After removing the text, the retained input is $(q,i,K)$; if all text completions that are "compatible with $q$, the retained material, and $K$, and still uniquely answerable" yield the same answer, then $D_T=0$ (text redundant), otherwise $D_T=1$. $D_I$ is defined symmetrically. There are two types of evidence:

- **Sufficient solution:** if a solution is given that satisfies well-grounded premises, valid derivation, and a complete conclusion, then the corresponding direction is redundant.
- **Witness pair:** if two legal completions are given that make the answers different, then the corresponding direction is necessary.

Failing to find a sufficient solution does not imply necessity, and a rejected witness does not imply redundancy; the evaluator outputs 0 or 1 only when it has accepted evidence, and otherwise remains undecided. Definitions, propositions, and proofs are in the mathematical reference; its information-theoretic formulation (residual uncertainty and partial information decomposition) is in Section 10 of that document.

### 3.2 Evaluation Unit and Fixed Denominator

Each method generates one question per material group. $N$ is the pre-arranged total number of attempts; all outputs, invalid questions, generation failures, and technical failures remain in the denominator. The number of technical retries and the budget are fixed before the run; model refusals are recorded separately.

### 3.3 Reporting: Two Categories

Reporting uses only two categories: **verified dual-necessary** and **the rest**. "The rest" does not mean that all are single-modality questions; it includes questions where a single modality already suffices to determine the answer, questions that are themselves invalid, and questions the evaluator has not yet adjudicated. The report lists these separately; undecided only means that the evaluator has not adjudicated, not a category of result.

Under fixed $N$, let $n_1$ be confirmed dual-necessary and $n_\bot$ undecided; the true count $m$ satisfies

$$
n_1\le m\le n_1+n_\bot .
$$

This interval expresses the effect of undecided cases and is not a sampling confidence interval; when false admissions exist it also no longer guarantees coverage of $m$. An evaluator confirmation counts only after human or delegated review.

### 3.4 Validity and Faithfulness

Validity $V\in\{1,0,?\}$ denotes confirmed valid, confirmed invalid, and undecided, respectively. For the question as generated, three things are checked: whether the presuppositions are supported by the input; whether the actual text and image together with common knowledge suffice to support a complete answer; and whether the requested object and comparison scope are clear. A disproven false presupposition, missing conditions that determine the answer, or an unclear scope of what is asked are defects of the generated question, recorded as $V=0$ and retained in the denominator; reviewer technical failures, illegible materials, or unresolved evidence disagreement remain undecided.

Faithfulness and topical relevance are recorded separately: a question may talk about objects in the input yet demand a conclusion that the input cannot determine. Reference answers are also verified separately; disagreement among model answers alone does not prove that a question is invalid. Invalid questions are not assigned modality necessity.

### 3.5 Behavioral Proxy: Four-Condition Answering of Fixed Questions

To compare with behavioral signals, four-condition answering of fixed questions is retained:

| Condition | Content visible to the answerer |
|---|---|
| TI | $Q+T+I$ |
| T | $Q+T$ |
| I | $Q+I$ |
| 0 | Only $Q$ |

The four conditions use the same answering model and answering rules; when material is removed, its caption, OCR, evidence summary, and other derived information are removed together, and different conditions do not share history. The per-question dual-modality dependence rate is defined as

$$
B_i^{\mathrm{BDR}}=\mathbf 1[V_i=1,\ C_i^{TI}=1,\ C_i^T=0,\ C_i^I=0],\qquad \widehat{\mathrm{BDR}}=\frac1N\sum_iB_i^{\mathrm{BDR}} .
$$

BDR is a behavioral proxy under a specific answerer and scoring protocol and **does not replace** $D_T,D_I$: a wrong single-modality answer may come from insufficient capability, a correct full image-text answer may come from guessing, and refusals are recorded as undecided in the protocol, so a genuinely dual-necessary question may appear as "undecided". The no-material answering accuracy is listed separately as an auxiliary reading to locate common-sense shortcuts and does not enter the BDR formula.

### 3.6 Correctness of Answers

Based on the verified reference answer and the actual requirements of the question, a three-valued judgment:

| Judgment | Criterion |
|---|---|
| Correct | Covers all necessary points, semantic paraphrase and equivalent units allowed, no substantive error or contradiction |
| Wrong or incomplete | Key result, unit, relation, or direction wrong, missing points explicitly required, or accompanying reasoning substantively contradicts the material |
| Undecided | Reference answer, scoring scope, returned content, or technical state insufficient to support a reliable judgment; recorded as `correct=null` |

There is no partial credit; refusals under missing-material conditions follow undecided. Answer correctness and question necessity are viewed separately; the performance of the actual solver does not replace the information-level judgment.

### 3.7 Answer Equivalence and Evidence Acceptance

- **Presupposition closure:** a legal completion must satisfy the presuppositions entailed by the question (including implicit presuppositions such as "the exchange or option asked about is indeed feasible").
- **Answer equivalence:** determined by a "slot pattern" generated only from the question, each slot containing exactly one value; equivalence holds only when all slots are identical after normalization, and missing values or inconsistent extraction are recorded as undecided. Granularity is determined by what is asked, without using domain lexicons.
- **Evidence acceptance:** two model groups each propose and recompute evidence, and it is accepted only when the two groups agree; when they disagree or one is undecided, a third model group arbitrates, best of three, and if there is still no majority it is recorded as undecided.

The basis and boundaries of these three (e.g., incidental invariance) are in Section 9 of the mathematical reference.

### 3.8 Review Source and Spot Checks

Review is performed by AI delegated by the researcher, and the source is truthfully recorded as `user_delegated_ai`, not written as human annotation, and human labels are not automatically filled in; the automatic, human, and final adjudication label sets are stored separately and are not substituted for one another. This review can be used for process freezing and result organization, but it is not an independent human gold standard, nor a human agreement validation.

For the formal experiment, about 10% of questions are spot-checked, stratified by method, with annotators blind to method names; reference-answer errors, validity misjudgments, false positives and false negatives of dual necessity, and undecided records are counted separately. Failure records are classified by the stage in which they occur: evaluation record errors (fixed if repairable, retained as undecided if unverifiable), question defects (counted as invalid), technical faults (limited retries under the pre-set policy), candidate filtering and refusals (all candidates and costs recorded). All pre-arranged attempts remain in the fixed denominator.

## 4. Data

### 4.1 ScienceQA Material Subset

The first round uses image-text materials from ScienceQA. Official statistics are 21,208 questions, of which 10,332 contain images and 6,532 contain both image and text context ([official data description](https://scienceqa.github.io/)). Text source policy:

- **`hint`:** the context used preferentially by default.
- **`lecture`:** if used as background text, marked separately as an extended-input setting, and repetition of visual facts is reviewed.
- **`solution`:** not used as default input, retained in the raw data.
- **Original question:** only independently usable background explanations are extracted; importing the full original question belongs to the existing-question revision setting and is recorded separately from autonomous question posing.

Field basis is in the [ScienceQA official repository](https://github.com/lupantech/ScienceQA#-download-the-dataset). The current baseline uses the 1,257 material groups in the official test split that contain a hint and an image, without filtering by joint solution space or output quality.

### 4.2 Material Admission Rules (Used in Method Comparison Experiments)

The material admission check looks for a natural joint solution space: text and image each contain at least one relevant piece of unique information, and together they can support a verifiable conclusion; text and image may partially overlap, but the key is that the required information is irreplaceable.

| Check item | Admission criterion | Exclusion or pending cases |
|---|---|---|
| Source and readability | Source, split, text field, and image hash are traceable; text and image are complete and legible | Image missing or corrupted, key axes, legends, or units unreadable |
| Text has substantive role | Background, rules, constraints, or relations necessary for the joint goal can be located | Only "see figure", object names, or duplicated captions |
| Image has unique information | Readings, states, structures, or spatial relations not fully restated in the text can be located | Decorative images, key visual facts all written into the text |
| Joint solution goal | Text rules or background together with visual facts derive a conclusion | Only independent questions can be stitched together, or extra rules must be fabricated |
| No answer leakage in input | Generation input contains only the specified text and image; text source policy is consistent | Mixing in the original answer, solution, candidate labels, or evaluation evidence |
| Split and source isolation | Duplication of images, sources, and templates is recorded; debug materials are isolated from formal tests | Shared sources used across rule debugging and formal evaluation |

Materials passing all checks enter the pilot; explicit failures retain their exclusion reason; unknown items remain pending. Material screening is completed before method runs, and the rules are identical for all methods; the pilot has about 30 groups, and the formal experiment targets about 300 independent material groups.

### 4.3 Illustrative Example (Synthetic)

- **Text:** This experiment lists temperatures whose activity is no lower than 75% of the highest measurement in this run as candidate operating temperatures, then selects the lowest temperature among the candidates.
- **Image content:** In the bar chart, 20, 30, 40, and 50°C correspond to activities 22, 47, 61, and 29.
- **Question:** According to the screening rule of this experiment, which test temperature should be chosen for the run?
- **Answer:** 30°C. The threshold is 61×75%=45.75; 30°C and 40°C qualify, and the lower one is chosen.

The text gives a decision rule specific to this experiment, and the image gives unique measurement data; together they determine the answer, and removing either modality makes the answer unobtainable. The 75% is a self-authored rule, and the image content is an illustrative description; this example is a synthetic illustration and is not counted in real-data results.

### 4.4 Sampled Diagnosis of Material Yield

From the training split `hint+image` candidates, 30 groups were sampled by simple random sampling without replacement (seed 20261111). Each group underwent one joint-question witness search and one challenge review: the model's two-round results were 15 passed, 6 failed, 9 undecided; a subsequent AI per-image review tentatively gave 5 passed, 18 failed, 7 pending, and both sets of records are retained. Scaling by 5/30, the point estimate for training candidates is about 657 groups, and the Wilson sampling interval scales to about 289–1,324 groups; if all pending cases pass, the point estimate is about 1,578 groups. This interval describes sampling fluctuation and does not correct automatic misjudgments; val/test have not yet been sampled. Records are in `outputs/scienceqa-screen-30/`.

## 5. Methods and Comparison

**The route below is the one the evidence supports; the earlier candidates are kept as comparison objects.** The method is a loop whose filter is the definition made executable: the answerer sees the full material and each single side separately (plus a blind condition), and a candidate is rejected when one side alone answers. The filter's usefulness on a corpus is decidable from two measured rates ($s>\ell$), and its recall is capped by the answerer's composition accuracy, which is repairable at the answering interface (per-item comparison: 0.517 → 1.000 accuracy, verifier recall 0.483 → 0.983). Generation-side circuit intervention was tested and is inadmissible (the copy is load-bearing). Details, numbers and non-claims are in "Technical Route" in [story.md](story.md).

### 5.1 Early Vision: Three-Stage Evidence Anchoring

The three-stage method operationalizes necessity as an explicit question construction process:

1. **Evidence extraction:** extract rules, conditions, and relations from the text, and readings, states, structures, or spatial relations from the image, saving source locations.
2. **Cross-modal evidence pairing:** select text and image evidence that jointly support one solution goal, and check their unique roles, redundancy, common-sense substitution, and reasoning connection; pairing targets solution relations such as "how a text rule acts on a visual fact", not topical similarity.
3. **Evidence-anchored question generation:** generate one question around the selected evidence and its solution relation; the stem retains necessary references and avoids directly writing out the required visual reading or replacing integration with two independent sub-questions.

Internal evidence, the final question, and failure trajectories are traceable; the external output is only the question.

### 5.2 Minimal Comparison Set

| Condition | Design | Question answered |
|---|---|---|
| Plain zero-shot | Pose questions directly from T, I | What modality dependence do default-generated questions have |
| Strong prompt | Explicitly require that missing either modality prevents a complete answer | How much necessary dependence can be achieved by merely stating the goal |
| Full evidence anchoring | Execute the three stages | What gains and costs explicit construction brings |
| Remove image anchoring | Keep the original T, I and remove the explicit image-evidence anchoring step | What contribution the image-anchoring component makes |

The pilot first runs the first three conditions, and the formal experiment adds the core ablation. All methods share the base model, materials, final question count, and technical retry policy; call counts, tokens, time, and candidate counts at each stage are all reported. When the self-play method enters the comparison, it is likewise added under the same base model, the same materials, and the same budget.

### 5.3 Decoding and Statistics

The main local-model experiment uses the greedy setting `do_sample=False, num_beams=1`, recording model version, seed, prompt, preprocessing, and environment. The API uses the most deterministic configuration it supports, with actual parameters recorded. Comparisons are paired on the same materials, bootstrap resamples by material or shared-source cluster, all methods use the same index, and metric differences with intervals are reported. The formal scale targets about 300 independent material groups initially; sample size and expansion policy are determined from the pilot before the formal run.

### 5.4 Checklist-Based Bounded Question Revision (Transferable Prototype)

Bounded revision applies the necessity criterion to improve existing questions: diagnose the gap, make one targeted rewrite, re-check under the same criterion, and verify the answer separately. Materials are fixed, the rewriter outputs only the question, and the reference answer and scoring points are not used as rewriting input. The checklist covers question intent and presuppositions, answer support, necessity in both directions, reasoning integration, stem leakage, the actual information dependence of the requested quantity, and the definition of quantities and units. Revision is organized around solution relations, e.g., changing from comparing temperature differences to computing the average cooling rate so that the duration in the text truly participates in the computation; merely adding phrasing such as "according to the text" does not constitute a necessity fix. When no question meeting the target can be obtained under the original materials, the unrepaired state and the denominator are retained.

Development run (30 training materials, records in `outputs/portable-refine-30/`): 7 questions survived automatically before rewriting, 14 in the end (46.7%), 20 triggered rewriting, and 7 rewrites succeeded (35.0% success among triggered items); in addition, 6 were undecided, 7 failed, and 3 had no compliant candidate. These are engineering metrics, reported separately from the dual-necessary proportion, and do not represent a human gold standard or a formal method improvement.

## 6. Current Results

| Content | Result | Source |
|---|---|---|
| Qwen3-VL-8B baseline, all 1,257 groups | Valid 962; of which single modality already suffices to determine the answer 896 (93%); verified dual-necessary lower bound 19 (1.5%); invalid 268, pending adjudication 27 | `outputs/qwen3vl8b-necessity-v2/probe_report.md` |
| luna baseline (test reference), all 1,257 groups | Answerable 1100, single-modality sufficient 976; verified dual-necessary about 23 (1.8%); original evaluator automatically tentative 64 | `outputs/necessity-v3-full-v1/probe_report.md` |
| Spot check of single-modality screening | 40/40 indeed have at least one redundant modality (Wilson 95% lower bound about 0.912); accordingly, redundancy accounts for at least about 84.8% of valid questions, and dual necessity for at most about 11.6% (relative to $N$) | Same as above, Section 6 |
| Direction of redundancy | Text redundant at least 630, image redundant at least 565 (evaluator screening judgment, directions not checked one by one) | Same as above |
| Structure of verified dual-necessary questions | Of the 19 questions, 14 are grid questions and 5 are yes/no exchange questions | `audit.jsonl` |
| Evaluator error | Of 28 automatically confirmed questions, 19 remained after review; incidental invariance 6, false admission 3; another 5 yes/no questions of the same type were not confirmed, judged by question type as likely missed but images not inspected | Same as above |
| Cost | Qwen generation and answering done locally; evaluation about 3,823 API calls, about 4.73 million tokens; about one third of the luna batch evaluation cost | Same as above |

The 12 controlled comparisons and 30 material screens during development served to expose evaluation blind spots and calibrate boundaries; label agreement rates do not represent formal results.

## 7. Discussion and Limitations

**Necessity and proxy metrics.** Dual-modality necessity is a property of the question's information requirements, while BDR is a behavioral proxy under a specific answerer and scoring protocol. A wrong single-modality answer can be affected by capability, a correct full image-text answer can be affected by guessing, and both modalities being separately necessary does not rule out stitching independent questions. Answering correctly with only $Q$ can come from common sense, memory, stem clues, or guessing, and is not used to infer the generator's internal knowledge source.

**Boundaries of the metric.** The evaluator's counterexamples and verification are done by models, which both falsely admit and miss cases; false admissions concentrate on presupposition binding, stem restatement, single-modality transcription, and incidental invariance (e.g., for a 2:2 grid the answer is 1/2 for either trait correspondence, judged as text redundancy by definition). The review precision for confirmed items is 19/28, and stratified sampling of "the rest" has not yet been done, so the dual-necessary proportion is a lower bound with an unknown number of missed cases.

**Status of evidence.** Review is AI review delegated by the researcher, not independent human annotation; current results come from one dataset, two generators, and a single run. The reuse of model roles and the fact that materials participated in debugging are retained in the records.

**Perception–utilization gap.** [*Seeing or Knowing?*](https://arxiv.org/abs/2607.26326) distinguishes the encoding of visual information from its reliable utilization. After the main pipeline is complete, about 100 groups can be selected for auxiliary analysis: for the same model, materials, and target visual facts, relate direct recognition performance to the role of evidence in generated questions. This analysis is behavioral evidence and cannot by itself identify internal causes in the model.

**Question dependence and generation response.** The evidence dependence of fixed questions (modify relevant facts, see whether the answer changes) and the response of regenerated questions (regenerate after modifying materials) are two different objects; counterfactual editing is limited to an auxiliary subset, recording naturalness and editing failures.

**Research extensions.** More datasets (the raw data of WebQA and MultiModalQA are already on the server, untested), more models, cognitive levels, diversity control, training and contrastive losses, and systematic counterfactual analysis are all future directions and are not prerequisites for the first-round contribution.

## 8. Execution Order and Deliverables

| Stage | Work | Delivery and completion criteria |
|---|---|---|
| 1: baseline profile | Two-category reporting under a fixed denominator, error sources, structural classification | Completed (Qwen and luna batches) |
| 2: evaluator stability | Stratified sampling of "the rest" to estimate the miss rate; repeated runs; more generators | Error description and stability data |
| 3: generalization test | Test the evaluator on non-educational data | Scope-of-applicability statement |
| 4: method | Local answerability gate + repaired local verifier, compared with API PVS and with the no-filter baseline under the same materials and budget | Dual-necessary proportion, precision/recall of selection, valid rate, cost, intervals, and the measured $s$ vs $\ell$ |
| 5: formal experiment | About 300 independent material groups, main comparison and ablation, paired bootstrap by material or source cluster | Complete raw results, intervals, spot checks, and cost |

Materials used in rule revision are moved to the diagnostic split, and confirmatory evaluation retains independent materials. Each stage is accepted by record completeness, evaluation inspectability, and comparison reproducibility; method effects are determined by actual results.

## 9. Record Specification and Related Work

**Record specification.** Each record retains protocol version, split, material and source ID, text field, image path and hash, method and base model version, prompt and decoding parameters, all candidates and the final question, failures and retries, calls and cost, reference answer and scoring points, annotation source, validity, necessity evidence and labels, and review trajectory. The run and recovery contract is in [docs/implementation_plan.md](docs/implementation_plan.md).

**Related work.** Detailed reading records are in [story.md](story.md) and [docs/references.md](docs/references.md); the distinction along the same coordinate is given here: how existing objectives improve multimodal question generation, and how this study further requires the answer to depend on information that each of the two modalities provides.

| Research direction or representative work | Existing objective and mechanism | Distinction of this study |
|---|---|---|
| [IM-VQG](https://openaccess.thecvf.com/content_CVPR_2019/html/Krishna_Information_Maximizing_Visual_Question_Generation_CVPR_2019_paper.html) | Maximize mutual information among the image, the expected answer, and the question | Does not pre-specify an answer; targets the necessity of the answer to both modalities |
| [ConVQG](https://arxiv.org/abs/2402.12846) | Dual contrastive objective distinguishes joint-modality and single-modality representations | Joint utilization is grounded in the answer's information requirements and certified per question |
| [SynMQG](https://shuangyinli.cn/publications.html) | Disentanglement and mutual-information-optimized collaborative multimodal question generation | Evaluation organized by answering necessity; representation collaboration and answer-required information examined separately |
| Multimodal QA-pair generation: SMMQG, FM2DS | Instruction requirements or single-modality answerability checks | Public materials report no precision of necessity checks; this study certifies with evidence |
| Question-posing self-play: [VisPlay](https://arxiv.org/abs/2511.15661), V-Zero | Reward based on the answerer's uncertainty | Difficulty is not equal to necessity; the reward cannot see whether a modality is redundant |
| Measures of modality dependence: Park et al., M3IRT | Accuracy differences, model-population answering records | Behavioral or population statistics; this study certifies per question |

**Expected contributions.**

1. **Problem and metric.** Distinguish "the question involves two modalities" from "both modalities are necessary", certify per question by answer identifiability, and provide an information-theoretic formulation and an analysis of evidential power.
2. **Diagnostic findings.** Quantify modality redundancy in generated questions on ScienceQA, and characterize the direction of redundancy and the structure of dual-necessary questions.
3. **Method.** Question-posing signals designed against redundant modalities and comparison experiments (not yet completed).
