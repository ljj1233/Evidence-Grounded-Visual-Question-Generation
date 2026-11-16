# Reference index

This document collects the external literature cited by the project, the data sources, and the project's own entry point for mathematical definitions. Each reference is annotated with a reading depth, to help judge how far it can be cited; "full text read" means the method and key sections were read, "abstract" means only the abstract page was seen, "search snippet" means only a fragment of search results was seen.

## 1. Mathematical definitions and argumentation (project-owned)

All in [necessity_formal_argument.md](necessity_formal_argument.md), not dependent on external literature.

| Content | Location |
|---|---|
| Notation, validity $V$, consistent completions $C_T,C_I$, assumption A1 | Section 1 |
| Modality necessity $D_T,D_I$ (Definition 1), dichotomy (Lemma 1), double necessity $B$ (Definition 2) | Section 2 |
| Sufficient solution (Definition 3, Proposition 1), witness (Proposition 2), failure does not propagate (Proposition 3) | Section 3 |
| Answerer behavior is not evidence of necessity (Proposition 4) | Section 4 |
| Absence is not necessity, invariance (Proposition 5), facts supplied by the stem cannot serve as witness variables (Proposition 6) | Section 5 |
| Minimal answer points and integration (Propositions 7, 8) | Section 6 |
| Three-valued summary, asymmetry of the two kinds of verification error (Proposition 10), conjunction of multiple verifiers (Proposition 11), bounds under a fixed denominator (Proposition 12) | Section 7 |
| What the argument does not resolve | Section 8 |
| Three operational conventions of the evaluation implementation: presupposed closure, stem slots defining answer equivalence, two-group agreement with two-out-of-three; incidental invariance boundary; the two reporting categories | Section 9 |
| Information-theoretic formulation: residual uncertainty (Proposition 13), per-question and question-level PID (Proposition 14), evidential strength of answerer signals (Proposition 15) | Section 10 |

Evaluation definitions and implementation conventions follow [necessity_formal_argument.md](necessity_formal_argument.md) and [VQG_Proposal.md](../VQG_Proposal.md); the earlier `evaluation_protocol_v1.0.md` has been deleted together with the old experimental framework code.

## 2. Papers

### 2.1 Shortcut learning and visual dependence

| Literature | Link | Cited in | Usage | Reading depth |
|---|---|---|---|---|
| Shortcut learning | <https://arxiv.org/abs/2004.07780> | Early narrative | The general problem of shortcut learning in machine learning | Carried over from the original text, not re-verified |
| VQA v2 | <https://arxiv.org/abs/1612.00837> | Early narrative | Language priors can mask insufficient visual ability; complementary images are used to check the dependence of answers on visual content | Carried over from the original text, not re-verified |
| Seeing or Knowing? Visual Context Sensitivity in Multimodal Large Language Models | <https://arxiv.org/abs/2607.26326> | VQG_Proposal.md section 7 | Distinguishes encoding of visual information from its reliable utilization | Carried over from the original text, not re-verified |
| VideoQA visual dependence audit | <https://arxiv.org/abs/2606.30220> | Early narrative | Compares performance with and without visual input | Carried over from the original text, not re-verified |

### 2.2 Generation evaluation

| Literature | Link | Cited in | Usage | Reading depth |
|---|---|---|---|---|
| Research on practices for evaluating generated text | <https://arxiv.org/abs/2202.06935> | Early narrative | The mismatch between surface quality, proxy metrics and the actual goal | Carried over from the original text, not re-verified |
| Evaluation of ConVQG | <https://arxiv.org/html/2402.12846v1> | Early narrative | QG evaluation commonly uses automatic text metrics together with human preference comparison | Carried over from the original text, not re-verified |
| QGEval | arXiv 2406.05707 | To be used | Evaluates generated questions along seven dimensions, including answerability | Search snippet |

### 2.3 Visual question generation (VQG)

| Literature | Link | Difference from this study | Reading depth |
|---|---|---|---|
| Early VQG | <https://aclanthology.org/P16-1170/> | Studies natural question asking | Carried over from the original text, not re-verified |
| IM-VQG | <https://arxiv.org/abs/1903.11207> | Maximizes the mutual information among image, expected answer and question; this study does not pre-specify the answer | Carried over from the original text, not re-verified |
| ConVQG | <https://arxiv.org/abs/2402.12846> | Dual contrastive objective; this study grounds joint utilization in the information requirement of the answer | Carried over from the original text, not re-verified |
| SynMQG | <https://shuangyinli.cn/publications.html> | Decoupling and mutual information optimization; this study organizes evaluation around answering necessity | Carried over from the original text, not re-verified |

### 2.4 Multimodal QA pair generation (direct competitors)

| Literature | Link | What was read | Reading depth |
|---|---|---|---|
| SMMQG (Wu et al., Findings EMNLP 2024) | <https://aclanthology.org/2024.findings-emnlp.759/>; arXiv 2407.02233 | Necessity is requested by the generation prompt; the main text claims the verification requires "every source", while the verification prompt published in Appendix D has only two items, "the answer can be derived" and "the style conforms"; no pass rate for that verification is given; the human study measures source relevance rather than necessity; no public code found in search | Full text and appendix read |
| FM2DS (Abaskohi et al., Findings EMNLP 2025) | <https://aclanthology.org/2025.findings-emnlp.383/>; code github.com/ServiceNow/FM2DS | Paper: discard if a single modality can answer; public code: judges "not multimodal" as soon as the returned string is non-empty, uses gpt-4o-mini, images enter the prompt as text placeholders; the reported rejection rate of 11.58% combines the multi-hop and multimodal checks; the paper itself states that in the zero-shot setting the model bypasses verification | Full text, appendix and public code read |

### 2.5 Question-generation self-play using answerer behavior as a signal

| Literature | Link | What was read | Reading depth |
|---|---|---|---|
| VisPlay (CVPR 2026) | <https://arxiv.org/abs/2511.15661> | Question reward is the uncertainty of the answerer's majority-vote confidence minus repetition; pseudo-labels are the answerer's own majority vote; no "requires image" check, quality is judged only by downstream accuracy | Full text read |
| V-Zero | <https://arxiv.org/abs/2601.10094> | Four-way multiple choice; reward is whether the intuitive answer agrees with the majority vote; no modality dependency check | Full text read |
| SPELL (ICLR 2026) | <https://arxiv.org/abs/2509.23863> | Three roles: questioner, answerer, verifier; Appendix F.6 records the questioner converging to trivial questions and the verifier's self-deception | Abstract and Appendix F.6 read |
| RGX (Luo et al.) | <https://arxiv.org/abs/2103.07449> | Grouped by answering loss, using only low and medium loss; "rewarding only correct answers converges to a trivial local optimum" | Method excerpt read |
| Self-Questioning Language Models | <https://arxiv.org/abs/2508.03682> | A single model's question-asking and question-solving loop without external data (text only) | Search snippet |

### 2.6 Measuring and diagnosing modality dependence

| Literature | Link | What was read | Reading depth |
|---|---|---|---|
| Park et al., modality bias in video QA | <https://arxiv.org/abs/2408.12763> | MIS = the accuracy difference between with and without a modality; complementary questions account for 2.1% of TVQA, 2.4% of LifeQA, 0.6% of AVQA; humans did not find complementary questions; human–machine kappa about 0.3; only one MLLM used; the arXiv page does not list a venue | Full text read |
| Multi-modal Data Spectrum (Madaan et al.) | <https://arxiv.org/abs/2509.23499> | Modality shuffling plus three-model majority voting; covers 23 benchmarks; on ScienceQA, question text alone is 17.5% above random | Method and results read |
| M3IRT (Uebayashi et al., ICLR 2026) | <https://arxiv.org/abs/2603.02663> | Decomposes ability and difficulty into three components—image, text, cross-modal—fitted from the response records of 24 VLMs; multiple choice only; lists "question generation that controls cross-modal difficulty" as a future direction | Abstract and conclusion read |
| Liang et al., information decomposition of multimodal interaction (NeurIPS 2023) | <https://arxiv.org/abs/2302.12247> | Uses PID to characterize the redundancy, uniqueness and synergy of a dataset, and proposes a scalable estimator | Abstract read |

### 2.7 Information-theoretic tools

| Literature | Link | Usage | Reading depth |
|---|---|---|---|
| Williams & Beer 2010, Nonnegative Decomposition of Multivariate Information | <https://arxiv.org/abs/1004.2515> | Partial information decomposition: decomposes the information that several sources carry about a target into redundant, unique and synergistic atoms | Abstract read; its $I_{\min}$ redundancy measure has drawn later criticism |
| Bertschinger et al. 2014, Quantifying unique information (Entropy 16(4)) | <https://arxiv.org/abs/1311.2852> | Another measure of unique, shared and synergistic information based on marginal distributions | Abstract read |
| Fano's inequality | Standard information-theoretic result | Relates the error probability to the conditional entropy (mathematical reference Proposition 15) | Textbook result |

### 2.8 Full texts not yet read

CaptionQA (blind-answering screening), VeriSciQA, MedVLSynther, MuMuQA, CRIT, See First, Answer Later (blind reader as reward). So far only search snippets or fragments have been seen; they are not used as prior conclusions.

## 3. Data sources

| Data | Link | Cited in |
|---|---|---|
| ScienceQA official data description | <https://scienceqa.github.io/> | VQG_Proposal.md section 4 |
| ScienceQA official repository (download and fields) | <https://github.com/lupantech/ScienceQA#-download-the-dataset> | README.md, VQG_Proposal.md section 4 |
| MultiModalQA (AllenAI) | <https://github.com/allenai/multimodalqa> | README.md |
| WebQA (WebQnA) | <https://github.com/WebQnA/WebQA> | README.md |

## 4. Notes

- Entries marked "carried over from the original text, not re-verified" come from earlier documents; their titles and links were not checked one by one online, and titles, authors and versions should be verified before submission; the reading depth of the other entries is as above.
- When changing a citation, please also update the relevant places in [story.md](../story.md) and [VQG_Proposal.md](../VQG_Proposal.md).
