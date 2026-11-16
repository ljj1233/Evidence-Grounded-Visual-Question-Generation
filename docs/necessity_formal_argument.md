# Dual-Modal Information Necessity: Definitions and Argument

Status: explanatory document, 2026-11-12, with Sections 9 and 10 added on 2026-11-13. This document writes the definitions on which the evaluator rests in mathematical form: the original evaluator (`open-answer-identifiability-dev-v2`, used by `simple-baseline-eval-*`) and the two-group-plus-arbitration evaluator implemented in the `necessity_ensemble` module, and explains item by item what each kind of evidence can establish, which assumptions it depends on, and what happens when verification errs. It modifies no protocol, metric, or historical result, and introduces no new metric. All examples in the text are taken from existing runs in this repository.

## 1. Notation

- $q$: the fixed question (including all options and presuppositions); $t\in\mathcal T$: text; $i\in\mathcal I$: image; $K$: the frozen common-knowledge convention.
- $\mathcal A$: the set of complete answers, and $\sim$ is the answer-equivalence relation (paraphrase, equivalent units). Throughout, "same/different" is meant in the sense of $\sim$.
- $\mathrm{Ans}_K(q,t,i)\subseteq\mathcal A/\!\sim$: the set of complete answers supported by $(q,t,i)$ under $K$.
- Validity: $V=1 \iff |\mathrm{Ans}_K(q,t,i)|=1$ and the presuppositions of $q$ hold; in that case the unique answer is denoted $a^\ast$.

**Compatible completion.** After removing the text, the retained input is $(q,i,K)$. Define

$$
C_T(q,i)=\{\,t'\in\mathcal T:\ t' \text{ is self-consistent and of the same kind as the original},\ (q,t',i) \text{ violates no constraint of } q,i,K,\ |\mathrm{Ans}_K(q,t',i)|=1\,\}.
$$

$C_I(q,t)$ is defined symmetrically. The three membership conditions correspond respectively to: the retained material is unchanged, the question and its referent are unchanged, and a unique complete answer is still possible after completion. Content that renders a modality blank, unreadable, contradictory, or informationally insufficient does not belong to $C$.

**Assumption A1.** When $V=1$, the actual material is a compatible completion: $t\in C_T(q,i)$, $i\in C_I(q,t)$.

## 2. Target properties

**Definition 1 (modality necessity).**

$$
D_T=1 \iff \exists\,t_1,t_2\in C_T(q,i):\ \mathrm{Ans}_K(q,t_1,i)\neq\mathrm{Ans}_K(q,t_2,i),
$$

$$
D_T=0 \iff \exists\,a\ \forall\,t'\in C_T(q,i):\ \mathrm{Ans}_K(q,t',i)=\{a\}.
$$

$D_I$ is defined symmetrically. $D_T=0$ means "the answer is identifiable under the retained input".

**Lemma 1 (dichotomy).** If $C_T\neq\varnothing$, then exactly one of $D_T=0$ and $D_T=1$ holds.

Proof: each element of $C_T$ corresponds to a unique answer, so the map $t'\mapsto \mathrm{Ans}_K(q,t',i)$ on $C_T$ is either constant or takes two distinct values. $\square$

**Lemma 2.** If $V=1$, A1 holds, and $D_T=0$, then the constant value in the definition satisfies $a=a^\ast$.

Proof: $t\in C_T$, so the claim follows by substitution. $\square$

**Definition 2 (dual-necessary).** $B = V\wedge D_T\wedge D_I$. Relevance, which premises a particular reference solution uses, and whether some answerer is right or wrong or refuses, do not appear in the definition.

Lemma 1 says that the property itself is one thing or the other; whether the evaluator **knows** which one is a separate matter, see Section 6.

## 3. What the two kinds of evidence each establish

**Definition 3 (sufficient solution).** A set of premises $P$ and a derivation $\pi$ satisfying:

- (H1) every premise in $P$ is supported by $q$, $i$, or $K$;
- (H2) the derivation is valid: the conclusion holds in every situation in which $P$ holds;
- (H3) the conclusion is the complete answer $a$ to the object asked about by $q$, not another object or a partial answer.

**Proposition 1 (sufficient solution $\Rightarrow D_T=0$).** If there exist $(P,\pi,a)$ satisfying H1–H3, then for every $t'\in C_T$, $\mathrm{Ans}_K(q,t',i)=\{a\}$.

Proof: take any $t'\in C_T$. By the membership conditions, $(q,t',i)$ violates no constraint of $q,i,K$, so $P$ holds in it (H1); by H2 and H3, $a\in\mathrm{Ans}_K(q,t',i)$; and since $|\mathrm{Ans}_K(q,t',i)|=1$, equality holds. $\square$

Proposition 1 involves no reference solution: a single legitimate solution suffices to refute necessity. Conversely, the fact that a reference solution uses some fact from $t$ does not establish $D_T=1$.

**Proposition 2 (witness $\Rightarrow D_T=1$).** If $t_1,t_2$ are given satisfying

- (W1) $t_1,t_2\in C_T(q,i)$;
- (W2) $\mathrm{Ans}_K(q,t_1,i)\neq\mathrm{Ans}_K(q,t_2,i)$;

then $D_T=1$.

Proof: this is Definition 1. $\square$

The correspondence between the five checks in a verification call and the above assumptions:

| Check | Sufficient solution | Witness |
|---|---|---|
| `question_preserved` | H3 | "does not violate $q$" in W1 |
| `retained_preserved` | H1 (premises do not conflict with the retained material) | "does not violate the retained material" in W1 |
| `well_posed` | H3 (complete, single-valued) | $|\mathrm{Ans}|=1$ in W1 |
| `premises_supported` | H1 | self-consistency and compatibility with $K$ in W1 |
| `conclusion_checked` | H2 | W2 |

Each of the three recorded false admissions breaks one of these: the hypothetical image in R08/I conflicts with the retained text "some magnets differ in shape" (W1); R14/T replaces "this passage" with a photograph (H3); the old witness in R16/T substitutes a brief summer for the ecosystem's overall precipitation ("does not violate $q$" in W1). If any one fails, the conclusion of Proposition 1 or Proposition 2 is no longer derived.

**Proposition 3 (failure does not propagate).**

- (a) failing to find a sufficient solution does not imply $D_T=1$;
- (b) a rejected witness does not imply $D_T=0$.

Proof: (a) the search covers only finitely many candidate derivations, and a solution satisfying H1–H3 may still exist outside them. (b) that one pair $(t_1,t_2)$ fails W1 or W2 does not exclude another pair from satisfying them. This is exactly the first-round probe E3/I: the first witness gave probability 1 for both square configurations (W2 fails) and was rejected, whereas the second witness (1/2 and 1) holds. $\square$

## 4. Why answerer behavior is not evidence

Let answerer $s$ map (question, visible material) to an answer or a refusal $\bot$.

**Proposition 4.** $D_T$ is not a function of the outputs of $s$ under the four visibility conditions: there exist instances with the same output pattern but different $D_T$, and all of the following combinations can occur.

| Output of $s$ without $T$ | $D_T$ | Instance in the repository |
|---|---|---|
| Correct | 1 | Solution problem asking for the total number of particles: $s$ defaults to one particle per ball and answers 9, whereas the text with two particles per ball gives 18 |
| Refusal | 0 | Symmetric 50% genetics problem: $s$ claims the dominance relation is missing, but both relations give 1/2 |
| Incorrect | 0 | A13: two bottles both containing 10 balls, $s$ miscounts |
| Refusal | 1 | Koi eye-color problem: the phenotype mapping is indeed missing |

Thus "the answerer did not answer" and "this modality contains indispensable information" are two different propositions. Answering under the four conditions can be used to look for candidate solutions or candidate missing facts, but its conclusions must still pass the hypothesis tests of Proposition 1 or Proposition 2. The historical BDR ($V\wedge C_{TI}\wedge\neg C_T\wedge\neg C_I$, refusal as undecided) is a behavioral proxy; its definition is unchanged, and it does not replace $D$.

## 5. Missing facts, invariance, and facts supplied by the question stem

**Proposition 5 (missing does not mean necessary).** Suppose the answer depends on the text only through some quantity $g(t')$, i.e. there exists $h$ such that $\mathrm{Ans}_K(q,t',i)=\{h(g(t'),i)\}$. If $h(\cdot,i)$ is constant on $g(C_T)$, then $D_T=0$, even if $g(t)$ cannot be obtained from $(q,i,K)$.

Proof: directly from Definition 1. $\square$

Three examples:

- Cancellation: when each ball represents $k>0$ particles and the solvent volumes are both $v$, $\dfrac{k\,n_A/v}{k\,n_B/v}=\dfrac{n_A}{n_B}$, independent of $k$.
- Symmetry: the squares are $\{Ii,Ii,ii,ii\}$, with mappings $\varphi\in\{I\mapsto\text{spotted},\ I\mapsto\text{unspotted}\}$. Under both $\varphi$ the number of squares showing spots is 2, so the probability is always $2/4$.
- Irrelevant condition: when the temperature drop $\Delta\theta=\theta_0-\theta_1$ is asked for, the duration $\tau$ does not appear in the requested expression; when the average cooling rate $\Delta\theta/\tau$ is asked for it does appear, and 5 minutes and 10 minutes give 0.6 and 0.3.

**Remark (the strength of the argument depends on the space examined).** Let the set of completions actually checked be $S\subseteq C_T$. Enumerating cases proves constancy only over $\forall t'\in S$; $D_T=0$ is obtained only when $S=C_T$ (exhaustive, with the reason stated) or when the argument holds algebraically for the parameters concerned. If $S\subsetneq C_T$, the conclusion is not derived and the case should be recorded as undecided. The scope of $C_T$ itself (for example, whether incomplete dominance is allowed) is a convention and must be stated with the record.

**Proposition 6 (facts supplied by the question stem cannot serve as witness variables).** If $q$ implies a fact $f$, then every $t'\in C_T$ is consistent with $f$, so $f$ is invariant on $C_T$, and no witness can be constructed by changing $f$.

Proof: members of $C_T$ must not violate $q$. $\square$

This is the formal meaning of "a question-stem leak renders an originally necessary modality unnecessary": in E1, changing "E corresponds to red eyes" yields two answers, red and black; after E2 writes this mapping into the question stem, the same change no longer belongs to $C_T$.

## 6. Minimal answer components and integration

Suppose the complete answer consists of minimal, independently decidable components $c_1,\dots,c_n$, with $\mathrm{Ans}=\prod_j \mathrm{Ans}^{(j)}$. For each component, Definition 1 gives $\mathrm{dep}_T(c_j),\mathrm{dep}_I(c_j)\in\{0,1\}$.

$$
D_M=\bigvee_j \mathrm{dep}_M(c_j),\qquad S=\bigvee_j\big(\mathrm{dep}_T(c_j)\wedge\mathrm{dep}_I(c_j)\big).
$$

**Proposition 7.** When $n=1$, $D_T\wedge D_I\Rightarrow S$. Proof: the single component depends on both. $\square$

**Proposition 8 (merging components turns concatenation into integration).** If two components are merged into $c=(c_1,c_2)$ with $\mathrm{dep}_T(c_1)=1,\mathrm{dep}_I(c_1)=0,\mathrm{dep}_T(c_2)=0,\mathrm{dep}_I(c_2)=1$, then computing with the merged component gives $\mathrm{dep}_T(c)=\mathrm{dep}_I(c)=1$, hence $S=1$; computing with the minimal components gives $S=0$.

Proof: the value of $c$ changes whenever either $c_1$ or $c_2$ changes. $\square$

So components must be minimal for integration to be meaningful. E7 ("which eye color is dominant, and what is the genotype in the upper-left square") is exactly a case with $D_T=D_I=1$ but $S=0$. "Dual-necessary" and "integrated" are two conclusions, and the latter is stronger.

## 7. Three-valued summary: conservativeness and error direction

The epistemic state of the evaluator is denoted $\hat D_M\in\{0,1,\bot\}$: it is 0 only when a sufficient solution has been accepted, 1 only when a witness has been accepted, and $\bot$ otherwise. $\hat B$ uses three-valued conjunction: 0 if any is 0, 1 if all are 1, and $\bot$ otherwise.

**Proposition 9 (conditional soundness).** If every accepted piece of evidence indeed satisfies H1–H3 or W1–W2, then

$$
\hat D_M\neq\bot\ \Rightarrow\ \hat D_M=D_M,\qquad \hat B\neq\bot\ \Rightarrow\ \hat B=B.
$$

Proof: the former follows from Propositions 1 and 2. For the latter: $\hat B=1$ requires all three to be confirmed 1; $\hat B=0$ requires at least one to be a confirmed 0, and the truth value of $B$ is then also 0. $\square$

Corollary: once one direction is confirmed 0, $B=0$ can be decided, and the other direction may remain undecided (as in R02, R06, and others).

**Proposition 10 (the consequences of the two kinds of verification error are asymmetric).** Let a "false rejection" be rejecting valid evidence, and a "false admission" be accepting invalid evidence.

- A false rejection only moves $\hat D$ from the correct value to $\bot$: it lowers coverage and produces no wrong label.
- A false admission can produce a wrong label, in either direction: accepting an invalid sufficient solution gives $\hat D=0$ while $D=1$ (the R14/T type); accepting an invalid witness gives $\hat D=1$ while $D=0$ (the R08/I type).

Hence a wrong label can only come from the event "invalid evidence was proposed and accepted".

**Proposition 11 (conjunction over multiple verifiers).** Let there be verifiers $v_1,\dots,v_k$, and accept only when all accept. Let $FA_j$ and $FR_j$ be the false-admission set and false-rejection set of $v_j$; then

$$
FA=\bigcap_j FA_j\subseteq FA_1,\qquad FR=\bigcup_j FR_j\supseteq FR_1.
$$

Proof: invalid evidence is accepted by the conjunction if and only if it is accepted by every verifier; valid evidence is rejected by the conjunction if and only if it is rejected by some verifier. $\square$

Implication: adding a verifier and taking the conjunction cannot increase wrong labels, and cannot increase coverage. If the false admissions of the verifiers were mutually independent, the false-admission probability would be $\prod_j\alpha_j$; but independence typically fails under the same base model, so only the set-inclusion relations can be relied upon. Therefore whether splitting verification duties is worthwhile depends on whether the new verifier can reject the evidence that the old verifier admitted, and this can only be answered by experiment, not guaranteed by structure.

**Proposition 12 (bounds under a fixed denominator).** For a pre-fixed set of $N$ questions, let $n_1=\#\{\hat B=1\}$ and $n_\bot=\#\{\hat B=\bot\}$. Under the premise of Proposition 9, the number $m$ of questions that truly satisfy $B=1$ satisfies

$$
n_1\le m\le n_1+n_\bot .
$$

Proof: questions with $\hat B=1$ have $B=1$, questions with $\hat B=0$ have $B=0$, and undecided questions may be either. $\square$

The interval $[n_1/N,(n_1+n_\bot)/N]$ expresses the effect of undecided cases and is not a sampling confidence interval; once a false admission occurs, it no longer guarantees that it contains $m/N$. The $[0\%,15\%]$ for the 20 real original questions should be understood in this way.

## 8. What the argument does not settle

- The boundary of $K$ and the scope of $C$ are conventions, not theorems. Changing $K$ or $C$ gives a different object of evaluation.
- Whether H1–H3 and W1–W2 hold is judged by model verification and named review, both of which can err. Propositions 9–12 all rest on the premise that "accepted evidence is valid"; the three recorded false admissions show that this premise is currently sustained by end-stage review, and that review is AI review commissioned by the researcher, neither blind nor an independent human gold standard.
- Validity $V$ is given by a separate review; this document does not discuss its determination.
- The answer-equivalence relation $\sim$ and the segmentation of the "complete answer" must be determined per question; the consequences of wrong segmentation are given in Proposition 8.
- The dichotomy of Lemma 1 holds only when $C_T\neq\varnothing$; if the retained input itself prevents the question from being well posed, this belongs to validity rather than necessity.
- Whether a question is easy to understand is outside these definitions. A question can satisfy $B=1$ while being obscurely worded, and this requires separate diagnostics and reader evidence.

## 9. Three operational conventions of the evaluation implementation

Status: obtained on 2026-11-13 from a small-scale trial on 64 automatic candidates and 26 control questions; it does not change the definitions in Section 2 and only fixes "how to judge".

**Convention 1: compatible completion includes the presuppositions of the question stem.** Let $\Pi(q)$ be the presuppositions implied by $q$ (including implicit presuppositions such as "the exchange, plan, or answer asked about is indeed feasible and exists"). The membership condition "does not violate $q$" for $C_T,C_I$ is made explicit as: the completion must satisfy $\Pi(q)$. This is a direct corollary of Proposition 6. Example: for the lunch-exchange problem, "neither side has the desired food in their lunch, so no exchange is possible" does not belong to $C_I$ and cannot serve as a witness. Sufficiency judgments may likewise use $\Pi(q)$ as a premise.

**Convention 2: answer equivalence $\sim$ is determined by the slot pattern of the question stem.** Generate the slot pattern $\sigma(q)$ from $q$ alone (without looking at the material or any answer): the minimal independent values needed to answer and their granularity, with one value per slot. Two complete answers are equivalent if and only if, after extracting values according to $\sigma(q)$, all slots have the same value after normalization; if any slot lacks a value or the two extracting sides disagree, the case is recorded as undecided, with neither equivalence nor non-equivalence judged. Granularity is thus determined by what is asked (for example, "what food can be exchanged" does not distinguish tomato slices from tomato sandwiches). In the trial, pairwise semantic judgments were affected by judge fluctuation, and the AUC of cosine-threshold calibration on embeddings was only 0.65; neither is usable.

**Convention 3: evidence is accepted by agreement of two groups or a majority of three groups.** Two model groups each propose and recompute the evidence, and it is accepted only when the two groups agree (the conjunction of Proposition 11); when the two groups disagree or one side is undecided, a third model group arbitrates on the same task, and two of three decide; if there is still no majority, the case is recorded as undecided (Proposition 3: failure does not propagate).

**Known boundary: incidental invariance.** Section 2 decides necessity by answer, so when the answer is the same for all $t'\in C_T$ (the "symmetry" of Proposition 5, e.g. a 2:2 square configuration gives 1/2 for either dominance relation), $D_T=0$, even if solving the problem would in practice still read both materials. This is a boundary of the definition, not an implementation error: this evaluation measures "whether the answer is determined by a single modality", not "whether the derivation uses both materials". If the research goal changes to the latter, a separate definition based on derivation use should be established, which is outside the scope of this document. In the implementation, the judge cannot stably enumerate symmetries; of the 8 square-configuration questions whose answer is 1/2, 5 were recorded as $\hat D=1$ in both runs, and these should be understood as possible false admissions under this boundary.

**Trial readings (for scope indication only).** Among the 64 candidates, 29 were confirmed dual-necessary in both independent runs, 13 were denied in both, 8 had inconsistent labels, and the rest await adjudication; the two runs agreed on 79/90 assignments. The known stable sources of false admission are the symmetry class above (about 5 questions) and the lunch-exchange class with answer-granularity disputes (5 questions); on this basis the robust lower bound is about 19 questions, rising to 29 including disputes; no extrapolation to the full set of 1257 questions has been made.

**Reporting convention.** Dual necessity is a binary property, and "undecided" only means the evaluator has not decided; it is not a result category. Reports distinguish only "verified dual-necessary" and "the rest", with undecided folded into "the rest" and diagnosed separately; the interval of Proposition 12 serves only to illustrate the sensitivity of undecided cases. The confirmed set is usually small, and evaluator confirmations should all be counted only after manual review.


## 10. Information-theoretic formulation: residual uncertainty and partial information decomposition

Status: added on 2026-11-13. This section restates the criteria of Sections 1–7 in information-theoretic language and explains why answerer signals have asymmetric evidential force; the criteria and the evaluator are unchanged, and no new metric is introduced. Proposition numbering continues from Section 7.

### 10.1 Setup

Fix a question $q$ and common knowledge $K$. Let $\mathcal W(q)$ be the set of text-image pairs $(t',i')$ that make $q$ well posed (presuppositions hold, complete answer unique), called **legal worlds**; the actual material satisfies $(t,i)\in\mathcal W(q)$. Take any distribution $P$ on $\mathcal W(q)$, requiring every legal world to have positive probability (full support, in particular $P(I'=i)>0$), and let $(T',I')\sim P$, $A=\mathrm{Ans}_K(q,T',I')$.

When the text is removed and the actual image $i$ is retained, the set of legal completions is the slice $C_T(q,i)=\{t':(t',i)\in\mathcal W(q)\}$, consistent with the definition in Section 1; $C_I(q,t)$ is symmetric.

### 10.2 Per-question level: residual uncertainty

$$
U_T(q,i)=H(A\mid I'=i),\qquad U_I(q,t)=H(A\mid T'=t).
$$

That is, the entropy of the answer over legal completions after removing one modality while keeping the actual material.

**Proposition 13 (entropy form of the per-question criterion).** For any full-support $P$,

$$
D_T=0\iff U_T(q,i)=0,\qquad D_T=1\iff U_T(q,i)>0,
$$

and $U_T\le\log_2|\mathfrak A_T(i)|$, where $\mathfrak A_T(i)$ is the set of answer classes occurring on the slice.

Proof: entropy is zero if and only if the distribution is concentrated on a single point; under full support this means the answer is constant on the slice, which is precisely $D_T=0$ of Definition 1. The upper bound is the maximum of entropy, attained at the uniform distribution. $\square$

Since $A$ is a function of $(T',i)$, we also have $I(A;T'\mid I'=i)=U_T(q,i)$: the text is redundant if and only if, given the question and the image, the text brings no further information about the answer.

**Remark.** Only "whether it is zero" is independent of the choice of $P$; the magnitude in bits depends on $P$, and only the upper bound $\log_2|\mathfrak A_T|$ is independent of $P$. Therefore the criterion uses only "whether it is zero", and the bit counts are descriptive only.

### 10.3 Question level: two-source partial information decomposition

Partial Information Decomposition (PID; Williams & Beer 2010) decomposes the mutual information that two sources $T',I'$ carry about a target $A$ into four non-negative atoms: redundancy $\mathrm{Red}$, text-unique $\mathrm{Unq}_T$, image-unique $\mathrm{Unq}_I$, and synergy $\mathrm{Syn}$, with the consistency equations

$$
I(A;T')=\mathrm{Red}+\mathrm{Unq}_T,\quad I(A;I')=\mathrm{Red}+\mathrm{Unq}_I,\quad I(A;T',I')=\mathrm{Red}+\mathrm{Unq}_T+\mathrm{Unq}_I+\mathrm{Syn}.
$$

Subtracting the three equations gives

$$
I(A;T'\mid I')=\mathrm{Unq}_T+\mathrm{Syn},\qquad I(A;I'\mid T')=\mathrm{Unq}_I+\mathrm{Syn}.
$$

Also $I(A;T'\mid I')=\sum_{i'}P(i')\,I(A;T'\mid I'=i')=\mathbb E_{i'}\,U_T(q,i')$.

Only conditional mutual information is used below, so no specific choice of redundancy measure is required (Williams–Beer's $I_{\min}$, the BROJA of Bertschinger et al. 2014, etc.); disagreements among these measures affect only the further split into $\mathrm{Unq}$ and $\mathrm{Syn}$.

**Proposition 14 (relation between the per-question and question levels).**

- (a) $\mathrm{Unq}_T+\mathrm{Syn}=0$ if and only if $D_T(q,i')=0$ for every image $i'$ in the support of $P$, i.e. for any legal image the text brings no information about the answer (question-level text redundancy).
- (b) if $D_T(q,i)=1$ for the actual material, then $\mathrm{Unq}_T+\mathrm{Syn}>0$.
- (c) the converse of (b) does not hold: text redundancy at the actual image $i$ does not imply question-level redundancy.

Proof: (a) $\mathbb E_{i'}U_T(q,i')$ is a weighted sum of non-negative terms with all weights positive, so it is zero if and only if every term is zero; then apply Proposition 13 and the expression above. (b) $\mathbb E_{i'}U_T(q,i')\ge P(I'=i)\,U_T(q,i)>0$. (c) Example: for some question, on the actual image the table already labels all objects, so the text is redundant; but for another legal image (with unlabeled objects) the answer must be determined from the text, so at the question level the text is not redundant. $\square$

The meaning of this distinction is: the evaluator's evidence (sufficient solution, witness pair) acts on the **actually retained material**, corresponding to the per-question $U_T$; the PID atoms describe the dependence structure of the question over all legal worlds. For a dual necessity confirmed per question, both question-level necessities also hold simultaneously.

**Question-level types.**

| Type | PID condition | Example |
|---|---|---|
| Text redundancy | $\mathrm{Unq}_T+\mathrm{Syn}=0$ | Aquarium problem in which the table carries its own labels |
| Image redundancy | $\mathrm{Unq}_I+\mathrm{Syn}=0$ | The text already gives all information determining the answer, and the image is merely illustrative |
| Synergistic dual necessity | $\mathrm{Syn}>0$ | Square-configuration problem: the text gives the rule, the image gives the genotypes, and either modality alone carries almost no information about the answer |
| Concatenative dual necessity | $\mathrm{Unq}_T>0,\ \mathrm{Unq}_I>0,\ \mathrm{Syn}=0$ | Two independent sub-questions are juxtaposed: each part of the answer comes from one modality |

Determining which type applies requires choosing a PID measure and estimating it, which is not done here; this document uses only the binary necessity, and the type assignment is explanatory only. The integration property $S$ of Section 6 corresponds to the synergistic type, and the concatenative type corresponds to the case in Proposition 8 where "dual necessity appears only after merging components". This is a correspondence, not an equivalence: PID is defined over distributions, whereas $S$ is defined over minimal answer components.

### 10.4 Evidential force of answerer signals

Suppose the answerer cannot see the text and outputs $\hat A$ (determined only by $(q,i)$, independent of the unseen $T'$). Let $\pi$ be the answer distribution on the slice; then under the prior $P$ its probability of answering correctly is $\pi(\hat A)$, with error probability $P_e=1-\pi(\hat A)\ge 1-\max\pi$. The weak form of Fano's inequality is

$$
P_e\ \ge\ \frac{H(A\mid Q,I'=i)-1}{\log_2|\mathfrak A_T|}=\frac{U_T-1}{\log_2|\mathfrak A_T|}.
$$

**Proposition 15 (the evidential force of a match and of a failure is asymmetric).**

- (a) if $D_T=1$, the probability that any text-blind answerer answers correctly is at most $1-\dfrac{U_T-1}{\log_2|\mathfrak A_T|}$ (non-trivial when $U_T>1$ bit): the more spread out the answer distribution, the less likely a correct answer by chance.
- (b) if $D_T=0$, there exists a checkable derivation (a sufficient solution, Proposition 1), and the probability of answering correctly depends only on the answerer's competence $c$ on that question.
- (c) there is no analogous bound for "failure": a question with $D_T=0$ can be answered incorrectly because of insufficient competence, an invalid question, or the prior, and the probability of an incorrect answer $1-c$ is not constrained by the information structure of that question.

Thus a single match that agrees with an independent reference answer and whose derivation can be checked is strong evidence for $D_T=0$ (with likelihood ratio $c/\pi(\hat A)$ under the prior $P$); the likelihood ratio of a failure, $(1-c)/(1-\pi(\hat A))$, can approach 1, so its evidential force is limited. Certification of necessity comes from witness pairs (Proposition 2), not from failures; this is the information-theoretic interpretation of Proposition 4.

### 10.5 Relation to existing PID work

Liang et al. (NeurIPS 2023) compute PID statistics with estimators at the dataset level. This section decomposes at the level of a single question, and the per-question zero versus non-zero is certified by a sufficient solution (Proposition 1) or a witness pair (Proposition 2), rather than estimated.

### 10.6 Limitations

- Propositions 13–15 all presuppose that "accepted evidence is valid" and that $P$ has full support; whether a full-support distribution can be taken on the real material space, and the scope of $C_T$, remain the conventions described in Section 8.
- The four PID atoms have not been estimated on this project's data; the type table is explanatory only.
- The probabilities in Proposition 15 are relative to the prior $P$; the actual material is a single world, and the evidential force of matches and failures is Bayesian in meaning.

### 10.7 Screens as hypothesis tests (added 2026-11-16)

Section 10 answers "what does a failure or a match establish?". This subsection states the operational consequence
used by the answerability gate, and the third negation that the two-source formulation leaves implicit.

**The third negation.** Let $K$ denote the answerer's prior knowledge. A question whose stem plus $K$ fixes the answer
has $U_T=U_I=0$ by Proposition 13, for every legal completion, because $A$ is then constant on every slice: such a
question is neither text- nor image-necessary, whatever the passage in front of the evaluator happens to state. Thus
the dual-necessary condition is

$$
U_T>0,\qquad U_I>0,\qquad H(A\mid q,K)>0 .
$$

The third term is implied by the first two whenever $C_T$ is taken over all legal completions, so it is a
completeness condition on the evaluator rather than on the definition. It becomes binding in practice because the
operational screen compares the passage against the *answer* ("does this text state it?"), not against the set of
legal replacements ("is the answer invariant over passages consistent with this image?"). A question naming a
well-known fact passes the first test and fails the second.

**Proposition 16 (the screen is exact for a faithful answerer).** Call $A_\theta$ faithful on side $S$ if it outputs an
answer exactly when $H(A\mid S)=0$. Then rejecting a question whenever $A_\theta$ answers from $S$ is sound (every
kept question satisfies $H(A\mid S)>0$) and complete (every question with $H(A\mid S)>0$ is kept).

Proof: immediate from Proposition 13 and the definition of faithfulness. $\square$

**Proposition 17 (the two ways faithfulness fails).** Let
$\ell=P(A_\theta\text{ answers}\mid U_S>0)$ and $s=P(A_\theta\text{ answers}\mid U_S=0)$. Then the single-side screen
has recall $1-\ell$ and precision $\pi(1-\ell)/[\pi(1-\ell)+(1-\pi)(1-s)]$, where $\pi$ is the dual prevalence, and it
improves on random selection exactly when $s>\ell$.

Proof: count the four cells; the inequality reduces to $s>\ell$ after cancelling $(1-\pi)$. $\square$

$\ell$ is the Fano-limited quantity of Proposition 15(a): a text-blind answerer's chance of hitting the answer on a
slice with $U_S>0$ bits of residual uncertainty. $1-s$ is the competence failure of Proposition 15(b), made explicit
here as a rate rather than an error bound. The screen is therefore useful when the answerer's competence advantage on
redundant rows exceeds its guessing rate on duals; a weak reader (low $s$) or a willing guesser (high $\ell$) makes it
useless even when both are individually unremarkable. Section 11 derives the decomposition,
its consequences for sampling, and the measurements that check it on MMQA and ScienceQA.

## 11. The method theory: screens, the gate law, and the closed loop

A theory of necessity screens: when a side-answerability gate can select dual-necessary questions. This part states the definitions the project already uses, derives the exact performance decomposition
of the answerability gate, states the single inequality that decides whether the gate is useful on a corpus, and
records the measurements that check each claim. Every quantity is estimated from a run, so the claims are
falsifiable rather than interpretive; Section 11.7 records where they held and where they did not.

This note states the definitions the project already uses, derives the exact performance decomposition of the
answerability gate, states the single inequality that decides whether the gate is useful on a corpus, and records the
measurements that check each claim. Every quantity in the decomposition is estimated from a run, so the claims are
falsifiable rather than interpretive; Section 11.6 records where they held and where they did not.

### 11.0 Why the measuring device is the task's own definition

The quantity this task asks for is not a property of a question's surface. "This question needs both modalities" is a
counterfactual: *remove one side and ask whether the answer is still determined*. In the notation of Section 11.1,

$$
\text{dual}(q,m)\iff H(a\mid T)>0\ \land\ H(a\mid I)>0\ \land\ H(a\mid q,K)>0 .
$$

Every term on the right is a statement about what an answerer can determine under a specified intervention. Three
consequences follow, and they are what make the rest of this note more than an accounting exercise.

**(a) An observational scorer is a proxy for an interventional target.** A judge that reads the question, a reward model
trained on human labels, and a difficulty statistic all estimate a function of the *observed* pair. They can rank
questions well and still measure something else, because none of them removes a side. The only instrument that measures
the definition without approximation is one that performs the intervention: run an answering process with one side
actually removed. The gate is therefore not a heuristic that happens to correlate with necessity; it is the definition
made executable, and the evaluator and the criterion are the same object.

**(b) No reference answer is needed.** Necessity is a *difference of determinability across conditions*, not a comparison
against a gold answer. A criterion that only reads the pattern of conditions $(TI,T,I,0)$ needs no annotated answer, so
labels are not an input. This is a structural consequence of (a), not a cost saving: the semantic content of the
criterion does not contain "the answer is correct".

**(c) Two sides and one answer bound the criterion.** With exactly two modalities there are four conditions, so the
criterion is a constant-size program. That is why one instrument serves four jobs: measuring a corpus, diagnosing a
generator, filtering candidates, and certifying a training round.

### The loop

```
material m ──► generator G ──► candidates Q
                                 │
                                 ▼
                 answerer A under TI / T / I / 0
                                 │
                                 ▼
        criterion C_A(Q): a single side answered? → reject
                                 │
                                 ▼
                kept set ──► fine-tune G' ──► next round
```

The loop needs no external auditor to produce a label, because $C_A$ is defined in terms of $A$; and for the same reason
it can drift without an external auditor noticing. Section 11.3 names the drift exactly: $\ell$ is the rate at which a side
that carries nothing still answers (false rejection of a dual), and $1-s$ is the rate at which a side that carries the
answer is not used (false acceptance of a redundant question).

What makes the loop auditable is that **the same device measures its own reliability**. The blind condition estimates the
tendency to answer with no material at all, which is the part of $\ell$ that is prior knowledge; the single-side
conditions estimate $s$ on a labelled pilot. So the loop does not need an external judge to know whether a round helped;
it needs a small labelled pilot, and the rest of the quantities are measured by the instrument that is already in the
loop.

### Where the loop can be amplified

Because $C_A$ is built from $A$, improving $A$ is a legitimate way to raise the certified supply. Section 11.5 shows why:
the verifier's recall is capped by the answerer's accuracy on the full material, and the per-item repair raised that
accuracy from 0.517 to 1.000 and certified recall from 0.483 to 0.981 without changing the criterion. The amplifier
therefore sits on the **answering** side.

The generating side offers no comparable lever on this stack. E4 measured that the copied span is *load-bearing*:
ablating the attributed heads leaves the question byte-identical (the copy is redundant across heads), and blocking
attention to the source span removes the copy but makes the question confabulate a threshold or lose its criterion. A
generation-side intervention is inadmissible here not because it is hard but because the copied span is what makes the
question well posed. The generator's supply is further bounded by mode collapse: best-of-8 sampling produced no new dual
shape.

So the loop's gradient is asymmetric: answering-procedure improvements change $s$ while leaving the criterion intact,
whereas generation-side improvements require new data or an explicit diversity term. This asymmetry is a measured
property of this stack, not an assumption.

### The loop as a noisy-oracle iteration

Formally the loop is ReST with a noisy filter. If $s>\ell$ the filter keeps duals preferentially and the loop moves toward
the task's target; if $s<\ell$ it keeps single-modality questions preferentially, and each round trains the model to
produce more of them. This is why $s>\ell$ is not only a usefulness criterion for one run, but the **stop rule** for the
iteration: a round executed with $s\le\ell$ cannot be trusted to have improved the data, and the drift is not visible in
the model's own confidence.

### 11.1 Definition: necessity is three negations and one positive term

A material is a pair $m=(T,I)$. A question $q$ has answer $a$ in answer space $\mathcal A$. Write $H(a\mid S)$ for the
residual uncertainty of the answer given the sources $S$ made visible.

- **image-redundant** (*text-anchored*): $H(a\mid T)=0$ — the passage alone determines the answer, so the image
  contributes nothing.
- **text-redundant** (*image-anchored*): $H(a\mid I)=0$.
- **dual-necessary**: $H(a\mid T)>0$ and $H(a\mid I)>0$ and $H(a\mid q,K)>0$, where $K$ is the answerer's prior
  knowledge: the question must not be answerable from the question stem and common knowledge alone.
- **unanswerable**: $H(a\mid T,I)>0$.

The third negation is the one the earlier two-negation statement left implicit. It is not decorative. Take a question
that names a well-known fact ("the 1982 film directed by Alan Parker and starring…"): every legal text $t'$ in the
completion set $C_T$ agrees on the answer, so $H(a\mid T)=0$ by Definition 1 even though the *specific* passage in front
of the evaluator does not contain the answer. An evaluator that checks "does this passage state the answer?" answers a
different question than "is the answer invariant over the passage's legal replacements?" and will certify such rows as
dual. Section 11.6 measures how often this happens: on both corpora, almost never, so the third negation is a definitional
completion rather than the binding constraint.

Two consequences carry the paper.

1. *Redundancy is a property of the pair $(q,m)$*, not of $q$ alone, so every denominator is a row, not a question.
2. *Leakage is a mechanism that drives a conditional entropy to zero.* Restating the text's threshold inside the
   question makes $H(a\mid T)=0$ because the question now carries the value. This is the E1 decomposition: **leak** =
   the question transported the content, **anchor** = the question asked a fact one side already held.

### 11.2 The gate as a hypothesis test

$H(a\mid S)$ is not observable: the true answer is unknown and no ideal answerer exists. What is observable is a real
answerer $A_\theta$ on side $S$. The gate tests the hypothesis $H_1(S):H(a\mid S)=0$ and **rejects the question when
$H_1$ is not rejected** — the conservative direction, because a false rejection costs recall while a false acceptance
costs precision, and only the latter silently inflates the reported dual rate.

Three statistics are available per side.

| level | statistic | what it estimates | failure mode |
|---|---|---|---|
| 1 *support* | $A_\theta$ returns an answer rather than abstaining | $H(a\mid S)=0$ | a willing guess answers a question the side does not determine |
| 2 *stability* | all $k$ samples fall in one equivalence cluster | $H_2(a\mid S)=0$ (Rényi-2 entropy) | a confidently wrong answer is also one cluster |
| 3 *agreement* | the single-side mode equals the full-material mode | the side states the same fact as the pair | on a genuine dual the pair's answer is itself a guess, so agreement measures shared bias, not sufficiency |

Levels 2 and 3 are nested evidence for the same hypothesis, so they trace a precision–recall frontier rather than three
methods: requiring more evidence rejects fewer questions.

**The blind condition.** A fourth measurement, answering with *no material at all*, tests $H(a\mid q,K)=0$. It is not a
level of the same ladder; it is the specificity certificate. If a question is answerable blind, the gate's rejection is
correct for a reason no side-based test can see, and the row should not have been labelled dual.

### 11.3 Exact decomposition of the gate's precision

Let $\pi$ be the prevalence of dual questions, $\ell=P(\text{side answers}\mid\text{dual})$ (*luck*, the false-rejection
rate of one side) and $s=P(\text{side answers}\mid\text{redundant})$ (*sensitivity*, the detection rate). A single-side
rule keeps the questions the side fails to answer, so

$$
\text{recall}=1-\ell,\qquad
\mathrm{prec}=\frac{\pi(1-\ell)}{\pi(1-\ell)+(1-\pi)(1-s)}.
$$

These are accounting identities, not predictions: they restate which rows survive. Their value is that they isolate the
three numbers that govern the gate, and they immediately give the decision rule below. For the two-sided rule
(both sides must fail) with conditionally independent sides,

$$
r_{\text{dual}}=1-(1-\ell_T)(1-\ell_I),\qquad \text{recall}=1-r_{\text{dual}},
$$

and *this* one is a genuine prediction: it is checked against the observed rejection rate among duals, so a gap measures
the dependence between the two sides or the noisiness of the label.

**Theorem (the gate is useful exactly when sensitivity exceeds luck).** For fixed $\pi\in(0,1)$, $\mathrm{prec}>\pi
\iff s>\ell$.

*Proof.* $\mathrm{prec}>\pi \iff \pi(1-\ell) > \pi[\pi(1-\ell)+(1-\pi)(1-s)] \iff (1-\ell) > \pi(1-\ell)+(1-\pi)(1-s)
\iff (1-\pi)(1-\ell) > (1-\pi)(1-s) \iff s>\ell$. $\square$

So a corpus does not need a good answerer, only one whose *gap* between detecting redundancy and guessing on duals is
positive; the size of the gain is governed by $s-\ell$, and the ceiling on recall is $1-\ell$ regardless of $s$. Two
consequences that the measurements confirm:

- **Improving the answerer raises the gain, not the ceiling.** Recall is pinned by luck; a stronger model reduces
  $\ell$ (better calibration, more abstention) and raises $s$, which moves precision along the frontier rather than
  past it.
- **Sampling does not help.** With $k$ samples and the *support* statistic, $\ell$ is replaced by
  $1-(1-\ell)^{k}$, so extra samples inflate luck faster than they raise sensitivity whenever the side is
  uninformative. The gate is a $k=1$ instrument; $k>1$ is only useful for the stability *level*, which measures a
  different thing.

### 11.4 Why level 2 and level 3 lose, quantitatively

Substituting the level-3 statistics into the same decomposition explains the measured frontier without a new
assumption: stability and agreement are properties of $A_\theta$'s prior, shared between the single-side and
full-material runs, so they raise $s$ and $\ell$ together. On dual rows the side has no information, so any agreement
there is the model reproducing its own bias in both runs. The measurements show exactly this: moving from level 1 to
level 3 on MMQA raises recall 0.524 → 0.714 and drops precision 0.786 → 0.366 — a movement along the frontier, with the
gap $s-\ell$ shrinking, not growing.

An earlier draft of this note predicted that level-3 agreements on duals would be *prior-answerable* questions. The
blind condition refutes that: on genuine duals the model answers with no material in 1/21 (MMQA) and 0/24 (ScienceQA)
rows, and agrees with the full-material answer in 0 and 0. So the level-3 agreements are not world knowledge; the
side itself answered, which means either the label is wrong or the evaluator's screen missed a derivation. The
honest statement is that **level 3 is a label-error detector**, and the blind probe is what makes that statement
checkable.

### 11.5 The other instrument: the verifier is capped by composition accuracy

Necessity says the answer needs both sides; it does not say the answerer can produce it. The verifier's rule — select
iff the answerer is right on $T\cup I$ and wrong on each side — is a *feasibility* test with a hard cap

$$
\text{recall}\le \mathbb P(A_\theta\ \text{correct}\mid T\cup I).
$$

A genuine dual whose full-material answer the answerer gets wrong is rejected whatever the single sides do. This is why
the E3 repair ladder moves recall in lockstep with full-material accuracy (0.483 at accuracy 0.517, 0.983 at 1.000 on
the aggregation pool): on the questions where the model composes badly, an unrepaired verifier cannot certify its own
good questions.

The gate and the verifier therefore estimate different terms of the same definition: the gate estimates the two
negations by sampling, the verifier estimates the positive term by accuracy. Gate-only selection admits unanswerable
questions; verify-only selection is capped by composition ability. The method uses the gate first because a rejected
question never needs the more expensive test.

### 11.6 Cost

| instrument | local answers per question | API calls | scales with |
|---|---:|---:|---|
| gate, one side | 1 (+1 for the full-material answer) | 0 | answerer competence $s,\ell$ |
| gate, two sides | 2 (+1) | 0 | both directions of redundancy |
| verifier | 3 | 0 | composition accuracy |
| API PVS | 0 | 1 screen per branch + 1 stitching | judge competence |

The relevant comparison is not "local versus API" but "how many paid calls buy the same operating point". Section 11.7
gives the measured answer per corpus.

### 11.7 What the measurements say

Pools: MMQA 149 unique questions from the 60-material control pool (21 dual, prevalence 0.141); ScienceQA 173 unique
questions from the 48-material confirmation pool (24 dual, 0.139). Gate answers are greedy single samples; API PVS is
`gpt-6.1-sol` on the same questions.

| check | prediction | measurement |
|---|---|---|
| $\mathrm{prec}>\pi \iff s>\ell$ | holds for all four side rules | MMQA text: $s$ 0.930, $\ell$ 0.333 → prec 0.609 > 0.141; MMQA image: 0.211 vs 0.095 → 0.158; ScienceQA text: 0.409 vs 0.292 → 0.162; ScienceQA image: 0.617 vs 0.250 → 0.254 |
| two-sided independence | $r_{\text{dual}}=1-(1-\ell_T)(1-\ell_I)$ | MMQA predicted 0.397, observed 0.429; ScienceQA predicted 0.531, observed 0.583 |
| recall ceiling | recall $=1-\ell$ | MMQA text rule 0.667 = 1 − 0.333; ScienceQA text 0.708 = 1 − 0.292 |
| level 2–3 move along the frontier | $s$ and $\ell$ rise together | MMQA two-sided: level 1 0.524/0.786, level 3 0.714/0.375 |
| sampling inflates luck | $k>1$ lowers precision at the support level | MMQA text rule: $k{=}1$ 0.609, $k{=}3$ stable level 0.349 |
| blind answerability on duals | $\approx 0$ | 0.048 (1/21) MMQA, 0.000 (0/24) ScienceQA |
| verifier cap | recall $\le$ full-material accuracy | E3 pool B: 0.483/0.517 and 0.983/1.000 |

The corpus-level comparison that follows from the decomposition:

| corpus | gate (two-sided, level 1) | API PVS | gate vs API |
|---|---|---|---|
| MMQA, text-anchored ($\ell_T$ low, $s_T$ high) | 0.786 precision, 0.524 recall, 0 calls | 0.625, 0.476, 149 calls | point estimates favour the gate, paired CI includes 0 |
| ScienceQA, image-anchored ($s_T$ low, judge strong) | 0.367 precision, 0.458 recall, 0 calls | 0.523, 0.958, 480 calls | API PVS significantly better on recall |

$s_T=0.409$ on ScienceQA is the whole story: the 8B cannot read the lecture passages well enough to notice that they
already answer the question, so the gate keeps 88 redundant rows. On MMQA the same model reads the passages fluently,
$s_T=0.930$, and the gate attains the best measured precision at zero API cost. The instrument is therefore not
dataset-invariant, and the theorem says exactly which number to measure before trusting it: run the gate on a labelled
pilot, estimate $s$ and $\ell$ per side, and use it only where $s>\ell$ by a margin that survives the cluster bootstrap.

### 11.8 What this note does not settle

- The identities in Section 11.3 are exact but the estimates of $s$ and $\ell$ are per-corpus and per-answerer; nothing
  here predicts them for a new corpus without a labelled pilot.
- The independence assumption behind the two-sided formula is approximately true (gaps of 0.03–0.05), not derived; a
  corpus whose passage enumerates the image should show a larger gap.
- Level-3 agreements remain unexplained in mechanism: they are not prior knowledge, and whether they are label errors
  or missed derivations needs the human review of those specific rows, not a new automatic metric.
