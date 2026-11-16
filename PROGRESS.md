# Project progress

This document is the single entry point for daily progress, rewritten according to the current state (2026-11-16), retaining only key decisions, numbers, failures and evidence locations. Earlier day-by-day logs have been compressed into the section "Early work and old-protocol results"; the original text is in Git history.

## Current status (2026-11-16)

**Route as of 2026-11-16: measure → diagnose → select → verify, all but the judge running locally.** The PVS prompt branch is closed; the probe route below is the 2026-11-13 historical record and is not mixed with these results.

| part | instrument | status | headline number |
|---|---|---|---|
| redundancy source (E1) | API judge decomposition | done (AI labels) | leak 100% synthetic vs 4–13% real; MMQA text-anchored (image redundancy 0.81), ScienceQA image-anchored (0.59) |
| mechanism (E2a) | 2×2 modality grid, 2,000 local answers | done | conditional composition is not cross-modal (II 0.990 is best); aggregation collapses (sum 0.580 TT → 0.230 II) |
| selector | local answerability gate, 0 API calls | **done, law derived** | MMQA either-side recall 0.571 / precision 0.800 (API PVS 0.476 / 0.625); ScienceQA 0.458 / 0.367 (API 0.958 / 0.523) |
| verifier (E3) | repaired local self-verification | done | per-item recall **0.983** = API 0.983 at zero API cost; direct 0.483 |
| generation-side (E4) | head attribution + ablation + attention blocking | closed negative, with mechanism | copy rate 1.000 → 0.000 under digit blocking, but true-threshold fidelity 0.000 and dual 0.000 |
| theory | `docs/necessity_formal_argument.md` §11, §10.7 | written | gate beats random iff $s>\ell$; recall ceiling $1-\ell$ |

**The method line the evidence supports** (one sentence each): (i) redundancy has two sources, leakage and single-source anchoring, and their shares are dataset-specific (E1); (ii) an evaluator only checks "does the side state the answer", which is why necessity is judged by whether a side fails, and the local answerer can perform exactly that test at zero cost (gate); (iii) whether that test is worth anything on a corpus is a measurable inequality, $s>\ell$, computable from a labelled pilot (performance law); (iv) a question that passes the gate still needs a model that can *combine* the two sides to be confirmed, and that ability is repairable (E3 per-item tier); (v) attention-level surgery on the generator cannot manufacture dual questions because the copy is load-bearing (E4).

**Open, in priority order:** (1) human review of the ~28 gate false positives, the E1 50-row sheet and the 199-row PVS review pack — every paper number is still AI-labelled; (2) the pilot-then-deploy rule stated operationally; (3) optional E3 patching tier and the two cheap E2 probes; (4) rewrite `story.md` / `VQG_Proposal.md`, which still describe the PVS-only mainline.

## Paths already tried (do not re-derive)

Compact record of closed routes, so the next agent does not repeat them. Numbers are the recorded ones; artifacts in `outputs/<run_id>/`.

| path | what was done | outcome | verdict |
|---|---|---|---|
| instruction only (`plain-refer-v1/v2`) | prompt tells the model to refer, not repeat | synthetic dual 0.333 → 1.000 at validity 0.983, but MMQA dual 0.136 → 0.072 and inline copying *up* | works only on leak; does not transfer |
| numeric suppression in the prompt | forbid copying the number | dual 0.406, validity 0.921; model writes English number words or truncates | unusable |
| chain-of-thought / two-step answering | make the model write steps | full-material accuracy rises, image hop collapses on bridging | not a repair |
| whitebox hidden-state probes (P3) | predict redundancy/direction from `early`/`question_end` states | M4 within-material AUC 0.523 [0.421, 0.648] | dead; measurement paper fallback only |
| activation steering / clamping | push activations along the text↔image axis | no usable effect on necessity | dead |
| best-of-n sampling (k=8) | buy dual supply by sampling | 480 generations → 86 unique; materials with a dual 20/60 at k=1 and at k=8; only `bridging` ever yields | supply is not buyable |
| head-output ablation (copy circuit) | zero the top-8/16 attributed heads | questions byte-identical; copy 1.000; all-head ablation destroys the model | copy redundant across heads |
| attention blocking to the source span | −1e4 bias on digit / whole-text keys | copy 1.000 → 0.000, but true-threshold fidelity 0.000 (confabulated 10/5/9) and dual 0.000 | copy is load-bearing |
| API PVS (prefilter → two screens → stitch) | the original judge pipeline | ScienceQA 173 q: 0.958 / 0.523 at 480 calls; MMQA 149 q: 0.476 / 0.625 at 149 calls | kept as the upper-bound control |
| local answerability gate | answer under one side, reject if it answers | MMQA 0.571 / 0.800, ScienceQA 0.458 / 0.367, 0 API calls | **kept**: cheap selector + direction diagnosis |
| multi-sample stability / agreement levels | 3 samples, stable cluster, agreement with full material | precision 0.786 → 0.366 / 0.349; sampling counterproductive | rejected |
| blind condition as a filter | answer with no material, reject if answered | only 1/21 and 0/24 duals answer blind | rejected as a filter; kept as a certificate |
| early stop after a wrong full-material answer | skip the two single-side answers | −31% local answers on pool B, recall 0.500 vs 0.483 | kept |
| per-item answering (repair tier) | one row per item with `value`/`counts`, harness aggregates | full-material 0.517 → 1.000; verifier recall 0.483 → 0.983 = API | **kept**: the repair that closes the loop |
| interval psychometrics (E2b) | vary distance to the boundary | effective width ≈ 1 unit; "at least N" read as "at least N−1" (10/13 errors); derived count not confirmed | kept as mechanism |
| schema-conditioned selection | check whether schema explains the leak | 0 of 84 skills contain >1 schema | not a causal lever |

- **Redundancy-source decomposition (E1, 2026-11-15; AI judge `gpt-6-luna`, 593 new calls / 898 cache hits / 1.15M tokens, `outputs/e1-redsrc-v1/`):** among questions already judged redundant, `leak` (the question restates the unnecessary side) is **100% on the synthetic threshold set (167/167)** but only **13.0% ScienceQA expand (72/552), 12.9% confirm (15/116), 13.1% MMQA SAQG (14/107), 8.7% MMQA plain control (9/103), 4.1% MMQA refer-v2 (5/122)**; the rest are `anchor` (the question asks something the retained side already contains). Leak is concentrated in the `both_suff` cells (control 7/11, SAQG 5/18, sq-confirm 12/47) and is negligible for image-side redundant questions (2/89 control). **Decision (pre-registered gate: MMQA leak share < 0.20):** the copy-circuit intervention E4 is demoted to a synthetic-set analysis; the paper's method rests on repaired self-verification (E3). The programmatic screen is a weak proxy (sq-expand 4.9% vs judge 13.0%, MMQA control 13.6% vs judge 8.7%), so only judge labels are quoted; the 50-row stratified manual sheet is pending human review, and every share above is AI-labelled. A strict re-read of the 28 MMQA `leak` labels (`outputs/e1-redsrc-v1/source_addendum_manual_read.md`, AI read) finds 14 of them are description-elicitation repeats rather than copies, so strict MMQA leak is **≤ 12/332 = 3.6%**; the pre-registered gate holds under either reading.
- **Composition grid (E2a, 2026-11-15, local GPU, 2,000 answers, `outputs/grid-e2a-v3/`):** the 2×2 modality grid (rule in text/image × values in text/image, 100 materials, paired) shows that conditional composition ("which items count") is **not** a cross-modal binding problem: 0.840 TT / 0.810 TI / 0.890 IT / 0.990 II, `cross − same` **−0.065 [−0.120, −0.010]**, and the cell holding both facts in one image is the *best*, not the worst. The psychometric curve is near-ceiling below the threshold (≤−4, −3, −2 all 1.000; −1 0.957) with a shallow counting-side dip (+1 0.950, +2 0.914, +3 0.917, ≥+4 0.987); errors are omissions, and `missed_one_A` alone is 17 of 47 failures (a text-list tracking effect, item A 0.950 → item D 0.988). The **sum** question collapses in every cell (0.580 TT / 0.320 TI / 0.430 IT / 0.230 II) despite 100% single-hop accuracy (`arithmetic_error` 157, `listed_values_not_summed` 82). Reading: the composition gap is **multi-step aggregation over read values**, worst when the values are read from an image, not modality binding; the sharp boundary-width defect stays specific to interval endpoints. This triggers the pre-registered contingency — the mechanism wording is now aggregation/precision, and **E3 (repaired self-verification) remains the main line**.
- **Repaired self-verification (E3, 2026-11-15, local GPU + ~880 API calls, `outputs/self-verify-{synth-v1,synth-v2,derived-v1,derived-v2,derived-v3,derived-compare-v1,mmqa-v1}/`):** the local 8B replaces the API screens ("select iff TI correct and both T and I wrong"), and repair is graded in five tiers: direct, two-step, delegated (model returns the qualifying values, the harness sums), delegated-compare (harness reads rule + values and decides), per-item (one row per item with value and counts, harness aggregates). On **60 designed aggregation duals**, the tier where the composition gap lives, recall runs **0.483 → 0.783 → 0.850 → 0.933 → 0.983 (59/60)** with precision 1.000 throughout; `per-item − direct` is `+0.500 [+0.367, +0.633]` paired over materials, and per-item matches **API PVS at 0.983** with zero API calls. On the 8B's own 231 generated questions (64 bridging duals + 167 single-image leaks), direct recall is already 1.000 (TI 0.900) and per-item mainly buys precision: **0.865 → 0.985, false-positive rate 0.060 → 0.006**. The error taxonomy is the mechanism signal: delegated's 13 failures are **10 `included_threshold_minus_one` + 3 `dropped_qualifying`** — "at least N" read as "at least N−1" while the arithmetic is right — so the repair works because it makes the comparison per item, exactly where E2b located the defect. Early stopping after a wrong TI answer saves 31% of local answers on pool B with recall unchanged (0.500 vs 0.483). Coverage is now the binding constraint: 60/180 materials yield a selected dual at 1.28 questions per material, so the generator's dual supply, not the verifier, limits selection. **MMQA real-data pilot: underpowered negative** — 120 pool rows collapse to 89 unique questions, only 7 duals, recall 1/7 (5 of 7 are refused by the 8B under every condition), T-acceptable 0.795 (label-validating), so the local signal does not transfer to MMQA duals at this n.
- **Generator supply is not buyable by sampling (2026-11-15, `outputs/bon-synth-v1/`, `outputs/bon-synth-verify-v1/`):** 60 materials × 8 samples at temperature 0.7 (`plain-v2`) produce 480 generations but only **86 unique questions (1–3 distinct per material)**; the materials that yield a dual are **20/60 at k=1 and still 20/60 at k=8**, and they are all `bridging` — the 20 `derived` and 20 `stitching` materials yield zero duals, because the model rewrites the rule into the question ("Which items have a value of at least 4?"). Best-of-8 therefore buys +27 unique questions and +5 selected duals for 8× the generation cost (precision 0.833 → 0.793 under direct, 0.958 under two-step with 24 selected). With the verifier no longer the constraint, **the binding constraint is a mode-collapsed generator that has only one dual shape**, and sampling cannot move it; on this set only a generation-side intervention can.
- **E4 copy-source intervention (2026-11-16, local GPU, `outputs/copy-heads-v1/`, `outputs/copy-ablate-*`, `outputs/copy-mask-e4-v1/`):** the leak *is* a copy — 20/20 derived materials put a context numeral into the question and attribution concentrates it in heads L02H31/L29H11/L28H23 (19/17/15 of 20 materials in the top-10) — but it **cannot be switched off at the circuit level**. Head-output ablation of the top 8 or 16 heads, or a random 8, leaves the question byte-identical (copy 1.000, dual 0.000) while ablating all 32 heads × 36 layers destroys the model, so the copy is redundant across heads. Blocking the generated positions from attending to the context numeral (working bias, verified: one numeral token, position 172) removes the copied value (**copy rate 1.000 → 0.000**) but the question then states a **confabulated threshold**: 0/80 of the blocked questions carry the true threshold, the first numeral is 10 in 44/80 cases, and the question still suffices from the image alone (single-image 1.000, dual 0.000). Blocking the whole text span removes the criterion entirely (threshold mention 0.000, "Which is the largest value?", single-image 0.963, and the model starts answering its own question in the output). **The copy is load-bearing**: a dual question would have to *use* the rule without quoting it, which no attention-level intervention on the redundant side can produce — the working lever is the question schema, which is exactly why the refer prompt reached dual 1.000 on this set. This is a mechanistic justification of the E1 demotion, not just an empirical one: E4 closes as a negative result plus the mechanism it exposes, and the method stays with framed generation + repaired local self-verification.
- **Local answerability gate, first pass (2026-11-16, local GPU, 0 API calls, `outputs/mmqa-answerability-v1/`, `outputs/sq-answerability-v1/`, `outputs/pvs-mmqa-aa-v1/`)** — superseded by the completed tables in the next bullet, which carries the blind control, the multi-sample check and the performance law. What this pass established: on **MMQA** (149 unique questions, 21 duals, base rate 0.141) the passage alone answers 0.846 of questions and the image alone 0.195, so the corpus is text-anchored as E1 said; the text rule selects 23 (recall 0.667, precision 0.609, FPR 0.070), the either-side rule 15 (0.571 / 0.800 / 0.023), and API PVS on the same questions 16 (0.476 / 0.625 / 0.047). On **ScienceQA confirm** (173 unique questions, 24 duals, base 0.139) the direction is reversed — text-answered 0.393 vs image-answered 0.590 — so the text rule collapses (105 selected, precision 0.162) while the either-side rule holds 30 selected at precision 0.367 [0.219, 0.545], lift +0.228 [+0.036, +0.455], recall 0.458. Reading: **which side answers identifies the corpus's redundancy direction**; the single-side rule is the sharp one when the direction is known, the two-sided rule is the portable one.
- **Answerability screens: complete tables, blind control, and the performance law (2026-11-16, local GPU, `outputs/{mmqa-answerability-v2,sq-answerability-v2,pvs-sq-aa-v1}/`):** the tables are now complete on both corpora and against API PVS.
  - **ScienceQA API PVS control** (`outputs/pvs-sq-aa-v1/`, `gpt-6.1-sol` on the same 173 questions): 44 selected, **recall 0.958 (23/24), precision 0.523**, FPR 0.141 — so on ScienceQA the API judge is significantly better than the local gate (paired material bootstrap: recall diff [−0.778, −0.154], precision diff [−0.358, +0.043]). On MMQA the local gate matches or beats it (0.786/0.524 vs 0.625/0.476, CI including zero). The union `either-side ∨ PVS` reaches recall 0.958 at precision 0.434 on ScienceQA and 0.667 at 0.552 on MMQA.
  - **Blind control (new `--conditions ... ,0`)**: the model answers with *no material at all* in 18/149 MMQA and 19/173 ScienceQA rows, but among genuine duals only **1/21 (0.048)** and **0/24 (0.000)**, and it agrees with the full-material answer in 0 and 0. This refutes the earlier hypothesis that level-3 false positives are prior-answerable questions: they are not world knowledge, so the side really did answer and the row is either a label error or a missed derivation. The blind probe is kept as a specificity certificate for the pool, not as a filter.
  - **Sampling does not help**: with 3 samples at temperature 0.7 (`*-answerability-v2`), the support-level rules are unchanged or worse (MMQA text rule 0.609 → the `stable` level 0.349; either-side 0.786 → 0.366), because $k$ samples replace the luck rate $\ell$ by $1-(1-\ell)^k$. The gate is a $k{=}1$ instrument.
  - **The performance law** (`docs/necessity_formal_argument.md` §11, new; §10.7, Props 16–17): with dual prevalence $\pi$, luck $\ell=P(\text{side answers}\mid\text{dual})$ and sensitivity $s=P(\text{side answers}\mid\text{redundant})$, the gate has recall $1-\ell$ and precision $\pi(1-\ell)/[\pi(1-\ell)+(1-\pi)(1-s)]$, and it beats random selection **iff $s>\ell$**. All four side rules satisfy the inequality (MMQA text $s$ 0.930 vs $\ell$ 0.333; ScienceQA text 0.409 vs 0.292), the two-sided independence prediction matches (0.397 vs 0.429 MMQA, 0.531 vs 0.583 ScienceQA), and the recall ceiling $1-\ell$ holds exactly. The ScienceQA weakness is a single number: $s_T=0.409$, the 8B cannot read the lecture passages well enough to notice they already answer the question, where on MMQA $s_T=0.930$. This is the "why it works and when it does not" statement the method line was missing.
- **Theory of the two instruments (2026-11-16, `docs/necessity_formal_argument.md` §11, §10.7):** necessity is restated as **three negations and one positive term** ($H(a\mid T)>0$, $H(a\mid I)>0$, $H(a\mid q,K)>0$, $H(a\mid T,I)=0$); the third is implied in theory but violated by an evaluator that checks "does the passage state the answer" instead of "is the answer invariant over legal passages". The gate is stated as a hypothesis test with a faithfulness assumption (Prop 16) and its two failure rates (Prop 17); the verifier carries the feasibility cap recall ≤ full-material accuracy; the two instruments estimate the two negations and the positive term respectively, which is why the method runs the gate first.
- **Next in dependency order (new paper plan):** the comparison tables are complete (both corpora, local gate vs API PVS, blind control, multi-sample), so the remaining work is consolidation rather than new measurement.
  1. **Human review of the gate's false positives** (the specific rows, ~28): level-3 agreements on duals are not prior knowledge (blind 0/24 and 1/21), so they are label errors or missed derivations. That read decides whether the gate's precision ceiling is a defect or the label's; it is the last open mechanism question of the method line.
  2. **Pilot-then-deploy rule** from the law: on a new corpus estimate $s$ and $\ell$ per side on a labelled pilot and use the gate only where $s-\ell$ survives the cluster bootstrap. This is the operational form the paper should state instead of "the gate works".
  3. E1 and PVS manual sheets still block every paper number (human review).
  4. Remaining optional measurements: E3's mechanistic (patching) tier; scaling the MMQA/ScienceQA real tier past n=89 / 7 duals; the two cheap E2 probes (numeric-representation decoding from image vs text tokens, text→TI value patching on the paired cells).
  5. Then rewrite `story.md` / `VQG_Proposal.md`, which still describe the old PVS-only mainline. The method line the results support is: E1 decomposition + direction-aware answerability gate (with its performance law) + repaired local self-verification, with E4 as a mechanistic negative appendix.
- **Generalization test (2026-11-15):** the synthetic `plain-refer-v2` prompt **does not generalize to MMQA**: dual necessity falls (consensus 13.6% → 7.2%; official cascade 6.4% → 2.8%), **inline copying rises (32.2% → 46.7%, +14.4 pp, material-clustered CI [+4.4, +25.0])**, and the questions become more answerable from the **text alone**: image redundancy 81.2% → 89.5% (+8.3 pp, CI [−0.6, +18.1]) while text redundancy barely moves (10.4% → 12.4%, +2.0 pp, CI [−4.4, +8.7]); answerability improves slightly (81.9% → 85.6%). The prompt therefore intensifies the real failure mode (a text fact question with a decorative image) instead of fixing it. The earlier wording "text redundancy rises 81.2% → 89.5%" inverted the formal definition and is withdrawn (see the work log entry for 2026-11-15). The interval mechanism probe finds the text interval **is used**: removing it changes 24/24 answers, image-only reproduces 0/24 wrong counts, and the wrong count locks in between layers 22–26 while the correct count never becomes decodable; the defect is **boundary precision**, not attention routing. Attention knockout is not worth pursuing; a per-bound comparison prompt is the cheap next test. Decision: stop the MMQA prompt branch; the interval result is a controlled synthetic probe (R4: only 6/25 hard MMQA duals are rule-on-value), not the paper's main real-data story; measurement paper remains the fallback.
- **R1-R4 follow-up (2026-11-15):** R1 confirmation with new seed/range and a new `interval` template: sum derived `+0.533` [+0.367, +0.683], interval `+0.583` [+0.450, +0.717]; count derived `+0.000` [-0.117, +0.117] (not confirmed), count interval `+0.383` [+0.250, +0.517] (24/58 clean off-by-one failures). R2 (numbers recomputed under the formal definition in `outputs/inline-r2b-v1/`): **image** redundancy dominates MMQA and inline questions are *more* image-redundant — plain control 88.9% inline vs 77.4% non-inline, SAQG 50.8% vs 25.8%; **text** redundancy is 24.2% inline vs 3.5% non-inline in the plain control (inline copying does predict a redundant text side when no refer prompt is active) but reverses in SAQG (13.1% vs 21.0%). R3: reference prompt v2 reaches dual 1.000 with local validity 0.983 (original 0.333/0.988); numeric-token suppression only reaches dual 0.406 and validity 0.921 because the model writes number words. R4: 6/25 (24.0%, Wilson [11.5, 43.4]) of the hard MMQA duals are text-rule-on-image-value; strict AI coder 2/25 (8.0%). Reports: `outputs/mechanism-w1-confirm-v1/confirmation_report.md`, `outputs/mechanism-w1-v1/{appendix_versions,r2_inline,r3_interventions,r4_external_validity}.md`.
- **Composition-gap mechanism route (2026-11-15, week 1 of 2):** new synthetic sets `synthetic-composition-v1` (sum endpoint) and `synthetic-composition-count-v1` (boundary-clean count isolation), 180 materials each, programmatic full + two single-hop answers. Qwen3-VL-8B sum endpoint: derived deficit `+0.467` [+0.350, +0.583], 28/60 both-hops-correct-but-full-wrong; bridging and stitching at ceiling (deficit 0). The sum deficit is partly arithmetic: 11/28 are set-selection errors, 11 are arithmetic/other. The count isolation removes summing: deficit `+0.150` [+0.067, +0.250], 9/60 failures, all counting one extra value just below the threshold. Two-step prompting raises sum full accuracy 0.533 → 0.850 (+0.317) and count full accuracy 0.850 → 0.967, but the image hop collapses, so the mixed deficit is −0.150. 2B/4B/8B sum deficits 1.000/0.617/0.467 and count deficits 0.400/0.150/0.150; dual-generation rates 0.125/0.338/0.333 (descriptive only; the generator copies the text threshold into Q on derived/stitching, so those questions are image-only). Stop-loss **not triggered**; the supported claim is conditional filtering, not all serial composition. Report: `outputs/mechanism-w1-v1/week1_report.md`.

- **Environment:** Python 3.11.11, torch 2.11.0+cu128, transformers 5.22.2, `uv sync`; **183 offline tests pass**; `scripts/sanity.py` passes. The container reports 64 CPUs but is capped at 8; `src/egvqg/__init__.py` caps OMP/BLAS/torch at 8.
- **Data:** ScienceQA `lecture-hint-image-v1` (1257; 1132 lecture+hint, 125 hint-only), `lecture-image-v1`; MultiModalQA 60 paragraph+image materials; WebQA is absent and not needed.
- **GPU mainline:** `qwen3vl8b-cand-expand-v2` complete (253 parents, 2024 candidates, distances, local answering); `detail-capture` for pilot/confirm (480 each; hidden_last, per-head `o_proj`, pooled image/text; 4.1–4.2 GB each).
- **Labels:** expand cascade (1846 calls) + consensus repair (457; 1102 new calls); confirm consensus repair (440/440; 276 calls). Inter-judge agreement 0.923 expand / 0.913 confirm; the historical cascade label is weaker (legacy non-redundant precision 0.79 on expand). Consensus rule: two independent judges must agree; disagreements stay undetermined; stitching is filtered.
- **Schema/PID:** dual necessity is schema-determined. Pooled confirm+expand: probability 60/60 S, exchange 19/26, classification 6/26, naming/explanation/comparison ≈0; 7/105 S are stitching. Artifact: `outputs/qwen3vl8b-schema-pid-v1/schema_table.md`.
- **Material variants (19 paired parents):** S hint 23.9%, lecture+hint 27.3%, lecture-only 18.6%, unique-v1 prompt 11.8% (R 56.9%); no variant significantly beats hint; the unique prompt is significantly worse.
- **Attention/steering:** question-end cross-modal attention predicts material-level S (0.979 redundancy, 0.954 direction) but within-skill 0.665 and only 3 mixed materials; attention bias raises image attention 0.63→0.86 but S +2.5 pp (CI includes 0), R rises. Contrastive decoding is unstable (gibberish at λ≥1; adaptive constraint no effect).
- **Generation:** SAQG v1 positive 0.370; SAQG v2 0.465 with **0 stitching**; plain lecture+hint 0.600. Forced SAQG on negatives 0.077.
- **PVS (portfolio + verify/select):** ScienceQA 60 portfolio coverage 56.7%, delivered 70.6%, oracle 43.3% (control 33.3/90.0/31.7; SAQG 43.3/61.5/30.0); MultiModalQA 60 portfolio coverage 88.3%, delivered 77.4%, oracle 81.7% (control 26.7/75.0/20.0; SAQG alone 85.0/82.4/80.0). See `outputs/qwen3vl8b-pvs60-v1/cross_dataset_report.md`.
- **Clean judge isolation (2026-11-15):** selection `gpt-6.1-sol`; evaluation jury `gpt-6-luna` + `DeepSeek-V4.1-Flash`; no shared labels. Budgeted 50-material subset: 100 candidates → text_suff 38, image_suff 25, stitching 0, dual 37 → delivered 21/50 = 42%; jury 16 dual / 1 not_dual / 4 undetermined; delivered precision 16/21 = **76.2%** conservative, 16/17 = **94.1%** among decided; **63 new sol + 89 jury calls**. The aborted 1000 run's 914 sol calls are sunk and not used. Report: `outputs/pvs-clean-1000-v1/subset50_report.md`.
- **Prefilter:** local Qwen v3 structured prompt recall 6/10 oracle-positive, false-positive 0/10; v1 10/10 + 10/10; v2 3/10 + 1/10. Main run uses pass-all; prefilter is an ablation only.
- **Unresolved:** material ceiling (oracle coverage 43.3% ScienceQA, 81.7% MMQA in the 60-material pools); 4 DeepSeek undetermined in the clean subset; weak prefilter; the full 1000 run was aborted for budget; labels are AI consensus, not human; the heuristic schema classifier leaves ~66% "other"; MMQA has no skill/affordance labels.
- **Next:** 100–200 material clean PVS estimate; reduce DeepSeek undetermined; audit delivered questions; then write Measurement / Diagnosis / Solution.

- **Research route:** probe route P0–P7 ([implementation route section 7](docs/implementation_plan.md)): where inside the generator, and on what basis, it decides which modality a question depends on; only question Q is produced, answer A is out of scope.
- **Protocol:**
  - Old protocol `plain-v1`: Chinese generation prompt with wording such as "educational text" and "clear, natural, answerable from the material", Qwen3-VL-8B, transformers 4.57.6.
  - New protocol `plain-v2`: a fixed neutral English prompt (`Based on the context and the image, ask one question.` / `Context:` / `<T>$T</T>` / `Output only the question.`), unified environment transformers 5.22.2, torch 2.11.0+cu128, flash-linear-attention 0.6.0 (`pyproject.toml` and `uv.lock` are locked). Criteria and thresholds are unchanged, written before any new data (plan 7.7–7.9).
- **In progress:** main P3/P5 analyses are complete on the pilot + reduced confirmation set; MultiModalQA evaluation and steering are not started. The confirmation set is the user-approved reduced one (20 skills × 3 new materials = 60 new parent materials, 480 candidates; documented deviation from 7.7's 320-parent expansion; G2 thresholds unchanged, intervals wider).
- **Context variants (B2, automatic AI labels, 17 parents):** C vs P raises text sufficiency +0.600 [+0.259, +0.882] but not redundancy +0.012 [-0.212, +0.235]; S vs O text sufficiency +0.067 [-0.133, +0.283] and S vs P +0.133 [-0.109, +0.375], both null; C vs S +0.442 [+0.088, +0.750]. The complementary split S does not detectably raise text sufficiency over O/P. See `outputs/qwen3vl8b-variants-v2/variant_comparison.md`.
- **Internal shifts across variants (exploratory, layer 22, prompt_end):** `image_removed` falls monotonically A 0.318 → O 0.293 → S 0.265 → C 0.246 (adjacent intervals exclude 0; length-matched C−P = −0.049 [−0.064, −0.034], 17/17 parents); `shift_diff` rises A −0.171 → O −0.106 → S −0.052 → C −0.043 (C−P +0.040 [+0.020, +0.065]); `text_removed` rises but is explained by length (C−P −0.008 [−0.020, +0.006]); `min_shift` does not rise at matched length (C−P −0.010 [−0.021, +0.003]). The representation trades one modality for the other rather than becoming dependent on both. See `outputs/qwen3vl8b-variants-v2/variant_shift_analysis.md`.
- **P3 probe (B3), pilot → confirmation:** G2 does **not** pass for the pre-registered main task (candidate-level redundancy): probe AUC 0.907 vs input-feature baseline 0.845, difference +0.062 [−0.008, +0.149] at `early` (and +0.065 [−0.042, +0.222] at `question_end`), so criterion (i) fails; selectivity and seeds pass. Secondary tasks pass on the confirmation set: direction AUC 0.840 vs 0.685 (+0.156 [+0.084, +0.236]) and 0.829 vs 0.575 (+0.254 [+0.146, +0.371]); material propensity R² 0.568 vs 0.241 (+0.327 [+0.109, +0.846], n=56). See `outputs/qwen3vl8b-probe-v2/probe_report.md`.
- **P5 shift readouts (pilot → confirmation, 7/8 raw adopted):** direction target at `prompt_end` AUC 0.917 [0.825, 0.983] and at `question_end` 0.902 [0.797, 0.973]; the min-shift direction reading is inverted (0.18–0.21). Redundancy-target min_shift AUC 0.88–0.89, but this is against 0.5 only, not against the input-feature baseline. **R1/R2 robustness fails:** within-skill residual AUCs are at chance for all 8 entries, and R2 gains are null except two small redundancy-target gains (+0.052 to +0.059 on a 0.823 baseline). The raw shift association is therefore mostly template/surface, not a within-template signal. See `outputs/qwen3vl8b-p5-v2/p5_report.md` and `outputs/qwen3vl8b-p5-robust-v2/p5_robust_report.md`.
- **P6 selection baselines (confirmation):** random success (valid and non-redundant) 0.110; intrinsic `min_shift` selector 0.113, difference +0.002 [−0.004, +0.011], G4 fails; on the 10 mixed materials 0.376 vs 0.362, +0.013 [−0.015, +0.063]. The judge-model selector is not yet run. See `outputs/qwen3vl8b-selection-confirm-v2/selection_report.md`.
- **Pilot label audit (AI review, not human; plan 7.7 / B1):** 40 redundant plus all 19 distinct non-redundant questions reviewed from rendered sheets. Redundant precision 37/40 = 0.925 (Wilson [0.801, 0.974]), passing the P1 precision criterion (lower bound >= 0.80); non-redundant precision 12/19 = 0.632 [0.410, 0.809]; estimated non-redundant recall 0.635 (bootstrap [0.420, 1.000]); direction agreement 32/40 = 0.800 [0.653, 0.895]. The non-redundant label is weak: 7/19 were in fact single-modality sufficient. See `outputs/qwen3vl8b-cand-pilot-v2/label_audit_report.md`.
- **New-protocol G1 (plain-v2 pilot):** PASS via criterion (b). 54 eligible materials (>= 4 valid candidates): mixed redundancy 7/54 = 13.0% (fails a), mixed direction 15/54 = 27.8% (passes b, threshold 25%). 480 candidates, 155 distinct questions, 426 valid, 353 redundant, 73 non-redundant, 0 redundancy-undecided; directions image 155 / text 62 / both 136. See `outputs/qwen3vl8b-cand-pilot-v2/g1_report.md`. The pass is driven by direction labels, which the old AI audit found noisy (27/40 agreement); no threshold was changed.
- **API ledger:** 3,273 calls cumulatively (limit 10,000), known tokens about 4.71 million (incomplete count, a lower bound); pilot evaluation 445 calls / 640,350 tokens, variants evaluation 439 / 555,854, confirmation evaluation 478 / 657,795, English evaluator validation 571 / about 790 thousand.
- **English evaluator translation validation (plan 7.10, standard written first, then run):** the 194 questions of `qwen3vl8b-cand-pilot-src-v1` were re-evaluated with the same model; redundant label agreement 142/152 (0.934, Wilson 95% lower bound 0.883), direction agreement 108/116 (0.931, lower bound 0.870, reported only). See `outputs/qwen3vl8b-cand-pilot-en-eval-v1/translation_validation.md`.
- **Conclusions already obtained under the old protocol (plain-v1), all exploratory or null results:**
  - G1 not passed: 52 evaluable materials, redundant mixed 13.5%, direction mixed 17.3% (threshold 25%); after strengthening the prompt, merging by (material, prompt) still did not pass (19.8% / 22.6%). Leaning toward the probe (P3) being registered as a null result; probe scores for P4 and P6 stopped.
  - Label audit (Claude, a non-blind AI review, not human annotation): among 40 items, "at least one modality alone is sufficient" 40/40 (Wilson lower bound 0.912); direction agreement only 27/40, direction labels are noisy.
  - P5 pilot readings: pooled AUC about 0.9, but the "skill one-hot + input features" baseline alone is already 0.84–0.87, the material-level signal is mainly templates and surface features; only two readings passed the within-skill and baseline-adjusted tests (non-redundant/question_end/smaller value; direction/question_end/displacement difference), one of the eight combinations passed both at the same time, the layer was selected on the same data, **not confirmed**.
  - Variance decomposition (20 skills, intervals very wide): about 0.59 of the variance of plain redundancy comes from skill, 0.16 from material, 0.25 within material; direction is about 0.9 determined by skill and material.
  - Strong prompting did not increase the proportion of "valid and non-redundant" (64 vs 49 candidates, significance not tested).
  - Geometry (60 materials): the anisotropic image-minus-text difference changes sign across layers, effective rank is limited by token count and not normalized, all descriptive.
- **Version recheck (P0, 8 materials, plain-v1):** 5.22.2 actually applies the model config's minimum pixel count, images with fewer than 65,536 pixels are upscaled (about 41% of ScienceQA materials; 4/8 in P0); greedy questions identical token by token 5/8; median relative error per layer for hidden state items 0–35 about 0.01, **the meaning of item 36 differs by version** (in 4.57.6 it is the output of the last block, before the final normalization; in 5.22.2 it is after it), the old P0 report has been corrected.
- **VQAG mainline code (2026-11-13, built, dry-run only):** `stats`, `centering`, `candidate_table`, `diag_template`, `diag_curve`, `label_repair`, `select`, `steer`, `factsplit`, `filter_baseline` and `scripts/sanity.py` are implemented as CLI subcommands; sanity passes; no paid API call and no GPU job was made. Details and dry-run estimates are in plan 7.11 and `INVENTORY.md`.
- **VQAG step 0 free readings (2026-11-14, no API/GPU):** 1b is replicated across pilot/confirm and layers 6–34: skill share 0.64–0.80 (layer 22: 0.767 early, 0.702 question_end), material 0.13–0.22, candidate 0.05–0.15; `prompt_end` candidate share is structurally 0 and 0.906 skill. 1c with old labels shows the global AUC is skill-driven; within material, confirm redundancy probe 0.957 vs lexical 0.893 (+0.064 [−0.000, +0.182]) but pilot does not replicate (0.588 vs 0.893, −0.153 [−0.422, +0.089]); within-material direction 0.75 (confirm) vs 0.67 (pilot) with intervals spanning ±0.5. 2c retention 75.2% / 73.7% with ≈+1 pp enrichment (intervals touch 0). **H2 supported descriptively; H3 not established.** Artifact: `outputs/vqag-step0-v1/step0_report.md`.
- **Unresolved:** non-redundant labels are weak (pilot precision 0.632, recall 0.635, AI review); direction label noise; the G2 main redundancy task fails while direction/propensity pass; MultiModalQA and steering await their runs; Qwen3.5 generation and hidden states on GPU unverified.

## Paper outline (draft, 2026-11-16)

The outline now follows the five parts the completed measurements support; numbers in brackets are the ones already produced and verified offline.

**1. Problem and measurement.** Educational VQG is asked to produce questions that need both the passage and the image; existing generators almost never do (ScienceQA 1,257 materials: verified dual-necessary lower bound 19, 1.5%). The instrument is three-valued: `dual_necessary`, `text_suff`/`image_suff`/`both_suff`, `undetermined`; the official cascade leaves `undetermined` explicit rather than scoring it 0, and the human review pack (199 rows, `outputs/review-v1/`) is the validity check that is still owed.

**2. Where the redundancy comes from (E1).** Among already-redundant questions, `leak` (the question restates the side it does not need) is 100% on the synthetic threshold set but 13.0 / 12.9 / 13.1 / 8.7 / 4.1% on ScienceQA expand, ScienceQA confirm, MMQA SAQG, MMQA control and MMQA refer-v2, and a strict re-read of the MMQA labels puts it at ≤ 3.6%. The residual is `anchor`. The same decomposition fixes each corpus's direction: MMQA is text-anchored (image redundancy 0.81), ScienceQA is image-anchored (0.59). Two consequences shape the method: (i) leakage is a synthetic-set phenomenon, so circuit-level intervention (E4) is an appendix, not the main line; (ii) on real corpora the target is anchoring, which is exactly what an answerability test can see.

**3. Why the model cannot certify its own dual questions (E2a, E3).** The composition gap is not cross-modal binding: in the paired 2×2 grid the all-image cell is the *best* cell (0.990) and the significant `cross − same` difference (−0.065) is carried by text-list item dropping. What collapses is multi-step aggregation (sum 0.580 TT → 0.230 II) with 100% single-hop accuracy, and the failure is a **boundary/comparison** error — "at least N" read as "at least N−1" (10 of 13 delegated-tier failures), matching the interval probe's effective width of about one unit. Consequently a verifier whose decision rule is "TI correct and both sides wrong" has recall bounded by its full-material accuracy: 0.483 in the direct tier, 0.983 once the comparison is made per item. This is the mechanism that makes an external judge necessary *and* the reason a local repair can replace it.

**4. Method.** Two local instruments, no API at inference time. (a) **Answerability gate**: run the answerer under each single side and reject the question if a side answers it; direction-aware single-side rules where the corpus direction is known, the two-sided rule otherwise. It is governed by the law `prec > π ⇔ s > ℓ`, so deployment is pilot-then-deploy: estimate `ℓ` and `s` on a labelled pilot and use the gate only where `s − ℓ` survives the material-clustered bootstrap. (b) **Repaired self-verification**: confirm survivors by "TI correct, T wrong, I wrong" using a repaired answerer (per-item comparison), which matches API PVS recall (0.983 vs 0.983) at zero API cost. (c) **E4 as a mechanistic negative**: the copied span is load-bearing — attention blocking removes the copy (1.000 → 0.000) but not the redundancy (dual stays 0.000, true-threshold fidelity 0.000) — so generation-side repair must change the question schema, not the copy circuit.

**5. Evaluation and cost.** Coverage, precision and cost against API PVS on the same questions: MMQA either-side 0.571/0.800 vs PVS 0.476/0.625 (149 calls); ScienceQA either-side 0.458/0.367 vs PVS 0.958/0.523 (480 calls) and the union 0.958/0.434; oracle upper bound from the PVS portfolio (ScienceQA 56.7% coverage, MMQA 88.3%). The honest claim is conditional: the gate is the cheaper instrument when the answerer can read the redundant side (`s = 0.930` on MMQA vs `0.409` on ScienceQA), and the judge remains correct otherwise.

**Non-claims that must stay explicit.** (i) The schema effect is a skill/material effect: in the 1,000-material pool `schema_hint` is perfectly nested in `skill` (0 of 84 skills contain more than one schema), so "probability/exchange questions are dual-necessary" is observationally "Punnett/Trade materials are dual-necessary" (`outputs/qwen3vl8b-schema-pid-v1/schema_table.md`); it is not evidence that schema is a causal lever. (ii) All labels are AI consensus; the 199-row human pack, the E1 50-row sheet and the gate's ~28 false positives are unreviewed. (iii) Pool-B aggregation questions are template-designed, not generated. (iv) The MMQA real-data verifier tier is underpowered (89 unique questions, 7 duals).

## Early work and old-protocol results (condensed)

- **Data:** the 1,257 ScienceQA test materials that "have a hint and an image"; the model sees only the hint and the original image; the original question, options, answer and lecture do not enter the material. MultiModalQA and WebQA have been fully downloaded (the latter's images are still an unextracted multi-part archive), MultiModalQA has been integrated into material import.
- **Evaluator:** cascade evaluation (validity, two single-modality screenings, counterexamples generated only for surviving questions); necessity evaluator (two model groups plus arbitration, answer slots judged for equivalence, independent reference answers). Definitions and argumentation in [docs/necessity_formal_argument.md](docs/necessity_formal_argument.md).
- **luna's baseline (API):** the original evaluator automatically provisionally judges 64 questions doubly necessary (5.1%); the necessity evaluator automatically provisionally judges 38 questions (3.0%), manual spot checking confirms 19, incidental invariance 7, false pass 12, additionally 4 missed cases found, verified lower bound about 23/1257 (1.8%). See `outputs/necessity-v3-full-v1/probe_report.md`.
- **Qwen3-VL-8B baseline (local):** all 1,257 groups generated; valid 962, failed 268, pending adjudication 27; single-modality screening removed 894 questions (93% of valid questions), sampled audit of the screening 40/40 (Wilson lower bound about 0.91); verified doubly-necessary lower bound 19 questions (1.5%), evaluation totals about 4.73 million tokens. See `outputs/qwen3vl8b-necessity-v2/probe_report.md`.
- **Reporting has only two categories:** verified doubly necessary and the rest; "the rest" includes questions answerable with a single modality, questions that themselves fail, and questions not yet adjudicated, listed separately.
- All the numbers above come from the old prompt plain-v1 and transformers 4.57.6, and are not mixed with results of the new protocol.

## Milestones

| Milestone | Status |
|---|---|
| P0 white-box capture | Done: three positions × per-layer hidden states and token spans; recheck under 5.22.2 done |
| P1 candidate dataset | Old-protocol pilot done (60 materials × 8, label coverage 98.5%); new-protocol pilot done (480 candidates, 155 distinct questions), G1 passed; expansion deferred |
| P2 propensity profiling | Old protocol: G1 not passed; new protocol (plain-v2): G1 passed via mixed direction 27.8% (mixed redundancy 13.0%) |
| P3 linear probe | Old protocol: registered as a null result; new protocol: main redundancy task fails G2 criterion (i), direction and propensity pass on the confirmation set |
| P4 causal test | Steering screening code ready (exploratory new branch), awaiting queue |
| P5 mechanism: geometry and readings | Old-protocol pilot exploration done; new-protocol P5 raw shift readouts replicate (7/8) but fail R1/R2 (template/surface); geometry descriptive only |
| P6 selection and steering | Random and intrinsic selection evaluated (both null); judge-model selector not run; steering blocked by the G2 main-task failure per 7.7 |
| P7 second model | Qwen3.5-9B environment and adaptation ready, awaiting GPU verification |

## Handover status and to-dos (2026-11-13)

**Done (this round, 126 offline tests passed):**
- The predecessor's "four generation methods" experimental framework has been removed from the code: `evaluation.py`, `generation.py`, `batch.py` deleted; `experiments.py`, `pipeline.py`, `metrics.py`, `faithfulness.py`, `baseline_analysis.py` keep only the functions actually used by the evaluator and the baseline; the corresponding commands, tests, configs (`stage_a.json`, `experiments*.json`) and fixtures have been deleted.
- `prompts/` reduced from 46 files to 10; the 9 evaluator/answerer prompts and the knowledge scope of `configs/necessity_assessment_v2.json` have been translated into English (placeholders, JSON keys, enums and decision rules unchanged), Chinese in the code that injects prompts or writes into reports has also been removed; in `src/` only the CJK detection range of `candidates.py` still contains Chinese.
- Translation validation passed per 7.10 (142/152, Wilson lower bound 0.883), the artifact is in `outputs/qwen3vl8b-cand-pilot-en-eval-v1/translation_validation.md`.
- Deleted `docs/evaluation_protocol_v1.0.md`; the README command table and plan sections 1–6 have been rewritten per the current CLI; `plain-v1` wording removed from `cli.py`/`whitebox.py`.
- Context variants add S (complementary split, 17 parent materials); the variant comparison of plan 7.8 precedes the data registration (S vs O, S vs P).

**Not done (in order):**
1. Finish the new-protocol pilot: cascade evaluation → `candidate-labels` → G1 per 7.7. The expansion (2,560 candidates) is deferred until G1 is known; all GPU steps use card 3.
2. Non-redundant label recall audit (label precision/reliability of plan 7.7, AI review): stratified sample of 40 non-redundant and 40 redundant candidates from the new pilot, report recall/precision and Wilson intervals.

## Work log

### 2026-11-13: Establishment of the probe route (P0 and the old-protocol pilot)

- New modules: `whitebox.py` (three positions × per-layer hidden states and spans), `candidates.py`, `probe_labels.py`, `label_audit.py`, `probing.py`, `probe_run.py`, `modality_distance.py`, `distance_analysis.py`, `distance_robust.py`, `selection_eval.py`, `geometry.py`, `variance_decomposition.py`, `intervene.py`; CLI and offline tests updated in sync (172 passed in total).
- Criteria and analysis details (G1–G4, G2 details, P5/P6/P4, R1–R3) written into `docs/implementation_plan.md` 7.7, before the corresponding labels; not changed afterwards.
- Positive control: the probes can decode "whether the question is in Chinese" (AUC 0.95–1.0) and material text length (R² 0.79–0.86), indicating the pipeline itself works.

### 2026-11-13: Unified environment and protocol change

- Environment: the project `.venv` unified to transformers 5.22.2; causal-conv1d has no prebuilt package adapted to torch 2.11 and the local CUDA toolkit is 13.x, so it cannot be compiled, and that part of Qwen3.5 falls back to the torch implementation; the old shared environment is no longer used. Backend fixes: `padding` must go into `processor_kwargs`, `mm_token_type_ids` must be padded in teacher-forcing forward passes, Qwen3.5 generates 2 more tokens after `<|im_end|>`.
- Protocol: the generation prompt changed to the neutral English `plain-v2` (plan 7.9); this change happened after plain-v1's G1 not passing, plain-v1's conclusions are retained as they are, plain-v2 is a new protocol instance, the criteria are unchanged, and the strong prompt is no longer generated.
- Pre-registration of new experiments (plan 7.8): context variants (17 materials × O/A/C/P, `data/processed/scienceqa/context-variants-v1`), MultiModalQA generalization (`data/processed/multimodalqa/mmqa-pool-v1`), Qwen3.5 recheck, steering (exploratory, layers 12/20/28).
- Note: `skill` is ScienceQA metadata, it does not enter the prompt and is only used for grouping and held-out splits; other datasets do not have it.

### 2026-11-13: CPU dry run and queue

- Small-scale dry run of the new code paths on CPU with the real model, fixed two problems that would only surface on GPU (missing `--max-new-tokens`; `mm_token_type_ids` length mismatch); steering at 60σ turns generation into gibberish while 1σ is almost unchanged, showing the hook acts on generation.
- The GPU queue was started on card 0: stream A's P0 recheck and plain-v2 pilot capture are done; stream B's Qwen3.5 smoke test was interrupted because the identity file was inconsistent after the change, it was restarted after clearing the empty directory.

### 2026-11-13: English prompts, translation validation and document sync

- The 9 evaluator/answerer prompts and the knowledge scope of `configs/necessity_assessment_v2.json` were translated into English; placeholders, JSON keys, enums and decision rules unchanged. Chinese in the code that injects prompts or writes into reports (`baseline_analysis.py`, `necessity_ensemble.py`, `simple_baseline*.py`) was removed in sync; `tests/test_necessity_ensemble.py` now uses English markers.
- Translation validation passed per plan 7.10 (standard written in first): the 194 questions of `qwen3vl8b-cand-pilot-src-v1` were re-evaluated with the same model (`simple-baseline-eval-init --cascade` + `eval-run`), redundant label agreement 142/152 (0.934, Wilson 95% lower bound 0.883); direction agreement 108/116 (0.931, lower bound 0.870), reported only. Artifact: `outputs/qwen3vl8b-cand-pilot-en-eval-v1/translation_validation.md`; this run used 571 calls, about 790 thousand tokens.
- Documents: deleted `docs/evaluation_protocol_v1.0.md`; the README command table and plan sections 1–6 rewritten per the current CLI; `plain-v1` wording removed from `cli.py`/`whitebox.py`; the old links in `docs/references.md` changed to point to the current definitions.
- Verification: `grep -rP "[\x{4e00}-\x{9fff}]" prompts configs scripts` produces no output; `src/` only has the CJK range of `candidates.py`; `.venv/bin/python -m pytest -q -o addopts=""` 126 passed.

### 2026-11-13: Context variant S (complementary split)

- Hand-wrote the S text after looking at the images of the 17 parent materials one by one: the text gives about half of the required facts, the image carries the other half, and the text does not restate visible content. The S vs O and S vs P comparison of plan 7.8 precedes any variant label registration.
- Data: `data/processed/scienceqa/context-variants-v1/materials.jsonl` has 85 rows (17 parent materials × O/A/C/P/S), `review_status` is pending; no same-day labels.

### 2026-11-13: Whole-document English translation

- Per the user's instruction, all human-facing documents were translated from Chinese into English: `AGENTS.md`, `README.md`, `PROGRESS.md`, `VQG_Proposal.md`, `story.md`, `docs/implementation_plan.md`, `docs/necessity_formal_argument.md`, `docs/references.md`.
- Constraints: structure, tables, links, numbers, thresholds, formulas, run_ids and CLI commands preserved; pre-registered sections 7.7–7.10 translated literally, with every gate, fallback and "report only" label kept distinct. Historical outputs, JSONL data and model outputs were not touched (evidence is not rewritten).
- Verification: `grep -rlP "[\x{4e00}-\x{9fff}]"` over the eight documents returns nothing; all relative links resolve (0 broken); line counts unchanged for all documents (AGENTS 74, README 101, PROGRESS 89, VQG_Proposal 268, story 332, plan 242, formal argument 303, references 102).
- Independent read-only audits: prompt translation fidelity (all 9 prompts + knowledge string OK, no placeholder/JSON-key/decision-rule change) and README/plan vs current CLI (all 30 commands exist; stale module/contract claims fixed).

### 2026-11-13: New-protocol pilot G1 (plain-v2)

- Pilot `qwen3vl8b-cand-pilot-v2`: 60 materials × 8 candidates = 480; 155 distinct questions after de-duplication; cascade evaluation `qwen3vl8b-cand-pilot-v2-eval` used 445 calls / 640,350 tokens (English evaluator).
- Labels: valid 426, redundant 353, non-redundant 73, valid-but-undecided 0; direction image 155 / text 62 / both 136. Six materials had no valid candidate and are outside the eligible denominator, as registered.
- G1 per plan 7.7: eligible materials 54; (a) mixed redundancy 7/54 = 12.96% (below 25%); (b) mixed direction 15/54 = 27.78% (above 25%) -> **G1 passes via (b)**. No threshold, prompt or selection was changed; the pass is not upgraded beyond what (b) states.
- Artifact: `outputs/qwen3vl8b-cand-pilot-v2/g1_report.md`; profile and labels in the same capture run.
- Next: label audit (40 redundant + all 20 distinct non-redundant questions from the pilot), then P3/P5. Expansion is triggered by the P1 criteria but deferred on the user's scale/time instruction; variants (680 candidates, 188 distinct) and MultiModalQA (60 materials) source runs are being produced on card 3.

### 2026-11-13: Pilot label audit, redundant precision and non-redundant recall (AI review)

- `label_audit.py` extended with a two-class mode (`label-audit-prepare --two-class`): 40 redundant-labelled questions stratified by direction plus all distinct non-redundant questions, rendered as large two-item sheets; `label-audit-finish` reports precision, false-omission rate, prevalence-adjusted recall and Wilson/bootstrap intervals. Tests added for sampling and summary.
- Reviewed 59 items (40 redundant, 19 non-redundant) without seeing claimed labels. Redundant precision 37/40 = 0.925, Wilson [0.801, 0.974] (passes the P1 precision criterion, lower bound >= 0.80); non-redundant precision 12/19 = 0.632 [0.410, 0.809]; estimated non-redundant recall 0.635, bootstrap [0.420, 1.000]; direction agreement 32/40 = 0.800 [0.653, 0.895].
- Implication: the non-redundant label is the weak link (7/19 were actually single-modality sufficient), so the true redundancy rate is above the labelled 82.9%; any downstream non-redundant claim must carry this caveat. This is an AI review, not human annotation.
- Artifacts: `outputs/qwen3vl8b-cand-pilot-v2/label_audit_report.md`, `audit_sample_two_class.jsonl`, `audit_reviews_two_class.jsonl`, `audit_summary_two_class.json`, `audit_sheets_two_class/`.

### 2026-11-13: Context variants O/A/C/P/S (B2) and reduced confirmation set

- Variants: 85 materials × 8 = 680 candidates, 188 distinct questions; cascade evaluation `qwen3vl8b-variants-v2-eval` used 439 calls / 555,854 tokens. Labels by variant (valid only): O 124, A 90, C 133, P 118, S 124.
- Paired by 17 parents: C-P text-sufficient +0.600 [+0.259, +0.882], redundancy +0.012 [-0.212, +0.235]; A-P text-sufficient -0.019 [-0.058, 0.000]; S-O text-sufficient +0.067 [-0.133, +0.283], redundancy +0.050 [-0.058, +0.208]; S-P text-sufficient +0.133 [-0.109, +0.375], redundancy -0.023 [-0.273, +0.227]; C-S text-sufficient +0.442 [+0.088, +0.750].
- Reading: writing image content into the text (C) makes the text sufficient; the complementary split (S) does not detectably raise text sufficiency or lower redundancy relative to O/P. Null result for S, with wide 17-parent intervals. Artifact: `outputs/qwen3vl8b-variants-v2/variant_comparison.md` (+ json).
- MultiModalQA: 60 materials captured, 480 candidates, 174 distinct questions, source answering complete; evaluation not started.
- Confirmation set (P3/G2): user chose a reduced set; capture started on card 3 with 20 skills × 3 new materials = 60 new parents × 8 = 480 candidates (`qwen3vl8b-cand-confirm-v2`), `--per-skill 6 --exclude-run qwen3vl8b-cand-pilot-v2`. This deviates from 7.7's 320-parent expansion and is recorded as a scale deviation; G2 thresholds and analysis rules are unchanged.
- Internal shifts across the same variants (descriptive, layer 22 prompt_end, 10,000 bootstrap over 17 parents; layer chosen after the distances existed): `image_removed` falls A 0.318 -> O 0.293 -> S 0.265 -> C 0.246, with adjacent intervals excluding 0; length-matched C-P = -0.049 [-0.064, -0.034] (17/17). `shift_diff` rises A -0.171 -> O -0.106 -> S -0.052 -> C -0.043 (C-P +0.040 [+0.020, +0.065]). `text_removed` rises but is length-explained (C-P -0.008 [-0.020, +0.006]); `min_shift` does not rise at matched length (C-P -0.010 [-0.021, +0.003]). This is the internal correlate of the behavioural "switch sides, no integration" result, not a causal manipulation. Artifact: `outputs/qwen3vl8b-variants-v2/variant_shift_analysis.md` (+ json).

### 2026-11-13: P3 probe and P5 readouts on the reduced confirmation set

- Confirmation run `qwen3vl8b-cand-confirm-v2`: 20 skills × 3 new materials = 60 new parents, 480 candidates, 173 distinct questions, zero material overlap with the pilot; cascade evaluation 478 calls / 657,795 tokens.
- P3 (`qwen3vl8b-probe-v2`): layers chosen on the pilot (redundancy 32/20, direction 22/16, propensity 30). On confirmation, the main redundancy task fails G2 criterion (i): probe AUC 0.907 vs input features 0.845, difference +0.062 [−0.008, +0.149] (`early`); `question_end` +0.065 [−0.042, +0.222]. Direction passes: AUC 0.840 vs lexical 0.685 (+0.156 [+0.084, +0.236]) and 0.829 vs 0.575 (+0.254 [+0.146, +0.371]); propensity passes: R² 0.568 vs 0.241 (+0.327 [+0.109, +0.846], n=56). G2 is reported as **not passed overall**, not as a partial pass.
- P5 (`qwen3vl8b-p5-v2`): direction-target asymmetry AUC 0.917 [0.825, 0.983] at prompt_end and 0.902 [0.797, 0.973] at question_end; min-shift direction reading inverted (0.18–0.21); redundancy-target min_shift 0.88–0.89 against 0.5 only. 7/8 entries meet the P5 adoption rule. This is not the G2 baseline test and is not written as beating template/surface features.
- Geometry (pilot and confirmation, card 2): anisotropy gap changes sign across layers; effective-rank gap large but unnormalised for token count; correlations with material redundancy weak. Descriptive only.
- Interpretation: direction and material propensity are readable above baselines; candidate-level redundancy adds little beyond input features; this is consistent with the B2 behavioural and shift results (switch sides, no integration) and does not establish a causal mechanism.
- Artifacts: `outputs/qwen3vl8b-probe-v2/probe_report.md`, `outputs/qwen3vl8b-p5-v2/p5_report.md`, `outputs/qwen3vl8b-geometry-pilot-v2/geometry_summary.json`, `outputs/qwen3vl8b-geometry-confirm-v2/geometry_summary.json`.
- P5 robustness (R1 within-skill, R2 baseline-adjusted; `qwen3vl8b-p5-robust-v2`): within-skill residual AUCs are at chance for all 8 entries (0.38–0.68, intervals include 0.5), and R2 AUC gains are null in 6/8 entries; the two small positive gains are on the redundancy target (+0.052 and +0.059 over a 0.823 baseline) and still fail R1. Conclusion: the raw shift association is mostly template/surface, not within-template; it is not reported as mechanism evidence. Artifact: `outputs/qwen3vl8b-p5-robust-v2/p5_robust_report.md`.
- P6 selection baselines (`selection-eval`, new CLI; random + intrinsic min_shift, layer 6 prompt_end chosen on the pilot): on the confirmation set, random success 0.110, intrinsic 0.113, difference +0.002 [-0.004, +0.011] -> G4 fails; mixed-material subset (n=10) 0.376 vs 0.362, +0.013 [-0.015, +0.063]. Intrinsic selection does not beat random. Code: `selection_eval.min_shift_scores` + `run_selection`, CLI `selection-eval`, one new test (129 offline tests pass). The judge-model selector is not implemented yet.
- Correction to the earlier expectation: R1/R2 do **not** preserve the direction signal. Within-skill residual AUCs are at chance for the direction entries too (0.47-0.68, all intervals include 0.5); R2 gains are null. Only the P3 full-state probe (judged by its own G2 rule) passes the direction comparison; the P5 scalar shift does not survive template/surface adjustment.

### 2026-11-13: VQAG mainline code (no experiments run)

- Added `INVENTORY.md` (candidate table, hidden-state format, P3/P5/P6/evaluator reuse points, and a list of gaps) and new modules `stats.py`, `centering.py`, `candidate_table.py`, `diag_template.py`, `diag_curve.py`, `llm.py`, `label_repair.py`, `select.py`, `steer.py`, `factsplit.py`, `filter_baseline.py`; CLI subcommands `diag-template`, `diag-curve`, `label-repair`, `vqag-select`, `vqag-steer`, `factsplit`, `filter-baseline`; the single check script `scripts/sanity.py`. Existing probe/distance/selection/evaluator code is called, not copied; `probing.oof` only gained an optional `folds=` argument (all 129 offline tests pass).
- Verification (CPU only): `.venv/bin/python scripts/sanity.py` passes (shares 0.496/0.299/0.205 vs 0.5/0.3/0.2; direction AUC 0.992 after material centering; skill-only AUC 0.500; α=1 clamp error 4e-16; null-bootstrap CI covers 0). All seven commands dry-ran on `qwen3vl8b-cand-confirm-v2` / `labels_...-eval.jsonl` or synthetic facts; no API call and no GPU allocation.
- Dry-run estimates: label repair ≈792 calls (60 parents) and ≈4,224 (320 parents); steering ≈6,720 local generations and ≈3,226 label calls for 60 materials × 14 conditions × 8 candidates, ~22–24 GiB GPU memory. Factsplit: 5 extraction calls for 5 materials (≤10 with retries), 200 generations.
- Pipeline-check observations only (old labels, not results): prompt-end candidate variance share is structurally 0; material centering removes the skill-level direction signal; `vqag-select` balance vs random is +0.037 [+0.001, +0.079] on old labels — not interpretable until `label-repair` runs.
- Not done: real `label-repair`, `factsplit`, steering, and any GPU/API execution; no threshold or protocol was changed.

### 2026-11-14: VQAG step 0 free readings (no API, no GPU)

- Ran 1b on both 60-parent runs at layer 22 and layers 6–34: skill share 0.64–0.80, material 0.13–0.22, candidate 0.05–0.15; cross-run and cross-layer replication. `prompt_end` skill 0.906 / candidate 0.000 (structurally identical states within a material). H2 is descriptively supported.
- Ran 1c with the existing cascade labels (old, known weak) at layer 22, bootstrap 2,000, three seeds. Global AUC reproduces P3, but after skill centering the global number collapses; material-internal redundancy probe is 0.957 vs lexical 0.893 in confirm (+0.064 [−0.000, +0.182]) and 0.588 vs 0.893 in pilot (−0.153 [−0.422, +0.089]); material-internal direction is 0.75 (confirm) vs 0.67 (pilot), intervals spanning ±0.5. H3 is not established with old labels.
- Ran 2c on 20 parents per run: retention 75.2% (confirm) / 73.7% (pilot); kept vs all dual-label rate +0.010 [0.000, +0.026] and +0.014 [0.000, +0.042]. Pipeline check only because `label_new` is absent. Fixed an unpacking bug in `filter_baseline.py` found by this first real run.
- Commands: `diag-template` (run ids `vqag-1b-confirm-v1`, `vqag-1b-pilot-v1`, `vqag-1b-*-layers-v1`), `diag-curve` (`vqag-1c-prelim-confirm-v1`, `vqag-1c-prelim-pilot-v1`), `filter-baseline` (`vqag-2c-confirm-v1`, `vqag-2c-pilot-v1`). Verification: `scripts/sanity.py` passes, 129 offline tests pass.
- Decision: H2 has a clear descriptive answer; H3 does not replicate, so the 320-parent expansion is not funded yet. Next paid step is `label-repair` on 20 parents (~264 calls), then 120 existing parents (~1,584 calls) only if the Wilson lower bounds pass 0.80; then re-run 1c/2a with `label_new`.

### 2026-11-14: CPU thread cap at 8 (container quota)

- Diagnosis: the container is capped at 8 CPUs (`cpu.cfs_quota_us=800000`, `period=100000`) while `os.cpu_count()`/affinity report 64, so torch defaulted to 64 intra-op threads and OMP/BLAS to 64; 127 threads then fought over 8 cores with constant CFS throttling. On 120 `qwen3vl8b-cand-confirm-v2` states files, `load_file + float + mean(0)` took 82.5 s with 64 threads vs 1.2 s with 8.
- Change: `src/egvqg/__init__.py` now calls `configure_threads()` at import, forcing `OMP/MKL/OPENBLAS/NUMEXPR/VECLIB` to 8 and (when torch is already loaded) `torch.set_num_threads(8)` / inter-op 8; `EGVQG_NUM_THREADS=<n>` overrides. `scripts/sanity.py` imports `egvqg` before numpy so the env applies; README documents the cap and the import-order requirement.
- Verification: `env -u OMP_NUM_THREADS ... python -c "import egvqg; import torch"` -> `torch.get_num_threads()==8`; the 120-file load benchmark through the package is 1.2 s; `scripts/sanity.py` passes; `pytest` 132 passed; `python -m egvqg --help` and a `diag-template --dry-run` run succeed.
- Artifacts: `src/egvqg/__init__.py`, `scripts/sanity.py`, `README.md`.
- Not changed: the 8-core quota itself is set by the container/scheduler and cannot be raised in-process; root `train.py` (untracked GPU memory stress script) was left untouched.

### 2026-11-15: Measurement + PVS generation route (handoff)

**Environment and process**
- Rebuilt the environment with `uv sync --extra local-inference`: Python 3.11.11, torch 2.11.0+cu128, transformers 5.22.2, CUDA available on A800-80GB. `pytest -o addopts=""` **134 passed**; `scripts/sanity.py` passes.
- Fixed the CPU oversubscription (container reports 64 CPUs, quota 8): `local_vision`/package cap torch and BLAS at 8; `probing.load_candidates` loads hidden states with 8 threads (260 s → ~4 s). Recorded a model-path relocation in `model_identity_relocations.jsonl` (only `model_fingerprint.path` differs; all file hashes and shard sizes identical).
- Added `detail-capture` (teacher-forced per-head `o_proj` inputs, per-step hidden states, pooled image/text states; cosine ≥ 0.9998 vs generation states) and ran it for pilot/confirm (480 each, 4.1–4.2 GB each).

**Main data and labels**
- `qwen3vl8b-cand-expand-v2`: 253 parents, 2024 candidates, 695 unique questions, distances and local answerer complete. Cascade evaluation 1846 new calls; `candidate-labels` G1 passed via mixed direction 26.2% (mixed redundancy 11.3%).
- Consensus label repair: expand 457 candidates (1102 new calls, 798 cache hits); confirm 440/440 valid (276 new calls). Inter-judge agreement 0.923 (expand) / 0.913 (confirm); the legacy cascade label has non-redundant precision 0.79, so the consensus rule is now primary (two judges agree; disagreements undetermined; arbiter only provisional).
- Label-cost selector: one-judge shortlist has recall 1.0 by construction and precision 0.70–1.0; saves 20–43% of API calls. See `outputs/qwen3vl8b-content-mechanism-v1/screen_selector.json`.
- Stitching filter: 7/105 dual-necessary candidates are concatenated questions in the pooled schema analysis; they are excluded from the synergistic count.

**Diagnosis / negative results**
- Schema/PID: S is schema-determined; probability 60/60, exchange 19/26, classification 6/26, naming/explanation/comparison ≈0 (`schema_table.md`).
- Cross-modal attention at question_end predicts material-level S (0.979 redundancy, 0.954 direction) but within-skill 0.665; attention bias raises image attention 0.63→0.86 and S 19.2→21.7% (CI includes 0), R 25→30%. Contrastive decoding unstable at λ≥1; adaptive constraint no effect.
- Material variants (19 paired): hint 23.9%, lecture+hint 27.3%, lecture-only 18.6%, unique-v1 11.8% (R 56.9%); no variant significantly beats hint. The four-way report is `outputs/qwen3vl8b-cand-lecturehint-unique-pilot19-v1/four_way.json`.
- SAQG v1 positive 0.370; v2 0.465 with 0 stitching; plain lecture+hint 0.600. Forced SAQG negatives 0.077.

**PVS results**
- Portfolio (plain + SAQG v2) + verify/select: ScienceQA 60 coverage 56.7%, delivered 70.6%, oracle 43.3%; MultiModalQA 60 coverage 88.3%, delivered 77.4%, oracle 81.7% (SAQG v2 alone 85.0/82.4/80.0). Report: `outputs/qwen3vl8b-pvs60-v1/cross_dataset_report.md`.
- Clean isolation (selection `gpt-6.1-sol`; jury `gpt-6-luna` + `DeepSeek-V4.1-Flash`): 50-material budget subset, 100 candidates → 37 dual candidates → 21 delivered (42% coverage); jury 16 dual / 1 not_dual / 4 undetermined; delivered precision 76.2% conservative (16/21) and 94.1% among decided (16/17); cost 63 new sol + 89 jury calls. Report: `outputs/pvs-clean-1000-v1/subset50_report.md`.
- The full 1000-material generation was completed (2000 SAQG candidates), but the associated 1000-material sol selection was aborted for budget; 914 sol calls are sunk and not part of the clean result. The 50-material subset is the current clean estimate.

**Code / prompts added**
- `src/egvqg/detail_capture.py`, `src/egvqg/saqg.py`, `src/egvqg/pvs.py`; prompts `saqg-extract-v1`, `saqg-generate-v1/v2`, `pvs-prefilter-v1/v2/v3`, `pvs-stitch-v1`; CLI `detail-capture`; tests for `identity_matches`, `select_parents`, `detail_positions`; label-repair concurrency/`select_parents` fix.
- Verification: 134 offline tests pass; `scripts/sanity.py` passes; `python -m egvqg --help` and `detail-capture --help` work.

**Unresolved / next**
- Material ceiling: oracle coverage 43.3% (ScienceQA 60) and 81.7% (MMQA 60); the method cannot create synergy where no complementary evidence exists.
- 4 DeepSeek undetermined in the clean subset; rerun or mark consistently.
- Prefilter v3 recall 6/10, false-positive 0/10; pass-all is the current main run.
- Budgeted next step: extend clean PVS to 100–200 materials, audit delivered questions, then write Measurement / Diagnosis / Solution.

### 2026-11-15: Advisor-style review and first checks of the planning-stage pivot

**Review received (not an experiment).** Three structural gaps were raised: (i) “oracle” is only the candidate-pool upper bound, not the material ceiling; (ii) the necessity side (witness pairs) has almost no independent validation, while all positive results sit on that side; (iii) the headline `1.5% → 32%` mixes protocol changes, `k`, selection and different denominators. Additional points: precision without recall, candidate-pooled CIs, schema/skill collinearity, stitching sensitivity, and AI labels being reliability rather than validity.

**Checks actually run (offline unless noted).**
- Clean-50 Wilson intervals recomputed: precision `16/21 = [0.549, 0.894]`; among decided `16/17 = [0.730, 0.990]`; material yield `16/50 = [0.208, 0.458]`. The review's numbers are correct; the current point estimates should be reported with these intervals.
- Schema/skill collinearity is stronger than “high”: in the 1000-material pool, `schema_hint` is perfectly nested in `skill` — 0/84 skills contain more than one schema; `probability` appears only in the two Punnett skills, `exchange` only in “Trade and specialization”. Any material-level schema effect is therefore indistinguishable from a skill effect.
- Stitching branch: it was reached 37/100 (clean 50) and 144/400 (200) times, and returned `False` every time. “Never positive” is correct; “never triggered” is not. Sensitivity of `pvs-stitch-v1` remains unverified.
- Protocol base rates (material-level any-dual, `k=8`, plain-v2 unless noted): pilot `14/54 = 25.9%`; confirm `11/56 = 19.6%`; expand repair sample `23/229 = 10.0%`; clean-50 with selection `16/50 = 32.0%`. The old full baseline is `19/1257 = 1.5%`. These belong in one reconciliation table; most of the headline jump is protocol + best-of-k + selection, not SAQG.
- Propensity within-skill check (the gating item for the pivot): overall prompt_end layer-30 `R² 0.568` (input `0.241`) reproduces, but **within-skill centered `R²` falls to 0.048** (input `−0.255`); within-skill pairwise concordance `0.559` (9W/1T/7L, 17 pairs; binomial p=0.80, within-skill permutation p=0.20). Between-skill variance share is 0.686. The signal is almost entirely between-skill, like P5.
- Binary “material has any dual candidate” target: overall prompt_end AUC `0.832` vs input `0.634` (diff CI `[0.047, 0.370]`), but **within-skill pairwise AUC is 0.083 (1/12 pairs)**; the apparent affordance signal is between-skill and anti-predicts within skill. This is the main caution for the “model knows but does not ask” story on ScienceQA.
- MMQA check: continuous material propensity is not predictable (`R² < 0`; 41/52 materials are all-redundant, so variance is too low). Binary “any dual” at prompt_end is predictable: probe AUC `0.665` vs input `0.299`, diff CI `[0.056, 0.619]` (52 materials, 11 positive). This is the best positive lead for an affordance/planning signal, but MMQA has no skill labels and the interval is wide.

**Implication.** The proposed planning-stage pivot is well motivated by the negative decoder results, but its G2 (“affordance probe within-skill”) is **not met on ScienceQA** with the current 56-material confirmation set. The defensible claim for now is “the decision collapses to a skill/template prior”, not “the model knows the material is dual-able and chooses not to ask”. Before committing, run G1 (position curve) with within-skill and n-gram baselines; for that, expand candidates need per-token detail capture (the existing expand run only has prompt_end/early/question_end states). MMQA `any-dual` is the most promising place to test the stronger affordance claim.

**Side status.** The 200-material clean jury is still running; at 38/81 it shows an elevated DeepSeek undetermined rate (11/38), so a retry/repair pass will be needed before those numbers are usable. The 200 selection itself is complete and cache-only (`144` dual candidates, `81` delivered).

### 2026-11-15: MMQA-first G3, within-material G1 and any-dual probe

**T1 G3 (generator–answerer gap).** Ran the local Qwen under TI/T/I/0 with `solve-v1`; reference = the same Qwen's TI answer (self-consistency, not human gold). MMQA: 42 dual questions (one per material with a consensus dual candidate) + 30 matched redundant controls. 17/42 dual questions had a TI answer; 25/42 did not (24 abstain + 1 null). Among the 17 TI-answerable duals: T `0/17`, I `4/17`, 0 `0/17`; 13/17 match neither single modality. Controls: TI 24/24, T `6/24 = 25.0%`, I `10/24 = 41.7%`, 0 `2/24 = 8.3%`. The gap exists only on the 17 answerable duals; 60% of dual questions are unanswerable even with full material. ScienceQA was run as a side check (84 dual questions, 59 TI-answerable; T 25.4%, I 16.9%) but is not the target. Artifacts: `outputs/g3-generator-answerer-v1/` (`table.md`, `answers.jsonl`, `mmqa_report.md`).

**T2 within-material G1 (MMQA SAQG).** Added selective `detail-capture` (`--material-ids-file`, `--prompts-file`) and built a teacher-forced SAQG replay with the per-material `saqg-generate-v2` prompt. SAQG: 294 decided candidates, 59 materials, **31 discordant materials**, 151 candidates, 161 within-material pairs. Material-demeaned features, grouped CV by material, layer 30, n-gram baseline of the first k question tokens + length. At p=0 the prefill state is identical within a material (AUC 0.5 structurally). p=1..8 probe AUC `0.38–0.55`, n-gram `0.42–0.62`; every probe−baseline CI includes 0 (several favour the n-gram baseline). **No within-material planning signal.** The control (plain) MMQA run has only 4 discordant materials, so the SAQG replay is the relevant set. Artifact: `outputs/qwen3vl8b-mmqa-saqg-detail-v1/t2_within_qwen3vl8b-mmqa-saqg-replay-v1.json`.

**T3 MMQA any-dual probe (downgraded).** 52 materials, 11 positive. L2 logistic input baseline (passage length + image geometry/format + coarse entity-type heuristic) vs prompt_end layer 30. Probe AUC `0.665`, 95% CI `[0.489, 0.825]`; probe−0.5 CI `[-0.011, 0.325]`; probe beats confounds (`[0.026, 0.571]`) but **is not above chance**. Per the pre-set rule the lead is **downgraded**. Artifact: `outputs/g3-generator-answerer-v1/t3_mmqa_confounds.json`.

**T4 / S0.** The 200-material jury finished at 81/81: original `37 dual / 12 not_dual / 32 undetermined`; all undetermined were DeepSeek. First retry (60 s, 2 attempts) resolved 1 item. S0 retry used the same judge with **300 s timeout, concurrency 2, max 3 attempts** and resolved 14 more (8 to dual, 6 to not_dual); final `45 dual / 19 not_dual / 17 undetermined`. The 17 remaining are recorded as `judge_transport_failure:TimeoutError:billing_unknown` (16) or `judge_invalid_api_response` (1), reported separately and **not** replaced by another judge. 200-material consequences: coverage `81/200 = 40.5%`; conservative precision `45/81 = 55.6%` (Wilson `[0.447, 0.659]`); among decided `45/64 = 70.3%` (`[0.582, 0.801]`); material yield `45/200 = 22.5%`. 17/81 transport failures remain, so the 200 precision is still less clean than the 50-material result (`76.2%` / `94.1%`). Artifacts: `outputs/pvs-clean-200-v1/jury.jsonl`, `jury_retry.jsonl`, `jury_retry_s0.jsonl`, `jury_final_s0.json`.

**T5.** Added the “schema ⊂ skill” diagnostic to the Paper outline above (0/84 skills with more than one schema; probability only two Punnett skills, exchange only Trade and specialization).

### 2026-11-15: Week-1 execution (S0–S4)

**S0 judge infrastructure.** Same DeepSeek judge, 300 s timeout, concurrency 2, max 3 attempts: 31 → 17 transport failures; 14 resolved (8 dual, 6 not_dual). Final 200-material jury `45 dual / 19 not_dual / 17 undetermined`; conservative precision `45/81 = 55.6%` `[0.447, 0.659]`, decided `45/64 = 70.3%` `[0.582, 0.801]`, material yield `45/200 = 22.5%`. The 17 remaining are transport failures, reported separately.

**S1 G3b (MMQA strong reference).** There is no stored jury reference answer; used `gpt-6-luna` TI with `solve-v1` as a strong reference. Of the 25 duals Qwen could not answer under TI, luna answered **19/25**; the 6 failures are candidate invalid/ambiguous/external-knowledge cases. Against the luna reference, Qwen's own answers are much weaker than the self-reference table suggested: TI `10/34 = 29.4%`, T `0/34`, I `3/34 = 8.8%`, 0 `0/34`. This favours the “8B capability limit” explanation over “can integrate but chooses not to ask”. The 4 image-only Qwen matches are exported for human review (`outputs/g3-generator-answerer-v1/s1_image_only_review.json`). Artifact: `s1_g3b.md`.

**S2 M4 (extractive vs derived answer target).** 592 decided MMQA candidates (plain 298 + SAQG 294); 470 have an answer, 466 typed (400 extractive / 66 derived), 122 no answer, 4 unclear. All-candidate association is strong: derived S rate `65.2%` vs extractive `16.5%`, AUC `0.665`. **Within-material fails:** SAQG 15 discordant materials / 66 pairs, AUC `0.523`, cluster bootstrap `[0.421, 0.648]`; plain 3 materials / 16 pairs, AUC `0.500`. The lower bound is below 0.5, so M4 does **not** meet the pre-set criterion. Per the pre-set rule: go with the **measurement paper**, not the mechanism paper. Artifact: `outputs/g3-generator-answerer-v1/s2_m4.md`; 50-item human check `s2_answer_type_review.csv`.

**S3 synthetic evaluator validation.** 296 synthetic materials, 8 categories × 37, PIL tables + programmatic ground truth; validity `review-question-faithfulness-v1` (luna) + screen `ensemble-sufficiency-v1` (luna + terra consensus). Against synergistic dual (derived + stitching): precision **1.000**, recall **98.6%** (73/74). Extractive/both-sufficient/common-knowledge: 0 false positives. Presupposition: validity rejects 37/37, without it the screen would be 37/37 false-positive. **Bridging is a validity-stage blind spot:** screen calls 35/37 dual, but validity declares 13 invalid and 24 undetermined, so final detected bridging duals = 0/37. Artifact: `outputs/s3-synthetic-v1/report.md`.

**S4 human review pack.** 199 rows: PVS delivered 63 (clean-50 + MMQA), selection-rejected 50, baseline single-sample 50, undetermined 36. Columns include question/material/jury label/flags/witness and blank `witness_valid`/`answer_correct`/`notes`. Artifact: `outputs/review-v1/review_sheet.csv` and `.jsonl`.

**Decision.** ScienceQA frozen (diagnosis + human-review stratum only). M4 failed within-material; the active plan is the measurement paper: three-valued certification + human/synthetic validation + diagnosis + PVS + systematic negative mechanism section.

### 2026-11-15: Week-1 composition-gap mechanism route (W1-A/B/C)

- Read the method sections of Biran et al. (Hopping Too Late, EMNLP 2024) and Compose and Fuse (ICLR 2026) before designing the run; registered section 7.12 in `docs/implementation_plan.md` before any synthetic answer data.
- Built `data/processed/synthetic-composition-v1` with `synthetic-build`: 60 materials per category (`derived`, `bridging`, `stitching`), each image hashed into the material record, each material carries the full question and two single-hop questions with programmatic answers. A separate count variant (`synthetic-composition-count-v1`) asks "How many items count?" to isolate conditional filtering from summation.
- Design iterations (all retained; final numbers use `-v4`): (i) the first bridging image put a row label in the left column, so "the left column" was not a number; fixed before formal use; (ii) the three-value derived design always made exactly two items count, so the text rule was redundant; switched to four values; (iii) thresholds could equal a value, leaving an "at least" boundary ambiguity; switched to strictly-between thresholds.
- W1-A sum (8B, TI, direct): derived full `0.533`, text hop `1.000`, image hop `1.000`, deficit `+0.467` `[+0.350, +0.583]`, both hops correct `60`, both-correct-full-wrong `28`; error modes `included_all_values` 5, `included_extra_value` 4, `omitted_qualifying_value` 2, `mixed_subset` 6, `other_numeric` 11. Bridging `1.000/1.000/1.000`, deficit 0; stitching `1.000/1.000/1.000`, deficit 0.
- W1-A count isolation (8B): derived full `0.850`, deficit `+0.150` `[+0.067, +0.250]`, 9/60 both-correct-full-wrong; all 9 expected count 2 but answered 3, including the largest below-threshold value. `outputs/synth-comp-count-qwen3vl8b-v2/`.
- Stop-loss: derived CI excludes 0 in both sum and count, bridging CI includes 0 in both → mechanism route continues. The narrow supported claim is conditional filtering of visual values by a textual threshold, not all serial composition.
- W1-B two-step (8B, sum): derived full `0.533 → 0.850` (`+0.317`), mixed deficit (two-step full minus direct hop product) `-0.150`. The two-step image-hop accuracy collapses to `0.017` because the prompt solves the composition during extraction; report the direct gap, two-step gap, and mixed gap. `outputs/synth-comp-qwen3vl8b-twostep-v4/`, `outputs/synth-comp-prompt-compare-v4/prompt_comparison.md`.
- W1-C (2B/4B/8B): sum derived deficits `1.000 / 0.617 / 0.467`; count derived deficits `0.400 / 0.150 / 0.150`; dual-generation rates (4 sampled questions per material, fixed grammar labels) `0.125 / 0.338 / 0.333`; No correlation statistic (three points, no repeated seeds). Per category, 4B/8B dual rates are 0.000/0.013 on derived, 1.000 on bridging, 0.000 on stitching; the generator usually copies the exact threshold into Q, making the question image-only (8B 1.000/1.000, 4B 0.988/1.000, 2B 0.508/0.742 on derived/stitching). 2B sum failures are dominated by returning the list of values instead of the sum, so its deficit is mostly aggregation. `outputs/synth-comp-cross-size-v4/comparison.md`, `outputs/synth-comp-count-cross-size-v2/comparison.md`.
- Combined clean 8B failure set for week 2: `outputs/mechanism-w1-v1/mechanism_samples_8b.jsonl` (9 count + 11 sum set-selection = 20 rows).
- Code/prompts: `src/egvqg/synthetic_composition.py`; CLI `synthetic-build`, `composition-answer`, `composition-generate`, `composition-report`, `composition-compare`, `composition-compare-prompts`; prompts `composition-answer-v1`, `composition-two-step-v1`; optional sampling in `local_vision.QwenVLBackend`; tests `tests/test_synthetic_composition.py` (4 passed).
- Verification: 139 offline tests pass; no API/judge calls were made.
- Unresolved: bridging at ceiling; only 9 clean 8B count failures; synthetic materials are not the certified complementary-evidence questions; the generation link is not established (the generator copies the threshold into Q, making derived/stitching questions image-only; the 4B/8B bridging rates are at ceiling); one material seed and one image template; two-step hop artifact.
- Next: week 2 attention knockout (image→last token, text→last token, per layer) and Patchscopes/logit lens on the 20 clean failures, then back-patching / early-layer attention softening and, only if a repair works, apply it to generation. Extend the count set to 150 materials per category if more clean failures are needed. The measurement paper remains the fallback.

### 2026-11-15: R1-R4 confirmation, inline analysis, interventions, external validity

**R1 confirmation (new seed/range/template; only this run's numbers are the confirmation).** Built `synthetic-composition-confirm-{sum,count}-v1` with seed `20261116`, values `5..40`, 60 per category, categories derived/bridging/stitching plus the new `interval` template (text gives a two-sided inclusive interval; image gives four values; text hop asks both limits; image hop asks the four values).
- Sum endpoint: derived full `0.333`, text hop `1.000`, image hop `0.867`, deficit `+0.533` `[+0.367, +0.683]`; interval full `0.383`, image hop `0.967`, deficit `+0.583` `[+0.450, +0.717]`; bridging and stitching 0.
- Count endpoint: derived full `0.867`, deficit `+0.000` `[-0.117, +0.117]` (7/52 both-hops-correct failures; the v4 single-threshold count effect did not replicate); interval full `0.583`, deficit `+0.383` `[+0.250, +0.517]` (24/58 failures).
- All 24 interval-count failures are off-by-one: expected 1 or 2, answered 2 or 3, including a value just outside the lower or upper bound. This is the cleanest confirmed conditional-filtering failure.
- Two-step: sum derived `0.333 -> 0.817` (mixed deficit `-0.050`), sum interval `0.383 -> 0.617` (mixed `-0.350`), count derived `0.867 -> 1.000` (mixed `+0.133`), count interval `0.583 -> 0.967` (mixed `0.000`); the image hop collapses to `0.017` under two-step in all cells.
- Artifacts: `outputs/synth-confirm-{sum,count}-8b-{direct,twostep}-v1/`, `outputs/mechanism-w1-confirm-v1/confirmation_report.md`, `mechanism_samples_confirm_8b.jsonl` (39 clean failures), `data/processed/synthetic-composition-confirm-{sum,count}-v1/`.
- Appendix ledger v1-v4 with reasons and numbers: `outputs/mechanism-w1-v1/appendix_versions.md`. Spearman was removed from the W1-C comparison and report.

**R2 inline analysis.** Added `inline_analysis.py` + CLI `inline-report`. Inline = exact material number (digits or number words), rule phrase, or (MMQA) a 5-word lexical span. Synthetic text redundancy uses the programmatic `single_image` proxy; MMQA uses the existing consensus `text_suff`/`both_suff`.
- Synthetic: 8B/4B inline 98.8-100% on derived/stitching and 100% on bridging (the name `alpha`); 2B inline 50.8% derived, 74.2% stitching, 37.5% bridging.
- Bridging shows the distinction: inline `alpha` questions are dual (text needed), non-inline image-value questions are text-redundant. Inlining a defined name is not the same as inlining the decisive rule.
- MMQA SAQG: inline any `78/360 = 21.7%`; **image** redundancy among inline `31/61 = 50.8%` vs non-inline `60/233 = 25.8%` (Wilson `[0.386, 0.629]` vs `[0.206, 0.317]`); text redundancy `8/61 = 13.1%` vs `49/233 = 21.0%`. *(Field name corrected 2026-11-15: `text_suff`/`both_suff` was read as "text redundant" but means text sufficiency = image redundancy; see the E1 entry.)*
- MMQA control: inline any `116/360 = 32.2%`; **image** redundancy inline `88/99 = 88.9%` vs non-inline `154/199 = 77.4%` (intervals overlap slightly); text redundancy `24/99 = 24.2%` vs `7/199 = 3.5%`. *(Same correction; corrected numbers in `outputs/inline-r2b-v1/`.)*
- Artifacts: `outputs/inline-r2-v1/`, report `outputs/mechanism-w1-v1/r2_inline.md`.

**R3 interventions.** Added `plain-refer-v2` (reference prompt; keep proper names), numeric-token suppression in `QwenVLBackend` (logits processor over digit tokens in the material text), and `local-validity-v1` + `local-validity`/`validity-report` CLI. Local validity is a local Qwen AI review, not the official cascade evaluator.
- Original: dual `0.333`; local validity `0.988` (668/720 parsed); inline 1.000 on derived/stitching.
- Reference prompt v2: dual `1.000`; local validity `0.983` (716/720); inline 0.000 on derived/stitching. Per category validity bridging `0.983`, derived `1.000`, stitching `0.966`.
- Numeric suppression: dual `0.406`; local validity `0.921` (671/720); inline number-word rate `0.808` derived, `0.975` stitching. The model bypasses digit suppression by writing number words; some questions are truncated.
- Prompt pilot: `plain-refer-v1` reached dual 1.000 but validity 0.704 (bridging 0.146) by adding spurious "according to the indicated threshold" clauses; v2 repaired it and is the reported intervention.
- Artifacts: `outputs/synth-comp-qwen3vl8b-{refer-v2,masknum-v1}/`, `outputs/validity-{orig,refer-v2,mask}-8b-v1/`, `outputs/inline-r3-v1/`, report `outputs/mechanism-w1-v1/r3_interventions.md`.

**R4 external validity.** Annotated the 25 MMQA duals with no Qwen TI answer (`s1_g3b.md` section a) using luna as a first pass plus a manual operational review; AI annotation, not human.
- Operation types: `text_rule_image_value` 6/25 = 24.0% (Wilson `[0.115, 0.434]`), `cross_modal_comparison` 7, `other_or_invalid` 6, `text_fact_image_entity` 3, `image_attribute_text_entity` 2, `temporal_or_numeric` 1.
- Strict luna coder: 2/25 = 8.0% (Wilson `[0.022, 0.250]`); the 4 disputed items mention a "rule" but the rule is not the decisive operation.
- Reading: even under the broader count, only ~1/4 of the hard MMQA duals are text-rule-on-image-value; the synthetic finding is a controlled probe, not the dominant MMQA failure.
- Artifacts: `outputs/mechanism-w1-v1/r4_mmqa_25_operations.csv`, `r4_external_validity.md`, `outputs/r4-mmqa-operation-annotation-v2/`.

**Verification.** 145 offline tests pass; `scripts/sanity.py` passes; no judge calls were used for R1/R2/R3; R4 used 3 luna calls (one earlier truncated call is sunk). No head-level ablation was run (R3 rule).

**Next.** R1 changes the mechanism target: the robust deficit is the two-sided interval rule, not the single-threshold count. Week-2 mechanism work should use `mechanism_samples_confirm_8b.jsonl` (24 interval-count + 7 derived-count + 8 interval-sum set errors). R3(a) needs the official evaluator and an MMQA run before it can be called a generation fix; R3(b) should not be pursued at the token level. R4 limits external validity and argues for reporting the synthetic result as a controlled probe.

### 2026-11-15: Generalization test — MMQA refer prompt vs interval mechanism

**A. MMQA feasibility of `plain-refer-v2` (subagent run).** Same 60 MMQA materials and 6 questions per material as the existing control; new generation with `plain-refer-v2`; labels via the same two-judge consensus path and an official cascade cross-check.
- Control vs refer-v2, two-judge consensus: dual `49/360 = 0.136` vs `26/360 = 0.072`; **image** redundant `242/298 = 0.812` vs `281/314 = 0.895`; text redundant `31/298 = 0.104` vs `39/314 = 0.124`; inline any `116/360 = 0.322` vs `168/360 = 0.467` (difference `+0.144 [+0.044, +0.250]`). **Correction (2026-11-15, E1 entry):** this entry originally reported the `text_suff`/`both_suff` share as "text redundancy"; under the formal definition (`docs/necessity_formal_argument.md` §9) that share is text *sufficiency*, i.e. image redundancy. Recomputed without new calls in `outputs/mmqa-refer-v2-compare-v2/`.
- Official cascade: dual `23/360 = 0.064` vs `10/360 = 0.028`; answerability `295/360 = 0.819` vs `308/360 = 0.856`; text redundant among answerable `0.136` vs `0.110`; image redundant among answerable `0.885` vs `0.945`.
- Conclusion: **does not generalize**. The prompt was a synthetic threshold-copying fix; on MMQA it produces long text-fact lookups, raises inline copying and image redundancy (text alone answers), and slightly lowers dual. Do not adopt it for MMQA. Cost: 1,387 new API calls / 1.507M tokens; 13 judge-side technical failures remain undetermined.
- Artifacts: `outputs/mmqa-refer-v2-v1/report.md`, `comparison.json`, `spot_check_ai_unreviewed.md`; code `src/egvqg/material_generation.py`, `src/egvqg/mmqa_refer.py`; CLI `material-generate`, `mmqa-refer-compare`.

**B. Interval mechanism probe (subagent run).** 24 confirmed interval-count failures + 8 correct controls, Qwen3-VL-8B; behavioral ablations + logit lens; 79.6 s GPU, peak 16.9 GiB, no API calls.
- Text ablation (numbers removed): 24/24 answers change to 4 (all values); disjoint interval `1..2`: 24/24 answer 0; shifted interval: 24/24 within ±1 of the new true count, 14/24 exact. Image-only: 24/24 answer 4, **0/24** reproduce the published wrong count.
- Logit lens: failures' wrong count dominates 24/24, becomes preferred between layers 22–26, final mean P(wrong) `0.959` vs P(correct) `0.041`; controls resolve to correct (P `0.980`), crossover layers 29–35.
- Conclusion: the text interval is used; the failure is **boundary precision** (the effective interval is ~1 unit wider), not image dominance or late overwriting. Attention knockout is not the high-value next step; a per-bound comparison prompt is. Two-step already raises this cell `0.583 → 0.967`.
- Artifacts: `outputs/interval-mechanism-v1/report.md`, `results.jsonl`, `behavior.jsonl`, `logit_lens.jsonl`; code `src/egvqg/interval_mechanism.py`; CLI `interval-mechanism`.

**Decision.** The generation-side prompt branch is stopped on MMQA; the interval mechanism is a confirmed synthetic probe but not externally dominant (R4). If one more experiment is run, it should be a per-bound comparison prompt on the interval set (cheap) and, only if it works, on the 6/25 MMQA rule-on-value subset. The measurement paper remains the fallback. Verification: 153 offline tests pass; sanity passes.

### 2026-11-15: E1 redundancy-source decomposition + correction of the inverted redundancy field

**Why.** The new paper plan needs to know whether single-modality redundancy comes from *leakage* (the generator copies the unnecessary side's content into the question) or from *single-source anchoring* (nothing is copied; the question asks something the retained side already contains). That decides whether a copy-circuit intervention or repaired self-verification carries the method, and it is nearly free because every label already exists.

**Definition bug found and fixed first.** `docs/necessity_formal_argument.md` §9 defines text redundancy as "the text brings no unique information" = image alone suffices = verdict `image_suff`/`both_suff`. `inline_analysis.py` and `mmqa_refer.py` had been reading `text_suff`/`both_suff` as "text redundant", i.e. they measured text *sufficiency* (image redundancy). Fixed in `src/egvqg/inline_analysis.py` (both quantities now reported), `src/egvqg/mmqa_refer.py` (`text_redundant`, new `image_redundancy*`, cascade mapping `text_suff` field → text redundant only when it equals `image_suff`/`both_suff`), and the two tests that encoded the old mapping. Consequences: `outputs/inline-r2-v1/` and `outputs/mmqa-refer-v2-compare-v1/` are stale; recomputed without new API calls as `outputs/inline-r2b-v1/` and `outputs/mmqa-refer-v2-compare-v2/`. The "does not generalize" conclusion stands; the mechanism sentence changes from "raises text redundancy" to "raises inline copying (+14.4 pp) and image redundancy (text alone suffices, +8.3 pp)".

**Method.** New `src/egvqg/redundancy_source.py` + CLI `redundancy-source`. A unit is a unique `(material_id, question)` pair; `undetermined`/non-redundant units are excluded from the split but kept in the row count. Two prompts (`redundancy-source-text-v1`, `redundancy-source-image-v1`) ask one question per redundant side whether the question *restates* information only that side holds (`leak`), only refers to / asks about it (`anchor`), or is degenerate/outside knowledge (`other`); question-level label is `leak` if any redundant side leaks, else `anchor` if all anchor, else `other`. A programmatic screen (material number or 5-word span for the text side; a number absent from the text as an OCR-free proxy for the image side) runs alongside. Command:

```bash
.venv/bin/python -m egvqg redundancy-source --run-id e1-redsrc-v1 \
  --dataset sq-expand=scienceqa:outputs/qwen3vl8b-cand-expand-v2/labels_qwen3vl8b-cand-expand-v2-eval.jsonl:data/processed/scienceqa/lecture-hint-image-v1/materials.jsonl \
  --dataset sq-confirm=scienceqa:outputs/qwen3vl8b-cand-confirm-v2/labels_consensus_full.jsonl:data/processed/scienceqa/lecture-hint-image-v1/materials.jsonl \
  --dataset mm-control=mmqa:outputs/vqag-mmqa-pvs60-control-label-v1/candidates.jsonl:outputs/mmqa-refer-v2-v1/materials.jsonl \
  --dataset mm-saqg=mmqa:outputs/vqag-mmqa-pvs60-saqg-label-v1/candidates.jsonl:outputs/mmqa-refer-v2-v1/materials.jsonl \
  --dataset mm-refer-v2=mmqa:outputs/vqag-mmqa-refer-v2-label-v1/candidates.jsonl:outputs/mmqa-refer-v2-v1/materials.jsonl \
  --dataset synth-v4=synthetic:outputs/synth-comp-qwen3vl8b-direct-v4/generated.jsonl:data/processed/synthetic-composition-v1/materials.jsonl \
  --max-calls 1600 --concurrency 8
```

**Results (judge `gpt-6-luna`, AI labels, not human review).**

| dataset | redundant units | leak share | anchor | screen leak | image redundancy | text redundancy |
|---|---:|---:|---:|---:|---:|---:|
| synth-v4 8B | 167 | **1.000** [0.978, 1.000] | 0.000 | 1.000 | 0.000 | 0.723 |
| sq-expand | 552 | 0.130 [0.105, 0.161] | 0.870 | 0.049 | 0.553 | 0.592 |
| sq-confirm | 116 | 0.129 [0.080, 0.202] | 0.871 | 0.086 | 0.464 | 0.700 |
| mm-saqg | 107 | 0.131 [0.080, 0.208] | 0.869 | 0.084 | 0.309 | 0.199 |
| mm-control | 103 | 0.087 [0.047, 0.158] | 0.913 | 0.136 | 0.806 | 0.113 |
| mm-refer-v2 | 122 | 0.041 [0.018, 0.092] | 0.959 | 0.131 | 0.835 | 0.137 |

- Leakage is a synthetic-set phenomenon (100% there) and a minority everywhere else (4–13%); the dominant real-data mode is **anchoring**, and on MMQA it is text anchoring (image redundancy 0.81–0.84, i.e. the text alone answers).
- Leak concentrates where both sides are redundant: mm-control 7/11, mm-refer-v2 5/13, mm-saqg 5/18, sq-expand 65/235, sq-confirm 12/47. Inside image-redundant units it is negligible (mm-control 2/89).
- The screen is a weak proxy in both directions (under-flags sq-expand 0.049 vs 0.130; over-flags mm-control 0.136 vs 0.087 and mm-refer-v2 0.131 vs 0.041), so it is kept only as a diagnostic and all quoted shares use judge labels.
- The judge never used `other` (0 across 1,000+ units); the 50-row stratified manual sheet (`outputs/e1-redsrc-v1/manual_sheet.jsonl`, 2 rows per dataset × direction × judge label, `human_source` empty) exists to test that, and is **pending human review**. Until it is filled every share is AI-labelled.
- Reading all 28 MMQA `leak` labels against the material text (`outputs/e1-redsrc-v1/source_addendum_manual_read.md`, AI operational read) splits them into 14 description-elicitation repeats (the question restates a descriptive span and asks for the entity's name, so the text still holds the answer — these are anchors), 12 genuine content copies and 2 unclear. Strict MMQA leak is therefore **at most 12/332 = 3.6%**, versus 4.1–13.1% under the judge's literal reading; both are far below the 0.20 gate, so the decision does not depend on the strictness. The genuine copies are of two kinds: quoting a visual detail (makes the image unnecessary) or quoting a text fact (makes the text unnecessary).

**Decision.** The pre-registered gate (MMQA leak share < 0.20) fires: **E4 (copy-circuit intervention) is demoted to a synthetic-set analysis** and the method section rests on **E3 (repaired self-verification)**. The real-data failure mode is anchoring, which a copy-circuit fix cannot address.

**Artifacts.** `outputs/e1-redsrc-v1/{source_report.md, source_summary.json, source_units.jsonl, manual_sheet.jsonl}`, screen-only `outputs/e1-redsrc-screen-v1/` (leak upper/lower bounds), smoke `outputs/e1-redsrc-smoke-synth-v1/` (32 calls, 167/167 leak, copied span "value of at least 6"). Cost: 593 new API calls, 898 cache hits, 1.151M tokens.

**Verification.** `tests/test_redundancy_source.py` (direction mapping, leak-wins combination, dedup + undetermined handling, screen behaviour) and the corrected `tests/test_mmqa_refer.py` pass; rerunning the E1 command with `--max-calls 0` reproduced the report from 1,491 cache hits and 0 new calls.

**Unresolved / next.** Manual review of the 50-row E1 sheet and of the existing 199-row PVS pack is still outstanding; the judge's total absence of `other` labels needs checking, and the strict-leak reading (`source_addendum_manual_read.md`) should be confirmed by a human before the "3.6%" figure is quoted in the paper. Next analysis: E2 (composition gap 2×2 + psychometric curve + patching) on A800, and E3's repaired-self-verification setup (behavioural two-step first, patching second).

### 2026-11-15: E2a composition grid (2×2 modality placement, psychometric curve, aggregation)

**Question.** Is the measured composition gap a cross-modal binding deficit, or does it come from the number/aggregation step? Pre-registered contingency: if the text-only cell (TT) shows the same gap, the mechanism wording changes to numeric aggregation/precision and the E3 main line (repaired self-verification) is untouched.

**Design.** `src/egvqg/composition_grid.py` + CLI `grid-build`, `grid-answer`, `grid-report`. One material = four item values plus a threshold strictly between the two smallest values (so exactly three items always count). The same values, threshold and questions are rendered in four cells: `TT` both facts in text, `TI` rule in text + values in an image table, `IT` values in text + rule in an image strip, `II` both in one image (rule strip stacked above the table). Five question types per cell: `full` (which items count, the conditional judgment), `count`, `sum`, `rule`, `values`. Scoring is programmatic (set of A–D letters, exact numeric sum), so no judge calls; `grid-report` re-scores from raw output, so scoring fixes never require regeneration. Prompt `grid-answer-v1` is modality-neutral ("the context text and/or the image") so text-only cells are legitimate.

**Commands.**
```bash
E=.venv/bin/python; M=/hpc2hdd/home/fye374/models/Qwen3-VL-8B-Instruct
$E -m egvqg grid-build --n-materials 100 --output-dir data/processed/composition-grid-v2
setsid nohup bash -c 'flock -w 7200 /tmp/egvqg_gpu0.lock -c "CUDA_VISIBLE_DEVICES=0 '"$E"' -m egvqg grid-answer --run-id grid-e2a-v3 --materials data/processed/composition-grid-v2/materials.jsonl --local-model-path '"$M"' --device cuda:0 --max-batch 8"' > tmp/grid-e2a-v3.log 2>&1 < /dev/null &
$E -m egvqg grid-report --run-id grid-e2a-v3 --materials data/processed/composition-grid-v2/materials.jsonl
```

**Results (Qwen3-VL-8B, greedy, 100 materials × 4 cells × 5 question types = 2,000 answers, `outputs/grid-e2a-v3/`).**

| cell | which items count | count | sum | rule hop | value hop |
|---|---:|---:|---:|---:|---:|
| TT | 0.840 (84/100) | 0.930 | 0.580 (58/100) | 1.000 | 1.000 |
| TI | 0.810 (81/100) | 0.940 | 0.320 (32/100) | 1.000 | 1.000 |
| IT | 0.890 (89/100) | 0.570 | 0.430 (43/100) | 1.000 | 1.000 |
| II | 0.990 (99/100) | 0.910 | 0.230 (23/100) | 1.000 | 1.000 |

Paired material bootstrap (10,000): `cross − same` **−0.065 [−0.120, −0.010]**; `TI − TT` −0.030 [−0.140, +0.080]; `IT − TT` +0.050 [−0.030, +0.130]; `II − TT` +0.150 [+0.080, +0.230]; `IT − TI` +0.080 [−0.020, +0.190]. The n=40 replication (`grid-e2a-v2`) gives the same ordering with wider intervals (`cross − same` −0.062 [−0.150, +0.025]; `II − TT` +0.150 [+0.025, +0.275]).

**Psychometric curve (per-item conditional judgment, signed distance from the threshold).** Below the threshold: ≤−4, −3, −2 all **1.000**, −1 0.957 (7 extras). Above: +1 0.950 (9 misses), +2 0.914 (12), +3 0.917 (10), ≥+4 0.987 (10). Errors are omissions of counting items, not boundary reversals; by position, item A 0.950 → D 0.988, and the error shape is dominated by `missed_one_A` (17 of 47 full-question failures). At the cell level the dip sits in the text-list cells (TT +2 0.829, TI +3 0.800) while II is at or near 1.000 at +1/+3/≥+4.

**Sum question.** 0.230–0.580 in every cell despite 100% hop accuracy: `arithmetic_error` 157, `listed_values_not_summed` 82, correct 156, unparsed 5. The order is TT 0.580 > IT 0.430 > TI 0.320 > II 0.230, i.e. reading the values from an image costs, but the aggregation step itself is the dominant loss in all four cells.

**Reading.** (1) Conditional composition ("which items count") is **not** a cross-modal binding problem: accuracy is 0.81–0.99, and the cell that holds both facts in one image is the **best**, not the worst. The small significant cross−same difference is carried by TI (rule text + values image) being lowest and by text-list item-dropping, not by an interaction between modalities. (2) The large deficit is the **multi-step aggregation** (filter then sum), which fails in the all-text cell too, and is worst when the values must be read from an image. (3) The psychometric view shows a shallow near-threshold dip on the counting side only, which does not reproduce the sharp boundary-width defect seen in the interval probe; the interval defect therefore remains specific to interval endpoints, not to threshold comparisons in general. (4) This lands on the pre-registered contingency: the mechanism wording becomes numeric aggregation/precision over read values, the "cross-modal binding" wording is withdrawn, and E3 (repaired self-verification) remains the paper's main line.

**Artifacts.** `outputs/grid-e2a-v3/{answers.jsonl, grid_summary.json, grid_report.md}` (main), `outputs/grid-e2a-v2/` (n=40 replication), `outputs/grid-smoke-v2/` (48-answer design smoke), materials `data/processed/composition-grid-v2/` (400 rows, 100 materials), `data/processed/composition-grid-v1/` (160 rows, 40 materials), prompts `prompts/grid-answer-v1.txt`, module `src/egvqg/composition_grid.py`, tests `tests/test_composition_grid.py`.

**Verification.** `grid-report` re-scores from raw output and regenerates both runs; `.venv/bin/python -m pytest tests/test_composition_grid.py` (5 passed) and the full offline suite (163 passed) pass; `scripts/sanity.py` passes. No API/judge calls: the whole E2a block is local GPU plus deterministic scoring (≈2 min per 640–2,000 answers on the A800).

**Unresolved / next.** (i) `count` is not usable as a dependent variable — it mixes format violations (letters returned instead of a number) with genuine off-by-one answers that flip between cells for the same material. (ii) The grid's `sum` accuracy (0.23–0.58) is much lower than the synthetic `derived` endpoint (0.733), because five question types are interleaved and the prompt differs; the two are not directly comparable and the grid is the stronger test. (iii) The mechanism probe still owes the numeric-representation comparison (decode the same number from image tokens vs text tokens) and the causal patching of a value representation from a text cell into the TI cell; with the grid built, the TI↔TT pair is the natural patching target. (iv) E3 setup is next: four-condition self-verification driven by `experiments.answering_input`-style isolation, behavioural two-step as tier 1, patching as tier 2, against API PVS.

### 2026-11-15: E3 repaired self-verification (local answerer as verifier, two pools, API PVS on the same questions)

**Question.** Can the local 8B replace the API screens as a verifier — select iff TI correct and both T and I wrong — and does repairing its composition ability raise the recall of that verifier? Pre-registered prediction (paper plan): unrepaired recall is very low because genuine duals fail under TI and are discarded; repair should lift recall toward API PVS.

**Design.** `src/egvqg/self_verification.py` + CLI `self-verify-run`, `self-verify-report`, plus `pvs-select` (API PVS selection stage only, now wired into the CLI). Selection rule `TI correct and T wrong and I wrong`, mirroring API PVS with the local answerer in place of the two API screens. Two repair tiers: `direct` (`composition-answer-v1`) and `two-step` (`composition-two-step-v1`, state the rule, state the values, combine). Each question is answered under all three conditions with the removed side not passed at all (`T` gets no image, `I` gets empty text). Gold is computed from the synthetic material (exact designed question first, then threshold/which-items/count/sum/item-value rules); questions without computable gold are dropped and counted, never guessed — coverage was 720/720 pool rows and 60/60 designed rows, 0 dropped. Scoring is programmatic, so no judge calls. `self-verify-report` also scores an API PVS selection file on the same questions and reports per-label recall.

**Two pools, because the prediction only has content where the gap exists.** Pool A = the 8B's own generations (`outputs/synth-comp-qwen3vl8b-direct-v4/generated.jsonl`, 720 rows → 231 unique questions: 64 `dual_bridging` + 167 `single_image`). Pool B = the 60 designed aggregation duals (`evaluation.full` of the `derived` materials, "What is the total value of the items that count?"), i.e. the question type where the composition gap actually lives (TI 0.517); their own single sides cannot answer (T 0.000, I 0.033). Pool B questions are template-designed, not model-generated — stated as such.

**Commands.**
```bash
E=.venv/bin/python; M=/hpc2hdd/home/fye374/models/Qwen3-VL-8B-Instruct
setsid nohup bash -c 'flock -w 7200 /tmp/egvqg_gpu0.lock -c "CUDA_VISIBLE_DEVICES=0 '"$E"' -m egvqg self-verify-run --run-id self-verify-synth-v1 --pool outputs/synth-comp-qwen3vl8b-direct-v4/generated.jsonl --materials data/processed/synthetic-composition-v1/materials.jsonl --local-model-path '"$M"' --device cuda:0 --max-batch 8"' > tmp/self-verify-synth-v1.log 2>&1 < /dev/null &
$E -m egvqg self-verify-run --run-id self-verify-derived-v1 --pool outputs/self-verify-derived-pool-v1/pool.jsonl --materials data/processed/synthetic-composition-v1/materials.jsonl --local-model-path $M --device cuda:0 --max-batch 8
$E -m egvqg pvs-select --run-id pvs-synth-v1 --candidates outputs/pvs-synth-v1/candidates_unique.jsonl --materials data/processed/synthetic-composition-v1/materials.jsonl --workers 8
$E -m egvqg pvs-select --run-id pvs-synth-derived-v1 --candidates outputs/pvs-synth-derived-v1/candidates.jsonl --materials data/processed/synthetic-composition-v1/materials.jsonl --workers 8
$E -m egvqg self-verify-report --run-id self-verify-synth-v1 --pvs-selection outputs/pvs-synth-v1/selection.jsonl
$E -m egvqg self-verify-report --run-id self-verify-derived-v1 --pvs-selection outputs/pvs-synth-derived-v1/selection.jsonl
```

**Pool A results (231 unique questions, 64 genuine duals, `outputs/self-verify-synth-v1/`).**

| tier | TI | T | I | recall | precision | FPR on non-duals |
|---|---:|---:|---:|---|---|---|
| direct | 0.904 (651/720) | 0.011 | 0.575 | **1.000 (64/64)** | 0.831 (64/77) | 0.078 (13/167) |
| two-step | 0.894 (644/720) | 0.000 | 0.582 | **1.000 (64/64)** | 0.889 (64/72) | 0.048 (8/167) |
| API PVS | — | — | — | 1.000 (64/64) | 1.000 (64/64) | 0.000 (0/167) |

**Pool B results (60 designed aggregation duals, `outputs/self-verify-derived-v1/`).**

| tier | TI | T | I | recall | precision |
|---|---:|---:|---:|---|---|
| direct | 0.517 (31/60) | 0.000 | 0.033 | **0.483 (29/60)** | 1.000 |
| two-step | 0.850 (51/60) | 0.000 | 0.000 | **0.850 (51/60)** | 1.000 |
| API PVS | — | — | — | 0.983 (59/60, 1 undetermined) | 1.000 |

Paired material bootstrap of the tier contrast on pool B: `two-step − direct recall` **+0.367 [+0.217, +0.517]** (10,000 resamples, 60 materials). On pool A the contrast is +0.000 [0.000, 0.000] because direct recall is already at ceiling.

**Reading.** (1) The verifier's recall factorises as P(TI correct) × P(both single sides wrong); the second factor is ~1 for genuine duals, so **recall is bounded by the answerer's TI accuracy**. (2) On the 8B's own generated duals (all bridging) TI is already 0.904, so the pre-registered "unrepaired recall is very low" **does not hold on this pool**: recall is 1.000 without repair, and repair buys precision instead (0.831 → 0.889, FPR 0.078 → 0.048) by making the single-side conditions more reliably correct. (3) On aggregation duals, where the composition gap is, recall is 0.483 and two-step repair raises TI 0.517 → 0.850 and recall 0.483 → 0.850, i.e. **repair is what makes self-verification work exactly where verification is hard**. (4) API PVS is near-perfect on both pools (1.000/1.000; 0.983/1.000) at 336 and 140 API calls; the local verifier matches its recall on pool A and reaches 0.85 on pool B at zero API calls but 3 local answers per question.

**Cost.** Pool A local: 4,320 answers (720 pool rows × 2 tiers × 3 conditions) ≈ 20 min on the A800. Pool B local: 360 answers ≈ 2 min. API PVS: 336 calls (pool A, 231 questions) and 140 calls (pool B, 60 questions), no judge/jury calls. Prompt `answer-equivalence-v1` was written for real-data reference scoring but is unused so far; the real-data tier (MMQA/ScienceQA) still needs a reference-answer budget and is not part of this entry.

**Artifacts.** `outputs/self-verify-synth-v1/{answers.jsonl, plan.jsonl, dropped.jsonl, manifest.json, self_verify_summary.json, self_verify_report.md}`, `outputs/self-verify-derived-v1/`, `outputs/self-verify-derived-pool-v1/pool.jsonl`, `outputs/pvs-synth-v1/`, `outputs/pvs-synth-derived-v1/`, `outputs/self-verify-smoke-v1/` (60-slot smoke), module `src/egvqg/self_verification.py`, tests `tests/test_self_verification.py`.

**Verification.** `.venv/bin/python -m pytest` → **170 passed**; `scripts/sanity.py` → SANITY PASSED; both reports regenerate from `answers.jsonl` via `self-verify-report`.

**Unresolved / next.** (i) Pool B uses designed template questions; the model does not generate that dual type, so "the generated pool contains no hard duals" is itself a finding to state, not to hide. (ii) The patching tier (mechanistic repair) is still owed; only the behavioural two-step tier exists. (iii) The real-data tier needs reference answers (MMQA has 17 usable strong-model references); without them only agreement with a strong model can be measured, not correctness. (iv) Cost can be cut by early-stopping after a wrong TI answer (would save ~30% on pool A, ~48% on pool B) — not implemented, since all three conditions are currently logged for analysis.

### 2026-11-15: E3 repair ladder, local signal as selector, early stop, MMQA real-data pilot

**Question.** If the composition gap is an *aggregation* defect (E2a/E2b), can a *structural* repair of the same 8B turn it into a verifier that matches the API at zero API cost — and where does that stop working?

**Design (fixed before the runs).** Tiers, all selecting "TI correct and both T and I wrong": `direct` (one answer from the joined material), `two-step` (condition → values → combine), `delegated` (model returns the qualifying values and the rule, the harness sums), `delegated-compare` (model returns rule + threshold + full item→value table, the harness does the comparison and the aggregation), `per-item` (one row per item with its value and a `counts` flag; the harness aggregates the model's own flags). Gold for pool B is `sum(values >= threshold)`; TI failure reasons are categorised by searching the below-threshold / qualifying subsets whose sum equals the error (`included_threshold_minus_one`, `included_other_below`, `dropped_qualifying`, `other`). Early stop: answer TI for everything, then the single sides only for TI-correct groups; a wrong TI is a decided rejection, not an undetermined item.

**Commands.**

```bash
E=".venv/bin/python -m egvqg"; M=/hpc2hdd/home/fye374/models/Qwen3-VL-8B-Instruct
# pool B ladder
$E self-verify-run --run-id self-verify-derived-v2 --pool outputs/self-verify-derived-pool-v1/pool.jsonl \
  --materials data/processed/synthetic-composition-v1/materials.jsonl --local-model-path $M --device cuda:0 \
  --tiers direct,two-step,delegated --max-batch 8
$E self-verify-run --run-id self-verify-derived-v3 ... --tiers direct,two-step,delegated,per-item
$E self-verify-run --run-id self-verify-derived-compare-v1 ... --tiers delegated-compare
# pool A with the repaired tiers, scored against the API PVS selection
$E self-verify-run --run-id self-verify-synth-v2 --pool outputs/synth-comp-qwen3vl8b-direct-v4/generated.jsonl \
  --materials data/processed/synthetic-composition-v1/materials.jsonl --local-model-path $M --device cuda:0 \
  --tiers direct,two-step,per-item --max-batch 8
$E self-verify-report --run-id self-verify-synth-v2 --pvs-selection outputs/pvs-synth-v1/selection.jsonl
# early stop
$E self-verify-run --run-id self-verify-es-check-v1 ... --tiers direct --early-stop
# MMQA real tier
$E self-verify-real-run --run-id self-verify-mmqa-v1 \
  --candidates outputs/vqag-mmqa-pvs60-control-label-v1/candidates.jsonl \
  --materials outputs/mmqa-refer-v2-v1/materials.jsonl --local-model-path $M --device cuda:0 \
  --sample-per-material 2 --reference-model gpt-6-luna
$E self-verify-real-report --run-id self-verify-mmqa-v1 --equivalence-model gpt-5.6-terra
```

**Pool B ladder (60 designed aggregation duals).**

| tier | TI | T | I | recall | precision |
|---|---:|---:|---:|---|---|
| direct | 0.517 | 0.000 | 0.033 | 0.483 (29/60) | 1.000 |
| delegated | 0.783 | 0.000 | 0.000 | 0.783 (47/60) | 1.000 |
| two-step | 0.850 | 0.000 | 0.000 | 0.850 (51/60) | 1.000 |
| delegated-compare | 0.933 | 0.000 | 0.000 | 0.933 (56/60) | 1.000 |
| **per-item** | **1.000 (60/60)** | 0.017 | 0.000 | **0.983 (59/60)** | 1.000 |
| API PVS | — | — | — | 0.983 | 1.000 |

Paired material bootstraps (10,000): `per-item − direct` **+0.500 [+0.367, +0.633]**; `two-step − per-item` −0.133 [−0.233, −0.050]; `direct − delegated` −0.300 [−0.433, −0.167].

**Pool A (`self-verify-synth-v2`, 6,480 answers, 231 unique questions).**

| tier | TI | T | I | recall | precision | FPR (non-duals) |
|---|---:|---:|---:|---|---|---|
| direct | 0.900 | 0.011 | 0.581 | 1.000 (64/64) | 0.865 | 0.060 (10/167) |
| two-step | 0.887 | 0.000 | 0.582 | 1.000 | 0.914 | 0.036 (6/167) |
| per-item | 1.000 | 0.060 | 0.665 | 1.000 | **0.985 (64/65)** | **0.006 (1/167)** |
| API PVS | — | — | — | 1.000 | 1.000 | 0.000 |

**Reading.** The repair ladder is monotone in how much of the *comparison* is made explicit per item rather than summarised: 0.483 → 0.783 → 0.850 → 0.933 → 0.983. The mechanism is visible in the failure taxonomy — 10 of delegated's 13 errors include the value one below the threshold and then compute the sum correctly, i.e. "at least N" is applied as "at least N−1", which is the question-level form of the E2b width-≈-1 defect. Making the comparison per item removes it **without moving the arithmetic out of the model** (the model still sets each `counts` flag; the harness only adds up flags the model chose). So per-item is the mechanism-targeted tier, not a delegation trick, and it reaches API PVS recall at zero API cost. On the model's own easy duals the same tier converts into precision (FPR 0.060 → 0.006) because the single-side conditions become reliably correct — which is what a verifier's false positives come from.

**Coverage (new readout).** With 1.28 unique questions per material, a selected dual appears for 60/180 materials (33%) under every tier, and mean selected is 0.36–0.41. The verifier is no longer the binding constraint: generator dual supply is. `_sampling_curve` reports precision and dual yield as a function of k samples per material so a best-of-n budget can be priced; the pool needs a `sample_index` field to use it.

**Early stop.** `--early-stop` on pool B direct used 124 answers instead of 180 (−31%), skipping 56 single-side answers, with recall 0.500 (30/60) vs 0.483 (29/60) without (answer sampling noise; TI 0.533 vs 0.517). Expected answers per question is 1 + 2·P(TI correct), so the saving shrinks as TI accuracy rises and vanishes for per-item (TI 1.000).

**MMQA real-data pilot (`self-verify-mmqa-v1`).** 120 sampled pool rows collapse to **89 unique (material, question) pairs**, of which 7 are consensus duals and 6 undetermined. Reference answers from `gpt-6-luna` (`solve-v1`, full material), equivalence judged by `gpt-5.6-terra` (`answer-equivalence-v1`). Local: TI 0.843 (70/83), T 0.795 (66/83), I 0.120 (10/83); **recall 0.143 (1/7) [0.026, 0.513]**, precision 0.250 (1/4), FPR 0.039 (3/76). The rule itself behaves correctly — for `text_suff` questions T is acceptable 50/54, which *validates* the label — but 5 of the 7 duals are refused under every condition (open-ended "what is the significance/connection" questions) and 2 have no reference, so the local signal cannot certify MMQA duals at this n. One material was blocked by provider image content policy (400, now visible in the ledger `error_detail`).

**Code.** `self_verification.py`: five tiers with `_combine`, `score_gold_{delegated,per_item,compare}` (4-tuple with `detail`), `_values_list`/`_per_item_table`/`_read_threshold`, `_ti_error_taxonomy`, `coverage` + `_sampling_curve`, `early_stop`, `prepare_real_pool` / `run_real_verification` / `report_real_verification`; prompts `composition-delegated-v1`, `composition-per-item-v1`, `composition-delegated-compare-v1`, `solve-v1`, `answer-equivalence-v1`; CLI `self-verify-run --early-stop`, `self-verify-real-run`, `self-verify-real-report`, `pvs-select`. `providers.py`: `JudgeHTTPError.body`, non-2xx bodies captured (2048 B), ledger `failed` events carry `error_detail`.

**Cost.** Pool B: 540 + 720 + 180 local answers (~10 min total). Pool A: 6,480 answers (~20 min). Early-stop check: 124. MMQA: **408 API requests** (120 references, 288 equivalence, 4 failed) plus 360 local answers.

**Artifacts.** `outputs/self-verify-{derived-v2,derived-v3,derived-compare-v1,synth-v2,es-check-v1,mmqa-v1}/`, `outputs/bon-synth-v1/materials.jsonl` (60 materials × 20 per category, prepared for the sampling curve, unrun).

**Verification.** `.venv/bin/python -m pytest` → **175 passed**; `scripts/sanity.py` → SANITY PASSED; every report regenerates from `answers.jsonl` + `references.jsonl` + `equivalence.jsonl` without new calls.

**Unresolved / next.** (i) The mechanistic tier (patching the numeric representation / back-patching) is still owed — the ladder is behavioural. (ii) MMQA transfer needs a larger pool (≥4–6 unique questions per material, ~250 questions) and probably a different selection rule for open-ended duals, since the 8B refuses them rather than answering them wrong; do not report the 1/7 as a transfer failure in the paper. (iii) The best-of-n sampling curve is prepared but unrun; it is the next test of the supply bottleneck. (iv) per-item is aggregation-shaped by construction; routing (per-item for filter/sum questions, two-step otherwise) is the natural next refinement and is not implemented.

**Best-of-n supply check (`bon-synth-v1`, appended after the entry above).**

```bash
$E composition-generate --run-id bon-synth-v1 --materials outputs/bon-synth-v1/materials.jsonl \
  --local-model-path $M --device cuda:0 --n-questions 8 --max-batch 8
$E self-verify-run --run-id bon-synth-verify-v1 --pool outputs/bon-synth-v1/generated.jsonl \
  --materials outputs/bon-synth-v1/materials.jsonl --local-model-path $M --device cuda:0 \
  --tiers direct,two-step --max-batch 8
$E self-verify-report --run-id bon-synth-verify-v1
```

480 generations (60 materials × 8, temperature 0.7, `plain-v2`) collapse to **86 unique questions**, 1–3 distinct per material: 320 samples are `single_image` (63 unique) and 160 `dual_bridging` (23 unique, all from the 20 `bridging` materials). Dual-bearing materials are **20/60 at k=1 and 20/60 at k=8**; `derived` and `stitching` produce no dual at any k. Direct tier: recall 1.000 (23/23), precision 0.793 (23/29), FPR 0.095; two-step: precision 0.958 (23/24), FPR 0.016. The per-k curve (questions 59→86, selected 24→29, raw dual rate 0.339→0.267, precision 0.833→0.793) shows the marginal yield of extra samples is ~0 while the cost is linear. Interpretation: the generator is **mode-collapsed on one dual shape**, so coverage is a generation-side property, not a sampling budget. This also corrects an implicit assumption in the earlier "supply is the bottleneck" bullet — it is supply, but specifically a *shape* problem that best-of-n cannot address. A reporting bug in `_render` (`_fmt` assumed Wilson dicts for the curve's float fields) was fixed; the summary JSON was already correct.


### 2026-11-16: E4 copy-source circuit — attribution, head ablation, attention blocking (negative, with mechanism)

**Question.** E1 put 100% of the synthetic threshold-set redundancy on `leak`, and the best-of-n run showed the generator is mode-collapsed on that leak. Is the copied threshold a localizable copy circuit that can be attenuated at generation time (the planned E4), or is it a property of the question schema?

**Design (fixed before the runs).** Materials: `data/processed/synthetic-composition-v1/materials.jsonl`, category `derived` (text holds only the rule "Only items with value at least N count.", the four values are only in the image, so any numeral in the question can only come from the text) with `plain-v2`, greedy 24 tokens.
1. `copy-attribute`: attention mass from the first generated numeral token to the context numeral token, all 36 layers × 32 heads, ranked by materials-in-top-10.
2. `copy-ablate`: zero the head output slice of the top 8 / top 16 / a random 8 / all heads on decode steps only (prefill untouched), then generate normally.
3. `copy-mask`: additive attention bias (−1e4) that forbids generated positions from attending to (a) the context numeral tokens, (b) the whole text span.
Readouts: copy rate (`contains_number(text)`), threshold mention, `classify_generated_question` label, unique questions, and — added before reading the numbers — the faithful-threshold rate (first numeral equals the material threshold).

**Commands.**
```bash
E=".venv/bin/python -m egvqg"; M=/hpc2hdd/home/fye374/models/Qwen3-VL-8B-Instruct
CUDA_VISIBLE_DEVICES=0 $E copy-attribute --run-id copy-heads-v1 --materials data/processed/synthetic-composition-v1/materials.jsonl --local-model-path $M --device cuda:0 --n-materials 20
CUDA_VISIBLE_DEVICES=0 $E copy-ablate --run-id copy-ablate-all-v1 --local-model-path $M --device cuda:0 --n-materials 20
CUDA_VISIBLE_DEVICES=0 $E copy-mask --run-id copy-mask-v1 --local-model-path $M --device cuda:0 --n-materials 20 --samples 4
CUDA_VISIBLE_DEVICES=0 $E copy-mask --run-id copy-mask-span-v1 --local-model-path $M --device cuda:0 --n-materials 20 --samples 4 --conditions block_text_span
$E copy-mask-report --run-id copy-mask-e4-v1 --sources copy-mask-v1,copy-mask-span-v1 --materials data/processed/synthetic-composition-v1/materials.jsonl
```

**Results.**

| condition | copy rate | threshold mention | faithful threshold | dual | single-image |
|---|---:|---:|---:|---:|---:|
| control (no bias) | 1.000 | 1.000 | **1.000** | 0.000 | 1.000 |
| head ablation top8/top16/random8 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| block context numeral | **0.000** | 1.000 | **0.000** | 0.000 | 1.000 |
| block whole text span | 0.000 | **0.000** | 0.000 | 0.000 | 0.963 |

Attribution: 20/20 materials copy a context numeral; top heads by materials-in-top-10 are L02H31 19/20 (mean top-10 mass 0.658), L29H11 17 (0.811), L28H23 15 (0.743), L28H00 14 (0.580), L20H08 13 (0.602), L25H30 11 (0.814), L24H23 11 (0.696), L23H04 10 (0.826). Ablating all 32 heads × 36 layers produces garbage, and the hook fires 432 times (36 layers × 12 steps), so the hooks are live; top-8/top-16/random-8 ablation changes nothing, i.e. the copy is redundant across heads. Masking: of the 80 blocked-numeral questions, the first numeral is 10 in 44, 5 in 12, 9 in 8, 13 in 8, 12 in 4, 16 in 3, 15 in 1 — never the true threshold (0/80); the whole-span block instead yields "Which is the largest value?" and the model continues into its own reasoning ("To determine the largest value in the table…"), so the degeneration is visible in the output.

**Reading.** The copied threshold is not a removable copy circuit: blocking the source token makes the model invent a threshold (and the question is then *unfaithful* to the rule, so validity would fall), and blocking the whole text span makes it abandon the criterion and ask an image-only question. Every attention-level intervention on the redundant side therefore produces a *single-modality* question, because a dual question must use the rule without quoting it — that is a generation-schema property. This mechanistically justifies the pre-registered E1 decision (E4 demoted): the working lever is the question schema (the refer prompt reaches dual 1.000 on this set, already recorded in R3), and the selection lever is repaired local self-verification (E3). Both the earlier null results were caused by a real bug in `_KeyBlock.bias()` (one query row per decode step made the mask return `None`); the bug is fixed, verified against a single-token prompt (one numeral token at position 172), and a regression test now covers the decode-step shape.

**Artifacts.** `outputs/copy-heads-v1/` (attribution.jsonl, heads.json, report), `outputs/copy-ablate-smoke-v1/`, `outputs/copy-ablate-all-v1/`, `outputs/copy-mask-v1/`, `outputs/copy-mask-span-v1/`, merged report `outputs/copy-mask-e4-v1/mask_report.md`. Module `src/egvqg/copy_circuit.py`, CLI `copy-attribute`, `copy-ablate`, `copy-mask`, `copy-mask-report`; tests `tests/test_copy_circuit.py` (4 tests). Offline: **179 passed**, `scripts/sanity.py` SANITY PASSED.

**Cost.** Local GPU only, no API calls: 20 attributions + 80 ablations × 4 conditions + 160 + 80 generations, each well under a minute per material batch.

**Unresolved / next.** (i) The masked-threshold run uses 20 materials and 4 samples per condition; the faithfulness claim (0/80) is strong enough to stop, but a 60-material replication would be needed before publishing a rate. (ii) Head attribution is correlational; path patching on the top heads is still the causal test, though the ablation null makes a positive causal result unlikely — deprioritized. (iii) The remaining generation-side option is schema control (a prompt or decoding constraint that requires using the rule without restating it, e.g. question templates with the criterion abstracted); this is the only lever the E4 negative leaves open and it should be tested on the same face-valid readouts before any real-data work. (iv) E4 is now a documented null, so the paper's method section rests on framed generation plus E3 self-verification; the mechanism section carries "the leak is load-bearing for the schema".

### 2026-11-16: local text-answerability screen vs API PVS on 149 real MMQA questions

**Question.** E1 showed that on MMQA the redundancy is *single-source anchoring*, not leakage (leak ≤ 13.1% lenient, ≤ 3.6% strict). The verifier-side question that follows is therefore not "does the model combine both sides" but "**can the passage alone already answer this question?**". Can a purely local screen of that form (no reference answers, no judge, no API) match or beat the API PVS selector on real data?

**Design (fixed before the run).** Rule `reject_if_text_answers`: select a question iff the local Qwen3-VL-8B **abstains or fails to answer** under the text-only condition (`T`) and returns an answer under the full material (`TI`). Secondary rule `reject_if_text_answers_no_ti`: same, ignoring the TI side (so a question cannot be dropped for an unparsable full answer). Universe: the 149 unique `(material, question)` pairs of `outputs/vqag-mmqa-pvs60-control-label-v1/candidates.jsonl` (360 rows over 60 materials, deduplicated after `prepare_real_pool` was changed to dedupe and to treat `sample_per_material<=0` as all rows). Gold = existing two-judge consensus `label_consensus.verdict == 'dual_necessary'` (21 duals, base rate 0.141). Comparison: API PVS `pvs-select` (`gpt-6.1-sol`) run on exactly the same 149 questions. Readouts fixed in advance: recall, precision, false-positive rate on the 128 non-duals, material-clustered bootstrap of precision and of lift over the base rate, coverage, and the local/PVS set intersections.

**Commands.**
```bash
E=".venv/bin/python -m egvqg"
M=/hpc2hdd/home/fye374/models/Qwen3-VL-8B-Instruct
CUDA_VISIBLE_DEVICES=0 $E answerability-audit --run-id mmqa-answerability-v1 \
  --candidates outputs/vqag-mmqa-pvs60-control-label-v1/candidates.jsonl \
  --materials outputs/mmqa-refer-v2-v1/materials.jsonl \
  --local-model-path $M --device cuda:0 --max-batch 8
$E pvs-select --run-id pvs-mmqa-aa-v1 \
  --candidates outputs/mmqa-answerability-v1/candidates_unique.jsonl \
  --materials outputs/mmqa-refer-v2-v1/materials.jsonl --workers 6
$E answerability-report --run-id mmqa-answerability-v1 --bootstrap 10000 \
  --pvs-selection outputs/pvs-mmqa-aa-v1/selection.jsonl
```

**Results.** 298 local answers (TI + T), ~5 min GPU, **0 API calls**. Local text-answered rate 0.846 (126/149) — consistent with the label mix and an independent sanity check on it. PVS verdicts on the same 149: text_suff 111, image_suff 6, stitching 15, dual 16, undetermined 1.

| selector | selected | recall | precision | FPR |
|---|---:|---:|---:|---:|
| local text-answerability (0 API calls) | 23 | **0.667 (14/21)** | 0.609 (14/23) | 0.070 (9/128) |
| API PVS (`gpt-6.1-sol`, 149 calls) | 16 | 0.476 (10/21) | 0.625 (10/16) | 0.047 (6/128) |
| local ∧ API | 10 | 0.381 (8/21) | **0.800 (8/10)** | 0.016 (2/128) |
| local ∨ API | 29 | **0.762 (16/21)** | 0.552 (16/29) | 0.102 (13/128) |

Material-clustered bootstrap (10,000): local precision **0.609 [+0.385, +0.750]**, lift over the 0.141 base rate **+0.468 [+0.296, +0.611]**. Coverage: 11/60 materials have at least one selection (18.3% [10.6, 29.9]); 0.383 selections per question. The two rules coincide because TI almost always answers on this data.

**Reading.** The local screen **matches PVS precision (0.609 vs 0.625) and beats it on recall (0.667 vs 0.476)** at zero API cost. The two selectors are complementary rather than redundant: the intersection is the high-confidence set (precision 0.800, FPR 0.016), and the union pushes recall to 0.762 at precision 0.552. PVS's extra rejections are 15 `stitching` cases the local screen keeps — PVS penalizes questions that look like two independent sub-answers, the local screen only asks whether the passage suffices; PVS's 6 `image_suff` rejections overlap the local ones. This is the first real-data method signal that is both cheap and at or above the API baseline, and it follows directly from the E1 diagnosis: MMQA needs an *anchoring* filter, not a composition repair. It also explains why the repaired self-verifier was the wrong instrument for MMQA (recall 1/7 pilot: the 8B refuses open-ended MMQA duals, so a "can both sides answer" rule cannot certify them, while a "can text alone answer" rule can reject the anchored ones).

**Artifacts.** `outputs/mmqa-answerability-v1/` (`pool.jsonl`, `answers.jsonl`, `audit_summary.json`, `audit_report.md`, `candidates_unique.jsonl`), `outputs/pvs-mmqa-aa-v1/` (`selection.jsonl`, `records.jsonl`, ledger). Module `src/egvqg/self_verification.py` (`run_answerability_audit`, `report_answerability_audit`, `_answered`, `_material_bootstrap_answerability`, `_interval`); CLI `answerability-audit`, `answerability-report`. Test `test_answerability_audit_scores_text_screen`.

**Verification.** `.venv/bin/python -m pytest` → **180 passed in 12.65s**; `scripts/sanity.py` → SANITY PASSED.

**Unresolved / next.** (i) One dataset, 60 materials, AI-consensus labels; the recall interval is wide at n=21 (0.667 [0.454, 0.828] Wilson) and precision is material-clustered, so a ScienceQA replication is the next check — it is already implemented (`answerability-audit` takes any candidates/materials pair with `label_consensus.verdict`; ScienceQA consensus candidates exist in `outputs/qwen3vl8b-cand-confirm-v2/`). (ii) A hybrid rule is the current best operating point (local ∨ PVS recall 0.762) but costs 149 PVS calls; a cheaper hybrid that only sends the local screen's *rejections* to PVS would need its own measurement. (iii) The `TI` side of the rule is nearly free of information here (TI abstains on only 8/149 and none of those overlap the 23 text-abstentions, so both rules select the same 23), so the rule is effectively "text-only fails"; the paper should state it that way and not overclaim a dual-condition test. (iv) No human review of the 14 true positives or the 9 false positives yet.

### 2026-11-16 (addendum): two-sided answerability rules and the ScienceQA direction check

**Why.** The first audit scored only `reject_if_text_answers` on MMQA. That rule is a *text-anchoring* detector, so it can only be as sharp as the dataset's redundancy direction allows. The report now computes three rules from the same answers (`--conditions TI,T,I`), which makes the audit a test of the E1 direction claim rather than a single selector: `reject_if_text_answers` (reject when the passage alone answers), `reject_if_image_answers` (reject when the image alone answers), `reject_if_either_side_answers` (both single sides must fail).

**Commands.**
```bash
# I condition added to both existing pools (resume adds only the missing rows)
for R in "sq-answerability-v1 outputs/sq-answerability-v1/candidates_src.jsonl data/processed/scienceqa/lecture-hint-image-v1/materials.jsonl" \
         "mmqa-answerability-v1 outputs/vqag-mmqa-pvs60-control-label-v1/candidates.jsonl outputs/mmqa-refer-v2-v1/materials.jsonl"; do
  set -- $R
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m egvqg answerability-audit --run-id $1 --candidates $2 --materials $3 \
    --local-model-path /hpc2hdd/home/fye374/models/Qwen3-VL-8B-Instruct --device cuda:0 --max-batch 8
done
.venv/bin/python -m egvqg answerability-report --run-id mmqa-answerability-v1 --bootstrap 10000 \
  --pvs-selection outputs/pvs-mmqa-aa-v1/selection.jsonl
.venv/bin/python -m egvqg answerability-report --run-id sq-answerability-v1 --bootstrap 10000
```
The ScienceQA pool was derived from `outputs/qwen3vl8b-cand-confirm-v2/labels_consensus_full.jsonl` (480 rows → 173 unique `(material, question)` pairs; 24 duals, 1 conflicting duplicate) by mapping `decided/redundant/direction` to the formal verdict (`image`→`image_suff`, `text`→`text_suff`, `both`→`both_suff`, `redundant=0`→`dual_necessary`), written to `outputs/sq-answerability-v1/candidates_src.jsonl`.

**Results (same-run, same-rules).**

| dataset | rule | selected | recall | precision | FPR | T answers | I answers |
|---|---|---:|---:|---:|---:|---:|---:|
| MMQA (149 q, 21 duals) | text | 23 | 0.667 | 0.609 | 0.070 | 0.846 | 0.195 |
| MMQA | image | 120 | 0.905 | 0.158 | 0.789 |  |  |
| MMQA | either-side | 15 | 0.571 | **0.800** | 0.023 |  |  |
| MMQA | API PVS | 16 | 0.476 | 0.625 | 0.047 |  |  |
| ScienceQA (173 q, 24 duals) | text | 105 | 0.708 | 0.162 | 0.591 | 0.393 | 0.590 |
| ScienceQA | image | 71 | 0.750 | 0.254 | 0.356 |  |  |
| ScienceQA | either-side | 30 | 0.458 | **0.367** | 0.128 |  |  |

**Reading.** The side that answers identifies the dataset's redundancy direction: MMQA is text-anchored (0.846 vs 0.195), ScienceQA confirm is image-anchored (0.590 vs 0.393), exactly as E1's leak/decomposition table implies. Consequently the text-only rule is the right instrument for MMQA and the wrong one for ScienceQA (precision 0.609 → 0.162, FPR 0.070 → 0.591), while the two-sided rule is portable: 0.800 precision on MMQA (15 selections) and 0.367 on ScienceQA (30 selections) — above the 0.139 base rate in both, with a ScienceQA material-clustered lift of +0.228 [+0.036, +0.455]. Against API PVS on the same MMQA questions the local gate's point estimates are better on both axes (precision 0.800 vs 0.625, recall 0.571 vs 0.476) but **not significantly** (paired bootstrap: precision [−0.019, +0.475], recall [−0.222, +0.421]), so the defensible claim is "equal-or-better point estimates at zero API cost and 3 local answers per question", not a win.

**Verification.** `pytest` → **181 passed in 12.80s**; `scripts/sanity.py` → SANITY PASSED. New test `test_answerability_audit_two_sided_rule_and_api_comparison` covers the two-sided rule, the per-side answered rates and `_selector_comparison`.

**Unresolved.** (i) ScienceQA labels here are the consensus repair set mapped to verdicts, not a fresh human check; the 1.6% conflicting duplicate was resolved by taking the first occurrence. (ii) No ScienceQA PVS run on the same 173 questions, so the ScienceQA comparison is against the base rate only. (iii) The either-side rule loses recall on both datasets (0.571 / 0.458) because a weak side that fails anyway is not evidence; a length- or schema-conditioned rule may recover it. (iv) n=21–24 duals keeps every interval wide.

### 2026-11-16: Completing the selector tables, the blind control, and the performance law of the answerability gate

**Question.** The local answerability gate had a partial table (MMQA only, no ScienceQA judge control, no sampling control, no blind control). Can the tables be completed so the method line is self-consistent, and can the gate's behaviour be derived rather than described?

**What was run (no new API model; 480 PVS calls on ScienceQA, local GPU otherwise).**

```bash
E=.venv/bin/python; M=/hpc2hdd/home/fye374/models/Qwen3-VL-8B-Instruct
# ScienceQA API PVS control on the same 173 questions (fix: pvs.py now accepts candidate_id as well as cand_id)
$E -m egvqg pvs-select --run-id pvs-sq-aa-v1 --candidates outputs/sq-answerability-v1/candidates_src.jsonl \
  --materials data/processed/scienceqa/lecture-hint-image-v1/materials.jsonl --workers 6
# blind condition (no text, no image) added to the two greedy runs and to the two 3-sample runs
$E -m egvqg answerability-audit --run-id mmqa-answerability-v1 --candidates outputs/vqag-mmqa-pvs60-control-label-v1/candidates.jsonl \
  --materials outputs/mmqa-refer-v2-v1/materials.jsonl --local-model-path $M --device cuda:0 --conditions 0
$E -m egvqg answerability-audit --run-id sq-answerability-v1 --candidates outputs/sq-answerability-v1/candidates_src.jsonl \
  --materials data/processed/scienceqa/lecture-hint-image-v1/materials.jsonl --local-model-path $M --device cuda:0 --conditions 0
$E -m egvqg answerability-audit --run-id sq-answerability-v2 --candidates outputs/sq-answerability-v1/candidates_src.jsonl \
  --materials data/processed/scienceqa/lecture-hint-image-v1/materials.jsonl --local-model-path $M --device cuda:0 \
  --samples 3 --temperature 0.7
$E -m egvqg answerability-report --run-id <run> --pvs-selection <selection.jsonl>
```

**Results (greedy runs; base dual rate 0.141 MMQA / 0.139 ScienceQA).**

| corpus | rule | selected | recall | precision | FPR |
|---|---|---:|---:|---:|---:|
| MMQA | text | 23 | 0.667 | 0.609 | 0.070 |
| MMQA | either side | 15 | 0.571 | **0.800** | 0.023 |
| MMQA | API PVS (149 calls) | 16 | 0.476 | 0.625 | 0.047 |
| ScienceQA | text | 105 | 0.708 | 0.162 | 0.591 |
| ScienceQA | either side | 30 | 0.458 | 0.367 | 0.128 |
| ScienceQA | API PVS (480 calls) | 44 | **0.958** | **0.523** | 0.141 |
| ScienceQA | either side ∨ PVS | 53 | 0.958 | 0.434 | 0.201 |
| ScienceQA | either side ∧ PVS | 21 | 0.458 | 0.524 | 0.067 |

Blind condition: answers with no material in 18/149 MMQA and 19/173 ScienceQA rows, but on genuine duals only **1/21 (0.048)** and **0/24 (0.000)**, agreeing with the full-material answer 0 and 0 times. Paired material bootstrap (local − API): MMQA either-side precision [−0.035, +0.455], recall [−0.222, +0.300]; ScienceQA either-side precision [−0.358, +0.043], recall **[−0.778, −0.154]**.

**The law (why the corpus decides).** With dual prevalence π, luck $\ell=P(\text{side answers}\mid\text{dual})$ and sensitivity $s=P(\text{side answers}\mid\text{redundant})$: recall $=1-\ell$, precision $=\pi(1-\ell)/[\pi(1-\ell)+(1-\pi)(1-s)]$ (an accounting identity), and the gate beats the base rate **iff $s>\ell$** (proved in `docs/necessity_formal_argument.md` §11.3). Measured: MMQA text $s$ 0.930, $\ell$ 0.333; MMQA image 0.211/0.095; ScienceQA text 0.409/0.292; ScienceQA image 0.617/0.250 — all four satisfy the inequality, and the two-sided independence prediction holds (0.397 vs 0.429 MMQA, 0.531 vs 0.583 ScienceQA). ScienceQA is weak for one reason: $s_T=0.409$, i.e. the 8B cannot read the lecture passages well enough to notice they answer the question, where on MMQA it can ($s_T=0.930$). Sampling is counterproductive because $k$ samples replace $\ell$ by $1-(1-\ell)^k$: at 3 samples the text-rule `stable` level drops precision 0.609 → 0.349 (MMQA).

**Blind control refutes the prior-knowledge hypothesis.** The earlier draft attributed level-3 false positives to questions answerable from world knowledge. The blind condition shows the model cannot answer those rows without material, so the side itself answered: they are label errors or missed derivations, to be settled by human review of those rows, not by a new metric.

**Artifacts.** `outputs/pvs-sq-aa-v1/`, `outputs/{mmqa,sq}-answerability-{v1,v2}/audit_report.md`, `docs/necessity_formal_argument.md` §11 (new: closed loop, gate as hypothesis test, precision decomposition, the $s>\ell$ theorem and stop rule, sampling result, verifier cap, measurements), `docs/necessity_formal_argument.md` §10.7 (third negation, Props 16–17). Code: `pvs._cand_id`, `--samples/--temperature`, conditions `TI,T,I,0`, `reject_if_prior_answers`, `reject_if_either_side_or_prior_answers`, `_selector_comparison`, evidence ladder with the `0` side.

**Verification.** `pytest` → **183 passed in 15.74s**; `scripts/sanity.py` → SANITY PASSED; new test `test_answerability_audit_prior_condition_rejects_blind_answers`.

**Unresolved.** (i) Human review of the ~28 level-3 false positives decides label error vs missed derivation; (ii) ScienceQA and MMQA labels remain AI consensus, not human; (iii) the $s>\ell$ law is an identity-plus-theorem, so the *estimate* of $s$ on a new corpus still needs a labelled pilot; (iv) `story.md`/`VQG_Proposal.md` still describe the PVS-only mainline.

### 2026-11-16: Consolidating the current status for an advisor update (no new measurements)

**What was done.** Rewrote the "Current status" head so the five-part route is readable at a glance instead of being buried in a chronological bullet list: a date bump to 2026-11-16, a six-row at-a-glance table (part / instrument / status / headline number), a one-sentence-per-part statement of the method line, and a numbered open list. Collapsed the superseded first-pass answerability bullet into a short pointer that keeps only its MMQA numbers and the direction test, since the completed tables in the next bullet carry the blind control, the multi-sample check and the performance law. Rewrote "Paper outline (draft)" from the single stale §4 paragraph into the five-part structure the measurements support (problem and measurement, redundancy source, why the model cannot certify its own duals, method, evaluation and cost) plus an explicit non-claims block (schema is nested in skill; all labels are AI consensus; pool B is template-designed; the MMQA verifier tier is underpowered). Also fixed the stale 2026-11-13 date in the document preamble.

**Verification.** Documentation-only change: checked the new section boundaries (`grep -c ""` = 751 lines), that no work-log entry was removed (all `###` entries still present), and that every quoted number matches the source bullets it was moved from. No code path touched; the offline suite was green at 183 passed before this edit and no test asserts on PROGRESS.md content.

**Artifacts.** `PROGRESS.md` only. No `outputs/` change.

**Unresolved / next.** The advisor-facing summary is a condensed rendering of this section, not a new document (no separate report file per the project conventions). The substance of the open list is unchanged: human review of the gate's ~28 false positives, the E1 50-row sheet and the 199-row PVS pack; then the pilot-then-deploy rule; then the `story.md` / `VQG_Proposal.md` rewrite.

### 2026-11-16: Documentation consolidation and the technical route (no new measurements)

**What changed.** Documentation was condensed rather than extended, so that the next agent can see which paths are closed
without reading the run history.

- `docs/necessity_estimator_theory.md` was merged into `docs/necessity_formal_argument.md` as **§11** (the loop section,
  the gate as hypothesis test, the exact decomposition, the $s>\ell$ theorem and stop rule, the sampling result, the
  verifier's feasibility cap, the measurement table); the standalone file was deleted and every reference retargeted. One
  theory document remains.
- `INVENTORY.md` was deleted from the root: it described a mainline that is closed (whitebox probe reuse, "no tests/
  directory"), and its remaining facts are stale or already in `docs/implementation_plan.md`.
- `story.md` gained the section **Technical Route: A Closed Loop Built from the Task's Own Criterion**: why the technique
  is the definition made executable (interventional target, no reference needed, constant-size criterion), the loop
  diagram, the law as licensing test and stop rule, where the loop is amplifiable (answering side) and where it is not
  (generation side, load-bearing copy), the four layers with their evidence, and a non-claims list. The Overview bullet and
  the paper-structure rows for Method/Experiments were updated, and the claims table now carries C5–C8 with status.
- `VQG_Proposal.md`: the Abstract method bullet, §5 intro and the execution-order row now describe the gate + repaired
  verifier route instead of a "not yet settled" self-play candidate.
- `PROGRESS.md`: new section **Paths already tried (do not re-derive)** — a 16-row table of closed routes with outcome and
  verdict (instruction-only prompts, numeric suppression, two-step answering, hidden-state probes, steering, best-of-n,
  head ablation, attention blocking, API PVS, the gate, multi-sample levels, blind filter, early stop, per-item repair,
  interval psychometrics, schema conditioning), each pointing at its artifact.

**Verification.** Docs only. `grep -rn necessity_estimator_theory` returns nothing; every markdown link target in the
edited files exists (`story.md`, `VQG_Proposal.md`, `PROGRESS.md`, `docs/implementation_plan.md`); §11 numbering does not
collide with the existing §1–§10.7. Offline suite untouched (last run 183 passed).

**Next.** Human review packs, then the training round that C8 requires (`peft` + a compact LoRA SFT script, ~tens of GPU·h),
if the paper needs the loop to be demonstrated rather than licensed.
