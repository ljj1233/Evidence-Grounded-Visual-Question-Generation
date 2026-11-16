# Implementation Plan

Updated: 2026-11-13. For the research definition see [VQG_Proposal.md](../VQG_Proposal.md); for the mathematical reference see [necessity_formal_argument.md](necessity_formal_argument.md); for progress and the evidence index see [PROGRESS.md](../PROGRESS.md); for command entry points see [README](../README.md). Directory responsibilities, file increments, testing and efficiency rules are unified in [AGENTS.md](../AGENTS.md); no separate set of rules is defined here.

## 1. Current research mainline and execution order

Goal: describe and diagnose the questions the generator produces on real materials — how many are "grounded in the material, answerable, and require both text and image". The mainline is the probing route in Section 7, which produces only questions Q; answers A are out of the current scope. Answerability/faithfulness and bimodal necessity are judged separately; the actual solver's correctness and refusals are reported separately. The four-method comparison framework has been removed; four-condition answering is no longer run by the main pipeline, and the related schema and BDR aggregation functions are retained as legacy helpers, with historical data kept in `outputs/` and marked as the old protocol.

| Item | Current state | Boundary |
|---|---|---|
| Data | ScienceQA test: 1,257 groups with hint and image; context variants 17 groups × 5 text kinds; MultiModalQA pool 120 paragraph+entity-image groups (60 groups sampled for the experiment) | Only a material subset; not filtered by joint space or output quality; original questions, options, answers, and explanations do not enter the material |
| Generation | Qwen3-VL-8B local baseline (plain-v1, old protocol) 1,257 groups completed; luna API baseline 1,257 groups completed; plain-v2 new-protocol rerun executed by the GPU queue | The generator outputs only Q; same-model self-assessment is not an independent human gold standard |
| Evaluation | Cascade evaluation (`simple-baseline-eval-*`): validity/faithfulness → two single-modality screens → counterexamples constructed only for surviving questions; necessity evaluator (`necessity-ensemble-*`): two model groups plus arbitration | Counterexamples and verification are done by models and will let some through; undecided is not flipped into the opposite label |
| Probing route | P0 white-box capture completed and rechecked under 5.22.2; old-protocol P1 pilot and P2/P5 exploration completed; P4 redirected to an exploratory branch | Criteria and thresholds in 7.7–7.10 are not modified after seeing data |

Execution order:

1. Rerun candidate capture and cascade evaluation with plain-v2, and judge G1 per 7.7; if it fails, follow the registered fallback without changing thresholds.
2. After G1 passes, train the linear probe (G2) and take P5 readings R1/R2 on the confirmation set; if it fails, report a null result.
3. Use the same batch of labels to run P6's data-side selection evaluation; the redirected P4 is tested only as an exploratory check on the fixed readings.
4. Use context variants (including the supplementary split S) and MultiModalQA to test material affordance and generalization, without introducing new datasets.

## 2. Models and roles

| Role | Responsibility | Configuration principle |
|---|---|---|
| Development tool | Read the plan, implement code, run checks | Not used to decide the experimental model |
| Generation model | Generate Q from T, I; outputs only Q | Record model version, decoding parameters and prompt snapshot |
| Answering model | Independent full image-text answer (TI) for the baseline; the evaluator's T/I screening conditions are isolated from each other | Removed material and its derived information must not leak |
| Evaluator model groups | Answerability/faithfulness, single-modality sufficiency screening, counterexample recomputation, answer-slot extraction | The two groups are independent; see Section 9 of the mathematical reference |
| Arbitration model | Handles only questions the two groups did not settle | Two out of three; if still no majority, undecided |
| AI review | Verify confirmed items and sampled labels | Record the source truthfully (e.g. `user_delegated_ai`); do not pose as an independent human gold standard |

Model names, endpoints and call budgets are all configured (see README "Model and gateway configuration"); providers are not hardcoded. Role reuse is recorded transparently; self-consistency is not treated as ground truth. The main pipeline no longer runs four-condition answering; the T/I/0 record fields of the old framework are retained only as a historical schema.

## 3. Code structure

`src/egvqg/`, each module with a single responsibility; keep the modules the current pipeline reaches, plus analysis modules that are implemented and covered only by offline tests but not yet wired in (noted).

| Module | Responsibility |
|---|---|
| `cli.py` | Unified command entry point |
| `records.py` | Record format, JSONL read/write, hashing, file locking |
| `providers.py` | Model interface, mock, image transport, caching, call logging and usage |
| `prompts.py` | Template loading and snapshots; a missing variable raises an error |
| `dataset.py`, `downloads.py` | ScienceQA/MultiModalQA import, material normalization; raw data download and checksum |
| `simple_baseline.py`, `simple_baseline_evaluation.py` | Asynchronous baseline generation and TI answering; cascade evaluation after the whole batch completes |
| `baseline_analysis.py`, `faithfulness.py`, `evidence_verification.py`, `assessment_rules.py` | Evaluation input construction, faithfulness derivation, evidence verification gate, knowledge-scope configuration |
| `necessity_ensemble.py` | Necessity evaluator with two groups plus arbitration; decision logic is pure functions |
| `local_vision.py` | Local Qwen3-VL provider: dynamic batching, interface consistent with API providers |
| `candidates.py`, `whitebox.py` | Candidate sampling and white-box capture (P0/P1) |
| `probe_labels.py`, `label_audit.py` | Label assembly and sampling audit |
| `probing.py`, `probe_run.py` | Linear probe, layer selection and G2 decision |
| `modality_distance.py`, `distance_analysis.py`, `distance_robust.py` | Hidden-state shift from modality removal/replacement and R1/R2 |
| `geometry.py`, `variance_decomposition.py` | Geometry readings and variance decomposition (descriptive; the latter has no CLI entry yet) |
| `selection_eval.py`, `intervene.py` | best-of-k selection evaluation (implemented, no CLI entry yet) and steering (exploratory) |
| `synthetic_composition.py` | Week-1 mechanism route: programmatic derived/bridging/stitching/interval materials, split single-hop answers, local-model answering/generation, material-bootstrap composition deficits (CLI `synthetic-build`, `composition-*`) |
| `inline_analysis.py` | R2: inline numbers/rules in generated questions and their association with text redundancy (CLI `inline-report`) |
| `validity_check.py` | R3: local `local-validity-v1` review of generated questions (CLI `local-validity`, `validity-report`; local AI review, not the official evaluator) |
| `material_generation.py`, `mmqa_refer.py` | Generic N-per-material generation for arbitrary materials and the MMQA refer-prompt feasibility comparison (CLI `material-generate`, `mmqa-refer-compare`) |
| `interval_mechanism.py` | Time-boxed behavioural ablations and logit lens on the interval-count failures (CLI `interval-mechanism`) |
| `redundancy_source.py` | E1: splits already-redundant questions into `leak` / `anchor` / `other` by redundant side (programmatic screen + AI judge + manual sheet; CLI `redundancy-source`) |
| `self_verification.py` | E3: local verifier (select iff TI correct and both single sides wrong) with a five-tier repair ladder (`direct` / `two-step` / `delegated` / `delegated-compare` / `per-item`), computable synthetic gold, per-tier recall/precision/coverage, early stop, TI error taxonomy, and optional API-PVS scoring on the same questions; real-data pilot path (`prepare_real_pool`, `run_real_verification`, `report_real_verification`) (CLI `self-verify-run`, `self-verify-report`, `self-verify-real-run`, `self-verify-real-report`, `pvs-select`) |
| `composition_grid.py` | E2a: paired 2×2 modality grid (rule text/image × values text/image) with programmatic conditional/sum scoring, material bootstrap contrasts and a per-item psychometric curve (CLI `grid-build`, `grid-answer`, `grid-report`) |
| `pipeline.py`, `metrics.py`, `experiments.py` | Helper functions from legacy modules still used by the current pipeline; the rest deleted |

`prompts/*.txt` are separately maintained prompt templates (10, `$variable` placeholders; the fixed generation prompt is `plain-v2`); `configs/` holds run configurations without credentials (currently only `necessity_assessment_v2.json`); `tests/` holds a small number of high-value offline tests (mock, synthetic fixtures in `tests/fixtures/`); `data/`, `outputs/`, `tmp/` are not in Git. Records use JSONL first; no database, web annotation platform or distributed scheduler is introduced.

## 4. Phased acceptance

| Phase | Content | Acceptance | Status |
|---|---|---|---|
| A | Engineering skeleton and offline closed loop: records, config, CLI, mock provider, minimal aggregation | One command yields complete records and an aggregate from a small set of mock materials; invalid questions enter the fixed denominator, undecided listed separately; reruns do not double count | Completed |
| B | Real model adaptation and small smoke test: image encoding, timeouts, limited retries, caching, structured parsing, usage logging | Inputs inspectable; cache recovery does not consume twice; failures traceable | Completed |
| C | Real-material baseline: Qwen3-VL local 1,257 groups, luna API 1,257 groups and cascade evaluation | Every question traceable to material, output, answer, evidence; failures and undecided retained | Completed (plain-v1) |
| D | Evaluator freeze and recheck: translation validation, versioned prompts and config, spot-check major misjudgments | Results reproducible; remaining misjudgments and applicability boundaries clear | In progress (English translation has passed the 7.10 agreement threshold) |
| E | Probing route: candidate capture, cascade labels, G1–G4 and P5/R1/R2 | Judged by the pre-registered criteria in 7.7–7.10; null results reported truthfully | In progress (plain-v2 data awaiting queue) |

A method not outperforming a strong prompt is not an engineering failure, and labels must not be changed for that reason. Multiple datasets, training, systematic counterfactuals and perception-exploitation analysis are opened later according to the value of the results.

## 5. Run and record contract (current)

- **`simple-baseline-*` / `simple-baseline-eval-*`**: `manifest.json` freezes the protocol, configuration, plan/prompt hashes and the parameters actually in effect; `plan.jsonl` generates all slots before calls; `records.jsonl` has exactly one row per `attempt_id`, updated with file locking and atomic replacement; `calls.jsonl`/`requests.jsonl` record each technical attempt, prompt, image path and hash, raw output, parsed result, latency and usage. Resuming the same configuration fills only pending slots; changing the configuration requires a new `run_id`.
- **`candidate-*` / `whitebox-capture`**: `captures.jsonl`, `selection.json`, `summary.json`, `model_identity.json` record candidates, prompt snapshots, seeds and white-box tensors; `candidate-source-init` writes `source_materials_*.jsonl`/`source_map_*.json`, and `candidate-labels` assembles cascade labels into `labels_<eval-run>.jsonl` plus a disposition profile.
- **`necessity-ensemble-*`**: `manifest.json` freezes questions, prompts, models and limits; `items.json` stores the frozen questions; `results.jsonl` has exactly one row per `role|kind|attempt`; failed calls are retained but not cached, and are rerun on resume; the report is written to `results.json` and `final_items.json`.
- **Failure semantics**: only technical errors are retried up to the limit; JSON errors, truncation, semantic failures and refusals are not resampled; all failure end states are retained, and resumption fills only pending.
- **Cascade evaluation**: `init` only after generation and answering are fully complete; after validity/faithfulness pass, two single-modality screens are run, and counterexamples are generated only for surviving questions; a screen failure is not treated as "not sufficient", and undecided is not flipped into a label.
- **Necessity evaluator**: answer equivalence is decided by slots generated from the question stem alone; accepted only when the two model groups agree, otherwise arbitration takes two out of three, and no majority is recorded as undecided. Confirmed items count only after recheck.
- **Metrics**: automatic labels and rechecked labels are listed separately and never backfilled automatically; the fixed denominator N is all planned slots, not reduced according to valid or scorable questions; the lower and upper bounds for each item are "pass/N" and "(pass+undecided)/N", which are undecided boundaries, not confidence intervals.

## 6. Validation principles

- A small number of high-value tests with parameterized boundary cases; required: fixed denominator and undecided propagation, material isolation, answer-question correspondence, cache invalidation, no double counting after resumption.
- Real-model prechecks are independent of mock tests and record actual cost; passing mock does not mean the model or metrics are valid.
- For any change to an evaluator, first use a zero-call replay of existing runs to confirm whether decisions change, then decide whether to rerun; English translation validation follows 7.10.
- Label and direction audits are always marked as AI review, never written as human annotation.

## 7. Probing route (from 2026-11-13)

This section is a newly added research route, running in parallel with the baseline and evaluator work in Sections 1–6; milestone numbering uses P0–P7 to avoid confusion with phases A–E in Section 4. For narrative and theory see [story.md](../story.md); for the mathematical argument see [necessity_formal_argument.md](necessity_formal_argument.md).

### 7.1 Research questions and scope

Input image and context text, output question Q. **Answer A is not generated for now, and is left as a later extension.** Research questions:

| ID | Question | Output |
|---|---|---|
| RQ-P1 | Before or during question generation, has the model internally already encoded which modality this question will rely on (the direction of redundancy)? At which layers and which positions? | Layer-wise probe curves, against baseline and control tasks |
| RQ-P2 | Is this encoding causally used by generation? | Effects of activation patching, direction removal and steering, against random directions |
| RQ-P3 | Why the bias toward one side? Is it related to geometric differences between image and text token representations? | Anisotropy and effective rank across layers; intervention results from explicit recentering |
| RQ-P4 | Can internal signals raise the proportion of verified bimodal necessity and reduce evaluation cost? | Intrinsic best-of-k selection, steering along the direction, compared with random selection, judge-model selection, plain and strong prompts |

The probed object is the **generator** (local Qwen3-VL-8B-Instruct, 36 layers, hidden size 4096, GPU 3). Labels come from the cheap cascade of the answer-identifiability evaluator (validity plus two single-modality screens); verified bimodal necessity covers only 19 questions and serves only as a small-sample supplement.

### 7.2 Design points

1. **Probe positions:** P-a at the end of the prompt (before the first generated token); P-b the first few generated tokens of the question; P-c at the end of the question.
2. **The target is a propensity, not a single output:** sample k candidates per material group, use the proportion of redundant among them (and the direction proportion) as the regression target; also retain the binary classification of individual candidates.
3. **Split by material and by template:** multiple candidates of the same material do not cross train and test; templates are held out as whole groups.
4. **Baseline and control tasks:** input-feature baseline (template ID, text length, image statistics); a control task with labels randomly assigned by template or material, reporting selectivity (true task minus control task).
5. **Linear probe as primary**, MLP only as an upper bound; multiple random seeds; grouped bootstrap intervals.
6. **Every probe conclusion is paired with a causal test:** activation patching, direction removal or steering, against random directions and equal-norm random directions.
7. **Do not use attention weights as "how much modality was used":** use hidden-state distance (with/without image, with/without text), patching effects and gradient attribution, and validate other readings against patching results.
8. **Modality removal must account for distribution shift:** use two or three removal methods simultaneously (no image, same-template image swap, centroid replacement, etc.), and accept only conclusions that agree.
9. **Analyze only on valid questions,** to avoid contamination by invalid questions; "cannot answer" is not taken as evidence of necessity.
10. **Write pass criteria in advance,** report null results; for cross-model replication compare layers by relative depth.

### 7.3 Milestones

Scale: S about hours to one day, M about 2–4 days, L about a week or more; all are estimates, corrected by measurement.

| Milestone | Content | Deliverable | Acceptance and gate | Scale |
|---|---|---|---|---|
| **P0 white-box capture** | Generate questions on Qwen3-VL-8B for the `plain-v1` prompt, while saving per-layer hidden states and token positions at three positions (including the image token span and the text token span) | Capture module, offline tests, capture samples for 8 material groups, GPU memory and speed records | Greedy output matches the existing baseline's first 8 groups, or differences are explained; hidden-state shapes and positions correct; GPU 3 occupancy recorded | S |
| **P1 candidate dataset** | Pilot first: about 60 material groups (stratified by template) × 8 candidates, sampled with a fixed seed; labeled with the existing cascade evaluation; after the pilot is confirmed, expand to about 300 groups | Candidate and label table (material, template, candidate, validity, redundancy direction, survival); captured hidden states | Label coverage and class distribution reported by template; 40 labels spot-checked by Claude, precision meets the bar; first pass **G0**: confirm API call scale before the pilot | M |
| **P2 propensity profile** | Statistics per material group of redundancy proportion, direction proportion and candidate diversity; whether non-redundant candidates exist | Profile tables and figures | **G1:** if the redundancy proportion is constantly 1 for the vast majority of materials (no target variance), first add prompt variants (e.g. strong prompt) to create variance, or switch to probing the redundancy direction; if there is still no variance, switch to reporting that finding | S |
| **P3 linear probe** | Three positions × each layer, linear probe plus baseline and control tasks; classification (redundant or not, direction) and propensity regression | Layer-wise curves, selectivity, grouped intervals | **G2:** must satisfy both: better than the input-feature baseline and the interval excludes 0; selectivity is positive. Otherwise report a null result and stop subsequent causal investment | M |
| **P4 causal test** | Compare activation patching between candidates under the same material; removal or addition/subtraction along the probe direction; random and equal-norm random controls | Localization of layer bands and heads, label flip rate, validity retention | **G3:** effect larger than random controls, and the question remains valid; otherwise keep only the probe conclusions and make no causal claims | M–L |
| **P5 mechanism: geometry and gradients** | Anisotropy of image and text tokens, effective rank across layers; intervention by explicitly subtracting the dataset mean; input-times-gradient as corroboration | Geometry curves, intervention results, correlation with P3 and P4 | No gate, report truthfully; when gradient readings conflict with patching, patching takes precedence | M |
| **P6 solution** | (a) intrinsic best-of-k selection (score from probe or hidden-state distance); (b) steering along the validated direction; controls: random selection, judge-model selection, plain and strong prompts | Verified bimodal necessity proportion, validity rate, cost, with intervals | **G4:** verified bimodal necessity proportion increases and the interval excludes 0, and validity rate does not decrease; otherwise write it up as a negative result. Contrastive decoding is last because the literature shows no gain on generative tasks | L |
| **P7 replication and document sync** | Replicate P3 with a second model (candidate: another vision-language model available locally, subject to confirming it can be loaded); sync story.md, VQG_Proposal.md, references.md, PROGRESS.md | Replication table and document updates | Compare layers by relative depth; record differences truthfully | S–M |

### 7.4 Resource estimates (to be corrected by measurement)

- **Storage:** each candidate at each position and layer is 4096-dimensional, bf16, about 8 KB. In P0 measurement, each candidate stores 6 vector groups (1 at prompt end, 4 question-prefix tokens, 1 at question end) × 37 layers, about 1.8 MB; 2400 candidates about 4.4 GB.
- **API:** based on the mean of the last cascade evaluation (about 2.4 calls, about 3.4 thousand tokens per question), the pilot's 480 candidates need about 1.2 thousand calls and about 1.6 million tokens; 300 groups × 8 about 5.8 thousand calls and about 8.2 million tokens. Confirm scale before launch.
- **GPU:** Qwen3-VL-8B bf16 weights are about 17 GB; GPU 3 is a shared device, check occupancy before launch, use only GPU 3, do not touch other people's processes.

### 7.5 Code and record conventions

- Follow [AGENTS.md](../AGENTS.md): new modules with a single responsibility, created as needed; unified CLI entry; outputs go to `outputs/<run_id>/`; offline tests use mock; progress appended to `PROGRESS.md`.
- Suggested module split (adjustable during implementation): one module each for candidate sampling and hidden-state capture, probe training and controls, and causal intervention; annotation reuses the existing cascade evaluation, and candidates can simply be written in the baseline run's record format.
- Evaluation-related definitions are in `docs/necessity_formal_argument.md`; evaluator prompts and decision rules are not modified within this route.
- Batch composition changes low-order numerics: use a fixed batch or batch size 1 during capture, and record it in the run's `model_identity.json`.
- Local loading defaults to sdpa and does not return attention weights; this route does not rely on attention readings.

### 7.6 Risks

| Risk | Mitigation |
|---|---|
| Probe only learns the template, not the model's decision | Grouped splits, input-feature baseline, control tasks |
| No target variance (almost all candidates redundant) | G1; prompt variants; change probing direction |
| Too few bimodal-necessity positives | Main task uses redundancy labels; bimodal necessity only as small-sample intervals |
| Decodable does not mean used | P4's causal tests and random controls |
| Modality removal causes distribution shift | Take agreement across multiple removal methods |
| Single model, single dataset | P7's second model; conclusions limited in scope |
| Evaluation cost | Pilot first; reuse cascade evaluation; intrinsic signals as a cost-reduction direction |

### 7.7 Pre-registration for autonomous execution (2026-11-13, from P1; written before any pilot labels appeared, not changed afterwards)

**Total budget (hard upper limit for the whole route):** API calls ≤10,000, tokens ≤15M; P1 pilot ≤1,500, expansion to about 300 groups ≤6,500, P6 ≤2,000. The ledger is written in `ledger.json` in each run directory and in PROGRESS.md. GPU: use only card 3; if free GPU memory before launch is <30 GB, wait and retry.

**P1 design decisions:**
- Template ID = the `skill` field of ScienceQA (used only for stratification and grouping, never enters the prompt). Pilot = 20 skills × 3 material groups; skills are taken from the 40 skills with ≥6 materials in the test set (seeded 20261113 random order, first 20); materials are sampled randomly within a skill, not selected by any label. Expansion = 40 skills × 8 material groups (nested: the pilot is a subset), and the expansion run fills in only materials not in the pilot.
- 8 sampled candidates per material group: temperature 0.7, top_p 0.8, top_k 20 (taken from the model's own `generation_config.json`), seed hashed from (run seed, material, candidate index), batch size 1.
- Labels: the existing cascade evaluation (validity → two single-modality screens), without the necessity evaluator; candidates with identical text within the same material are evaluated only once, and labels are duplicated by multiplicity. Candidate question language (Chinese character proportion) is recorded per candidate and enters the input-feature baseline.
- Candidates are written in the baseline source-run format (pseudo-material = material × unique question), answered greedily by local Qwen3-VL-8B (`solve-v1`, same as the baseline), then evaluated with `simple-baseline-eval-init --cascade`.

**P1 pilot pass criteria (trigger condition for expansion, all must be met):**
1. Label coverage: ≥95% of candidates receive a decided label (valid/invalid, excluding evaluation failures and pending).
2. Label precision: Claude draws 40 items by stratified random sampling from valid candidates "screened as single-modality answerable" (AI review, not independent human annotation), Wilson 95% lower bound of agreement ≥0.80.
3. If not passed, fix the cause only once; if it still fails, stop that branch and report.

**G1 (pre-registered criterion for P2):** among materials with at least 4 valid candidates, the target is considered to have variance if either of the following holds: (a) materials with redundancy proportion strictly between 0 and 1 ≥25%; (b) materials with mixed direction (image sufficient / text sufficient / both sufficient) ≥25%. If neither holds, follow the fallback in 7.3 (strong-prompt variant; or probe the redundancy direction).

**G2 (P3):** the main task is classification of "whether the candidate is redundant" (on valid questions), the secondary task is direction and propensity regression. Passing requires all of: (i) the lower bound of the best layer's grouped bootstrap interval is above the corresponding interval of the upper point estimate of the input-feature baseline (i.e. the two intervals do not overlap, or the bootstrap 95% interval of the difference excludes 0); (ii) the selectivity (true task minus control task) interval excludes 0; (iii) it holds across three seeds and with template whole-group hold-out.

**G3 (P4):** the change in label flip rate (or probe reading) caused by intervention along the probe direction is larger than for random directions and equal-norm random directions (bootstrap interval of the difference excludes 0), and the proportion of intervened questions that remain "valid" is not lower than the control.

**G4 (P6):** under the verified-label criterion, the selection method's "non-redundant and valid" proportion is higher than random selection, the grouped bootstrap 95% interval of the difference excludes 0, and the validity rate does not decrease (decrease no more than 2 percentage points).

**P7:** done only if another vision-language model in `/home/lijunjian/models` can be loaded with the installed stack and placed on card 3; otherwise explain the reason and skip.

**Label criteria and G2 analysis details (pre-registered, supplementing G1–G4 above; does not change their thresholds; written before pilot labels appeared):**
- Valid candidate = `question_answerability==1` from the cascade evaluation. Redundant = `screen_cut` is true; non-redundant = both single-modality screens are explicitly false (no null values); the rest (screen null and not cut) are recorded as undecided and do not enter classification. Direction: image-only sufficient / text-only sufficient / each of the two sufficient.
- The `prompt_end` state is identical for the 8 candidates of the same material, so it can only predict material-level propensity (redundancy proportion, regression); candidate-level classification uses only `early` (mean over 4 tokens and each token individually) and `question_end`. `question_end` can read the question content, so a separate "question surface features" baseline is set (Chinese proportion, length, logistic regression on question bag-of-words TF-IDF), and claims target only the part beyond that baseline.
- Discovery and confirmation are separated: **the pilot (60 groups) is used only to select one layer per position (highest cross-validated AUC with template whole-group hold-out, R² for regression), and the confirmation set = materials newly added by the expansion (materials outside the pilot)**; on the confirmation set that layer is fixed and evaluated with 5-fold by skill whole-group hold-out (3 seeds shuffling the fold assignment); curves for all layers are reported separately as descriptive results on the merged data and are not used for the pass gate. If the pilot does not trigger expansion, G2 is recorded as "not tested".
- Probe: standardized L2 logistic regression (C=0.01, fixed), regression uses ridge (α fixed at 1e4 / number of features; the scaled form is in the code), no hyperparameter tuning; MLP only as an upper-bound reference. Intervals: bootstrap over materials as whole groups (1000 times).
- Input-feature baseline: A = text character count, number of options, image width/height, grayscale mean and standard deviation, question language (candidate-level); B = skill one-hot (usable only in the "held-out by material" split). With template whole-group hold-out the baseline is A. Selectivity = true-task metric minus control-task metric, where the control task = randomly permuting material-level labels across materials (preserving the marginal distribution), through the same pipeline.
- G2's criteria are checked item by item on the confirmation set: (i) probe AUC (R² for regression) minus baseline bootstrap 95% interval excludes 0; (ii) selectivity interval excludes 0; (iii) the point estimates of the 3 seeds have the same sign.

**Pre-registration of the G1 fallback (written on 2026-11-13, after the pilot `plain-v1` G1 had been judged a failure under the above rules and before strong-prompt labels appeared):** G1 result for the pilot `plain-v1`: 52 evaluable material groups, redundancy-proportion mixed 13.5%, direction mixed 17.3%, both below 25%, failed; variation was mainly between materials (8 mixed, 4 all non-redundant, 43 all redundant, concentrated in a few skills such as "Punnett square" and "Trade and specialization"). Per the 7.3 fallback, a prompt variant was added: for the same 60 material groups, sample 8 candidates each with `strong-v1` (same sampling parameters and seed rule), same evaluation pipeline.
- Decision: merge the candidates of the two prompts, **with (material, prompt) as the unit**, recompute G1 with the same two 25% criteria; if it passes, continue to P3; otherwise report per the second fallback in 7.3 (direction): the direction target already failed on the pilot `plain-v1`, so G1 is treated as failing overall, the propensity probe becomes a "null-result report", only the descriptive profile is kept, the probe-dependent parts of P4–P6 stop, and the P6 baselines that do not depend on the probe (random selection, judge-model selection, plain/strong prompt comparison) proceed as usual.
- If it passes: the prompt variant itself will be "read out" by the probe, so (i) input-feature baseline A adds prompt-variant one-hot; (ii) P3 additionally reports results **within each single prompt only**, and passing G2 requires the within-prompt results to satisfy the criteria as well (interval excludes 0); (iii) the confirmation set likewise needs both prompts, so strong-prompt candidates are also generated for expansion materials (within budget).
- Label precision audit (AI review, not human): pilot `plain-v1` 40 sampled items, "at least one modality alone sufficient" 40/40, Wilson lower bound 0.912, passed ≥0.80; direction agreement with the reviewer 27/40 (Wilson lower bound 0.52). Therefore **the reliability of direction labels is clearly lower than that of redundancy labels**; all direction conclusions are downgraded to exploratory, and G2 is based on the redundancy and propensity tasks. Expansion is therefore triggered.

**Pre-registration for P5 / P6 / P4 (written before any "distance and label" relationship was examined; 2026-11-13):**
- **P5 readings (no gate, report truthfully):** for each candidate, teacher-force the same question tokens through six prompts: full, no image, gray image (preserving token count), same-skill image swap, no text (empty string), same-skill text swap (`modality_distance.py`); take the prompt-end and question-end positions, and compute per-layer relative L2 distance and cosine distance. Targets: redundancy (binary) and direction (image sufficient vs. text sufficient, only on candidates with a clear direction), statistic AUC, intervals by material bootstrap. An "accepted conclusion" requires that at least two of the three paired schemes (no image | no text, gray image | no text, image swap | text swap) have AUC intervals in the same direction and excluding 0.5. Geometry: per-layer anisotropy of image tokens and text tokens (mean of pairwise cosines after unit normalization) and effective rank (exponent of the entropy of singular values after centering), on the 60 pilot materials, by material bootstrap; correlated with material-level propensity, descriptive. Gradient attribution corroboration (input times gradient, attribution share of image tokens relative to text tokens) is done only on a subset of pilot materials; when it conflicts with patching/distance, patching takes precedence.
- **P6 selection evaluation (reuses existing labels, produces no new calls):** for each material, the selector picks one of all its candidates, and the result = the selected candidate being "valid and non-redundant" (cascade-label criterion, not "verified bimodal necessity", which is reported separately only on the small sample). Selectors: random (expectation), intrinsic score (`min_shift`, layer and position chosen only on the pilot), probe score (P3's cross-fold prediction, only if G2 passes), judge-model selection (1 call per material, ≤ number of materials). Comparison is done on the confirmation set: the material-level paired bootstrap 95% interval of selector minus random excludes 0 and the validity rate of selected candidates is not more than 2 percentage points lower than random, which is G4 passing; also report the subset of materials containing both redundant and non-redundant candidates. The success-rate comparison of `strong-v1` vs `plain-v1` is listed separately, as a prompt baseline that does not depend on internal signals.
- **P4 causal test (done only if G2 passes; when API is needed, use the remaining route budget, ≤1,500 calls, and not exceeding the 10,000 total cap):** at the prompt-end position, in the selected layer band, perform removal and addition/subtraction along the probe direction (strength 1× and 2× the standard deviation of the reading along the probe direction), against random directions and equal-norm random directions, generating candidates for each group and using the same cascade evaluation; G3 criteria as above. Direction removal and steering act only on the last prompt position and positions generated after it.

**Pre-registration of P5 robustness checks R1–R3 (2026-11-13; written before the expansion confirmation set had labels. The exploratory readings already seen on the pilot — shift difference AUC about 0.9 on direction, non-redundant positives coming from only about 12 materials — are not treated as evidence for these checks):** the purpose is to rule out two confounds: the signal is merely "template difference", or merely "input surface features" (text length, image statistics).
- **R1 within skill:** for each reading (target × position × metric × paired scheme, layer taken as selected on the pilot), first subtract the mean of that reading within each skill (keeping only skills containing both label classes), then compute AUC on the residuals, intervals by material bootstrap; also report the number of skills, materials and positives involved.
- **R2 baseline adjustment:** logistic regression (C=1, standardized) adds the shift reading (shift difference and the smaller value) on top of "skill one-hot + input features A", compared with the model without the reading, cross-fold AUC difference over 5-fold by material whole-group hold-out, 3 seeds; the difference interval by material bootstrap. A = text character count, number of options, image width/height, grayscale mean and standard deviation, question language.
- **R3 cross-template:** the readings themselves need no training; cross-template hold-out of the hidden-state probe is G2 (already registered).
- **Decision scale:** pass = for at least one target, on the confirmation set the R1 interval excludes 0.5 and the R2 difference interval excludes 0, and at least two of the three paired schemes are in the same direction; good = pass, and G2 passes; hard evidence = G3 passes. Pass only means "conditional signal exists within template" and does not extrapolate across templates. If any one is not met, write truthfully that "the signal can be explained by template or surface features".
- If positives are too few (for any target, R1 has fewer than 3 usable skills or fewer than 20 positives), the check is recorded as "not evaluable", without relaxing the criteria.

### 7.8 GPU day queue after environment unification and pre-registration of new experiments (2026-11-13; written before any labels or readings of these experiments appeared)

**Environment and reruns:** all local model steps uniformly use the project `.venv` (transformers 5.22.2, torch 2.11.0+cu128, flash-linear-attention 0.6.0; `pyproject.toml` and `uv.lock` are locked). 5.22.2 upscales images with fewer than 65,536 pixels according to the model config (4.57.6 does not; about 41% of ScienceQA materials are affected), so all previous Qwen3-VL data (P0, pilot, expansion, shift, geometry) are rerun under 5.22.2 with new run_ids (suffix `-v2`), old results kept for reference and not overwritten. The first step is the 8-material recheck of P0 (`whitebox-capture` plus `whitebox-compare`), truthfully reporting differences in greedy output and hidden states relative to the old records. Execution scripts: `scripts/gpu_queue.sh` (stream A: Qwen3-VL main data; stream B: Qwen3.5 smoke test, context variants, MultiModalQA, Qwen3.5 pilot; stream C: steering screening) and `scripts/api_queue.sh` (cascade evaluation, with budget guard); each step has a completion marker, and restarts fill only missing steps. The API ledger is checked with `api-ledger` (as of this section, 1,340 calls have been used); the guard before a single evaluation is that used plus 2.5× the number of distinct questions does not exceed 9,000.

**V context-variant experiment:** 17 materials (`data/processed/scienceqa/context-variants-v1`) × 5 text kinds O (original), A (information-poor), C (original plus a literal statement of image content), P (A plus task-irrelevant boilerplate, length-matched to C), S (complementary split, see below), image unchanged, 8 sampled candidates per kind (same seed rule), cascade labels. Reading one (no labels): the order of text-removal and image-removal shifts across O/A/C/P/S, only as a sensitivity check of the reading, not mechanism evidence. Reading two (with labels): the change in question type — the proportion of "text alone is enough to answer" (direction text or both) and the proportion "redundant"; main comparisons are C vs P, A vs P, S vs O, S vs P, paired bootstrap with parent material as the unit (17 units, intervals will be wide). The difference interval of C vs P excluding 0 is recorded as "information-content effect", and the difference interval of P vs A excluding 0 as "text-length effect"; either or both may be absent, reported truthfully.

**S supplement (2026-11-13, written before any variant labels appeared):** on the same 17 parent materials, add a 5th text kind S (complementary split): the text gives only about half of the facts needed to answer a question, and the image carries the other half; the text does not restate content visible in the image (more than half of it). S is written manually after viewing each image, and `review_status` is recorded as pending. S is used to test whether "distributing facts complementarily across the two modalities" can raise the proportion of verified bimodal necessity, rather than merely writing image content into text as C does. The outcome variables are the same as C vs P: proportion of direction text or both, redundancy proportion, paired bootstrap by parent material (17 units, intervals will be wide); the difference interval of S vs O and S vs P excluding 0 is recorded as "complementary-split effect". Like C, no pass gate; if it does not appear, report truthfully; the C vs S difference is additionally described, to distinguish "text restating the image" from "text supplying missing facts".

**M MultiModalQA generalization:** 60 "Wikipedia paragraph plus entity image of the same name" materials (`data/processed/multimodalqa/mmqa-pool-v1`, seed 20261113, no questions or answers involved), 8 candidates each, cascade labels. Report: validity rate, redundancy proportion and direction composition (side by side with ScienceQA); P5 readings use the layer selected on the ScienceQA discovery set, and compute AUC on M (material whole-group bootstrap); R2's baseline contains only input features (no skill one-hot, since there are no templates); P6 intrinsic-score selection vs random paired comparison. No gate; "accepted" follows the P5 rule (at least two paired schemes with intervals in the same direction and excluding 0.5).

**P7 Qwen3.5-9B replication:** the same materials and seeds as the Qwen3-VL pilot (`qwen35-cand-pilot-v1`; non-thinking mode; answering still uses Qwen3-VL-8B to keep the label criterion consistent), P5 readings compared with Qwen3-VL's discovery-set results by relative depth (layer index divided by layer count); write "replicated" only when sign and interval agree.

**Concrete settings of steering (exploratory new branch):** training uses only the labels of the plain pilot and expansion (v2), randomly holding out 8 skills for evaluation; direction is taken from the `early` state, first removing the skill mean, then the Chinese-English direction (`intervene.py`); layers 12, 20, 28, strength 1× and 2× the standard deviation of the training reading along the direction; conditions are the true direction with both signs, random same-distribution directions, and isotropic Gaussian directions, with the same norm; evaluation materials are 16 from the held-out skills, 4 candidates each, using the same seed as the no-intervention control. Label-free screening: those survive that end normally at the end-of-sequence token with proportion ≥0.9, have mean token count within 0.5–2× the control, and change in Chinese proportion relative to the control ≤0.25; only surviving conditions and their same-layer same-strength random controls enter API evaluation (≤1,200 calls). Criteria follow G3 (the lower bound of the interval of the true direction's "valid and non-redundant" improvement is above the upper bounds of the two random controls, and the interval lower bound is greater than 0, with validity rate decreasing by no more than 2 percentage points). This branch does not change the registration that "the propensity probe becomes a null result" after G1 fails; results are reported in a separate paragraph, marked exploratory.

### 7.9 Generation prompt changed to neutral plain-v2 (2026-11-13; protocol change, written before any plain-v2 data appeared)

- **Reason:** the `plain-v1` instruction contained leading wording — "educational text" (implying a teaching scenario), "clear, natural, answerable from the material" (a quality requirement, and "from the material" pushes the question toward where the answer can be found in the material), "the image is provided via image input" (a meta note), and the whole instruction was in Chinese, so English materials were also led toward Chinese questions (392 of the pilot's 480 candidates were Chinese). The researcher required fixing a prompt that only says "ask a question combining the context and the image".
- **plain-v2 (fixed, no longer changed):** English, the full text is `Based on the context and the image, ask one question.` / `Context:` / `<T>$T</T>` / `Output only the question.`; the `<T>` markers are kept because position localization depends on them; the last sentence is a format constraint required for parsing, not a content restriction. The material the model sees is still the image plus ScienceQA's `hint` text; the original question, options, answer, and explanation are not given.
- **Timing and honesty statement:** this change happened after plain-v1's G1 failed. plain-v1's conclusions (G1 failed, propensity probe becomes a null result, pilot exploration of P5 readings) are unchanged and will be written into the final report; plain-v2 runs separately as a new protocol instance, **criteria and thresholds follow 7.7 (G1–G4, P5/R1/R2, G2 details) unchanged**, and plain-v1's judgment will not be changed retroactively because of plain-v2's results. Strong prompts (plain-v1's fallback for G1) are no longer generated.
- **Isolation of the version recheck:** P0's recheck under 5.22.2 still uses plain-v1 (run `qwen3vl8b-whitebox-p0-sw5`), only testing the effect of the software version; the other reruns (`-v2`) use plain-v2. The list of replayed questions in the source runs (`candidate-source-*`) is still recorded as plain-v1's prompt snapshot; the actual generation prompt is determined by the capture run's `model_identity.json` and the `generation_identity.json` in the source run directory.
- **Language:** the instruction is English, consistent with English materials, avoiding Chinese instructions leading questions toward Chinese (under plain-v1, 392 of the pilot's 480 candidates were Chinese); this also removes "question language" as a confound, but language is still recorded per candidate.

### 7.10 Validation of the English evaluator translation (2026-11-13; written before any English evaluator labels appeared)

- **Purpose:** 9 evaluator and answering prompts plus the knowledge scope in `configs/necessity_assessment_v2.json` have been translated into English; placeholders, JSON keys, enum values and decision rules are unchanged. This section only tests whether the translation changed redundancy labels, and does not change any evaluation definition.
- **Data and runs:** 194 distinct questions from source run `qwen3vl8b-cand-pilot-src-v1`; old labels are stored in `qwen3vl8b-cand-pilot-eval-v1` (Chinese prompts, model `gpt-6-luna`). Use `simple-baseline-eval-init --cascade` with the same model and settings to create a new run_id, then rerun with `simple-baseline-eval-run`, estimated at about 460 calls (within the whole route's 10,000 cap).
- **Label criteria (following 7.7):** valid question = `question_answerability==1`; redundant = `screen_cut` is true; non-redundant = both single-modality screens are explicitly false (no null values); the rest are recorded as undecided and do not enter the agreement rate. Direction = image-only sufficient / text-only sufficient / both sufficient.
- **Criteria written in advance (not modified because of results):**
  1. Main criterion: on questions where both runs give redundant/non-redundant, agreement ≥ 0.90, and the Wilson 95% interval lower bound ≥ 0.85.
  2. Report only, no gate: direction agreement rate, number of disagreements stratified by old label and the specific items, agreement rate of validity decisions.
  3. One fix allowed (changing only translation wording, not semantics, placeholders or decision rules) before rerunning; if it still fails, stop and report "labels unreliable", and do not proceed downstream.
- **Artifacts:** `translation_validation.md` in the new run directory (comparison table and intervals); PROGRESS.md records the conclusion. AI review is always written as AI review, never posing as human annotation.

### 7.11 VQAG mainline code (2026-11-13; built and dry-run only, no real API/GPU run)

**Purpose:** code for the mechanism/solution route (H1 input-driven, H2 template-dominant, H3 single text↔image axis) without running experiments. Definitions remain those of `VQG_Proposal.md`; this section only fixes the implementation surface. Reuse points are listed in `INVENTORY.md`.

| Module | Responsibility |
|---|---|
| `stats.py` | paired bootstrap over parent materials, Wilson interval, grouped AUC interval |
| `centering.py` | none/global/skill/material centering, row-internal Pearson contrast, balance value |
| `candidate_table.py` | convert `captures.jsonl` + `labels_<eval>.jsonl` (+ `labels_new.jsonl`) to `candidates.jsonl`; load hidden states aligned by `cand_id` |
| `diag_template.py` | 1b: nested SS shares, norm ratios, within-material cosine, token-length control |
| `diag_curve.py` | 1c: probes after each centering level (parent-grouped, label-balanced folds), material-internal AUC, saves `w_dir.npy` |
| `llm.py` | one cached call path (`hash(model, prompt, image_id)`), call counter, `--max-calls` |
| `label_repair.py` | stage 0: stratified sample, two judges on isolated T-only/I-only payloads, third-judge arbitration, verdict + evidence, precision/recall report |
| `select.py` | 2a: PRISM-style material-centered outlier, balance to b, min_shift, random analytic baseline, uncentered ablation |
| `steer.py` | 2b: `h + α(b−⟨h,ŵ⟩)ŵ` clamp and `h + βσŵ` push at prompt_end / all_generated / both; orthogonal random and skill-mean controls |
| `factsplit.py` | 1a/P2: LLM fact extraction and A/B split, condition materials, existing/render image backends |
| `filter_baseline.py` | 2c: keep candidates changed/undeterminable under both counterfactuals; rewrite branch signature only |
| `scripts/sanity.py` | the single synthetic check of the shares, centering signal, steering formula and null bootstrap |

**Status:** all modules implemented; `scripts/sanity.py` passes; every command was dry-run on the existing confirm features (480 candidates) or synthetic data. Dry-run estimates: label repair ≈792 calls for 60 parents and ≈4,224 for 320 parents (≤3 candidates, 2 sides, 2 judges, +10% arbitration), within the remaining route budget. `vqag-steer` CPU hook check passes; its dry-run prints ≈6,720 local generations and ≈3,226 label calls for 60 materials × 14 conditions × 8 candidates. Real API/GPU runs remain for the researcher to launch.

### 7.12 Time-boxed mechanism route: cross-modal composition gap (2026-11-15; pre-registered before any synthetic answer data)

**Motivation.** All previous mechanism work read the generator's internal state or perturbed global decoding and treated "how the generator chooses a question" as the object. Those probes were association analyses and every intervention was null. The remaining untested hypothesis is a capability boundary: the generator only emits questions whose answer it can compose in a single forward pass; cross-modal serial composition is limited, so generated questions collapse to the stronger single modality. This explains why generator-side probes, global decoding perturbations and planning signals in the materials should all be null: the bottleneck is composition, not selection. G3b supplies the only current effect-size evidence: on MMQA dual questions with a luna reference, Qwen3-VL-8B reaches TI 29.4%, T 0%, I 8.8%; 19/25 duals that Qwen could not answer under TI are answerable by luna. Related work: Biran et al. (EMNLP 2024) locate the first hop early and the second late, and back-patching repairs up to 66% of initially wrong two-hop questions; Compose and Fuse (ICLR 2026) find a task-composition bottleneck, that two-step "recognize then reason" prompting restores performance, and that early-layer attention softening helps while late-layer edits do not.

**Scope.** Two weeks maximum; the measurement paper remains the fallback. No new generator-side probes. This section registers only week 1 (W1-A/B/C); later weeks (attention knockout, Patchscopes/logit lens, back-patching, then applying the best repair to generation) are opened only if the stop-loss is not triggered.

**Materials.** New synthetic set `data/processed/synthetic-composition-v1`, generated by `synthetic-build`, 60 materials per category, seed 20261115. Every material carries the full question plus two single-hop questions with programmatic answers:
- `derived` (serial): text gives "only items with value at least k count", image gives four values A-D, full asks the total of counting items; text hop asks k, image hop asks the four values. The threshold is chosen strictly between two adjacent values so that it decides whether the second-smallest value counts (2 or 3 items count) and never equals a value; this avoids an "at least" boundary ambiguity and makes the text rule necessary. A separate dataset `synthetic-composition-count-v1` (`synthetic-build --derived-full count`) asks "How many items count?" instead of the total, isolating set selection from summation.
- `bridging` (serial): text says "the value in the {column} column is called alpha", image has left/middle/right columns; full asks the value of alpha, text hop asks which column, image hop asks that column's value.
- `stitching` (parallel): text gives k, image gives item C; full asks both; text hop asks k, image hop asks item C; k is forced different from item C's value so the full answer cannot be satisfied by one number.

**W1-A composition gap (no judge).** Qwen3-VL-8B-Instruct, TI only, direct prompt `composition-answer-v1`, greedy. Answer the full question and both single-hop questions; score with programmatic answer kinds (`numeric`, `item_values`, `both_numbers`, `column`); abstentions and missing rows count as incorrect (fixed denominator = planned items). Per category report `acc(full)`, `acc(text hop)`, `acc(image hop)`, `product = acc(text)*acc(image)`, `gap = acc(full) - product`, material-level bootstrap 95% CI (10,000 resamples within category), and `P(full wrong | both hops correct)`. Also write the list of materials where both hops are correct but the full question is wrong; this list is the input to later mechanism work.

**W1-B two-step prompting.** Same items and model, prompt `composition-two-step-v1` (condition/name, image values, then answer). Primary readout is the change in full-question accuracy relative to W1-A. Because the two-step prompt can itself change single-hop extraction, report three gaps: direct gap, two-step gap (both hops under two-step), and a mixed gap (two-step full accuracy minus the direct run's hop product). No judge.

**W1-C cross-size check.** Repeat W1-A with Qwen3-VL-2B-Instruct and Qwen3-VL-4B-Instruct (8B reused), and let each of the three models generate four questions per material with `plain-v2` (temperature 0.7, top_p 0.8, top_k 20). Generated questions are labelled by a fixed grammar parser over the synthetic materials: dual (derived/bridging/stitching), single-text, single-image, other; report dual rate over all and over parsed questions, and `other` rate. Report the relationship between each model's mean serial gap (derived and bridging) and its dual-generation rate; with three models this is descriptive, not a test.

**Stop-loss.** If the 95% bootstrap intervals of both the `derived` and `bridging` composition gaps include 0, the mechanism route stops and work returns to the measurement paper. If only one includes 0, continue with the other category and report the distinction between serial and parallel composition. `stitching` is the parallel control: its gap is expected to be near 0 because the full answer is a conjunction of two independent hops.

**Honest boundaries.** These are synthetic materials with a controlled table image, not ScienceQA; the generator-side link is not tested in week 1. The contribution over Compose and Fuse must remain (a) certified complementary evidence in context, (b) separating serial versus parallel composition, and (c) eventually connecting the answer-side mechanism to the generation side. Week-1 results alone support only (a) and (b).


**Post-registration measurement-validity note (2026-11-15, after the first synthetic runs).** The first synthetic builds had three defects found during pipeline/smoke inspection, not from hypothesis testing: (i) the bridging image put a row label in the left column, so "the left column" was not a number; (ii) the three-value derived design always made exactly two items count, making the text rule redundant; (iii) thresholds could equal a value, leaving an "at least" boundary ambiguity. The final registered endpoint is the sum question with four values and strictly-between thresholds (`-v4` runs); the earlier runs are retained as evidence but are not used for the week-1 numbers. The count variant was added as a measurement-validity check because error inspection showed that many sum failures were arithmetic rather than set selection. It is reported as-is, not substituted for the pre-registered sum endpoint. Both endpoints exclude 0 on derived and include 0 on bridging, so the stop-loss is not triggered; the supported claim is narrowed to conditional filtering of visual values by a textual threshold.


### 7.13 Follow-up R1-R4 (2026-11-15; executed after the week-1 report)

**R1 confirmation.** Freeze v4 and run a separate confirmation set with seed `20261116`, values `5..40`, 60 materials per category, and one template not used in v1-v4: `interval` (text gives a two-sided inclusive interval; image gives four values; text hop asks both limits; image hop asks the four values; full asks the count or sum). Derived/bridging/stitching are unchanged controls. Rerun W1-A (direct) and W1-B (two-step) with 8B; only the confirmation numbers are reported as confirmation. The v1-v4 development ledger is `outputs/mechanism-w1-v1/appendix_versions.md`, with the change reason and the numbers at the time. The W1-C correlation statistic (Spearman) is removed from the comparison and report; the cross-size comparison is descriptive only.

**R2 inline rate.** Inline = the question repeats a material number (digits or number words), a rule phrase, or (MMQA) a 5-word lexical span from the text. Synthetic text redundancy is the programmatic `single_image` proxy; MMQA uses the existing consensus `text_suff`/`both_suff`. Report by synthetic category and model size, and by MMQA condition; report text redundancy separately for inline and non-inline questions. No judge is called.

**R3 interventions.** (a) `plain-refer-v2`: do not repeat numbers/thresholds/rule wording, refer to them instead, keep proper names and entity labels. (b) Numeric suppression: a per-request logits processor that sets every digit token appearing in the material text to `-inf`. Compare with the original generation on the synthetic set using the grammar dual rate and a local validity check (`local-validity-v1`, Qwen3-VL-8B). The local validity check is a local AI review, not the official cascade evaluator. No head-level ablation before R2 results; none was run.

**R4 external validity.** Take the 25 MMQA dual candidates where Qwen had no TI answer (`s1_g3b.md` section a), annotate operation types (text_rule_image_value, cross_modal_comparison, text_fact_image_entity, image_attribute_text_entity, temporal_or_numeric, other_or_invalid), and report the share of text-rule-on-image-value with a Wilson interval. Annotation is AI (luna first pass + manual operational review), not human; the strict luna count is reported as a sensitivity bound.

**Boundary.** R1 changes the mechanism target to the two-sided interval rule; the single-threshold count effect is not confirmed. R3(a) is a prompt-level candidate fix and must be checked with the official evaluator and on MMQA before being called a generation solution. R3(b) should not be pursued at the token level because the model bypasses digit suppression with number words. R4 limits external validity: the synthetic operation is not the dominant MMQA failure.


### 7.14 E1 redundancy-source decomposition (2026-11-15; screen run seen, judged run pre-registered before its results)

**Question.** Of the questions already judged single-modality redundant, how many are *leaks* (the generator copied the unnecessary side's content into the question) and how many are *anchors* (nothing was copied, the question simply asks for something the retained side already contains)? This decides whether a copy-circuit generation intervention (E4) or the self-verification route (E3) carries the paper's method.

**Definitions.** `direction` names the side that alone suffices, so `direction=image` means the text is redundant; the redundant sides are `redundant_sides(direction)` (`image`→text, `text`→image, `both`→both). Per redundant side the AI judge returns one of `leak` / `anchor` / `other`; the question-level label is `leak` if any redundant side is a leak, `anchor` if all are anchors, else `other`. `other` covers degenerate, underspecified, outside-knowledge, and wrong-judgment cases. Judge labels are AI, not human review, until the manual sheet is filled.

**Inputs (all existing labels; no new model generation).** ScienceQA expand (`qwen3vl8b-cand-expand-v2`, legacy cascade labels) and confirm (`qwen3vl8b-cand-confirm-v2` consensus); MMQA plain control, SAQG, and refer-v2 (`vqag-mmqa-pvs60-*`, two-judge consensus; treatment `vqag-mmqa-refer-v2-label-v1`); synthetic v4 8B direct (`synth-comp-qwen3vl8b-direct-v4`, programmatic `single_image`/`single_text` labels). Units are unique `(material_id, question)` pairs; duplicate rows are counted but do not multiply the denominator; `undetermined` and non-redundant units are excluded from the source split and reported separately.

**Programmatic screen (necessary, not sufficient; upper bound for the text side).** Text redundant → leak if the question repeats a material number or a 5-word lexical span from the text; image redundant → leak if the question contains a number absent from the text (an OCR-free proxy; lower bound because we cannot read the image text). The screen is validated where the answer is known: on the synthetic v4 derived/stitching families the 8B generator copies the threshold in 100% of redundant questions.

**Judge pass.** `gpt-6-luna`, `task=audit_question`, one call per redundant side; the image is attached only for the image side; the prompt states which side is unnecessary and forbids counting incidental mentions. Shared cache `outputs/cache/vqag/`, `--max-calls 1600`, `--concurrency 8`.

**Decision rules, fixed before the judged numbers were read.** (a) If the leak share among redundant questions on MMQA is below ~0.20 under the judge label, E4 is demoted to a synthetic-set analysis and the method section rests on E3 (repaired self-verification). (b) If the leak share is high but concentrated in the plain control and reversed in SAQG, the copy mechanism is reported as generator-specific rather than as the MMQA failure mode. (c) A 50-row stratified manual sheet (`outputs/<run>/manual_sheet.jsonl`) must be filled before any leak share is quoted as a paper number; until then every share is prefixed "AI-labelled".

**Outputs.** `source_units.jsonl` (per-unit screen and judge detail), `source_summary.json`, `source_report.md`, `manual_sheet.jsonl`. Command: `redundancy-source --dataset <label>=<kind>:<labels>:<materials>` (repeatable).

**Outcome (2026-11-15, `outputs/e1-redsrc-v1/`, 593 new calls, 898 cache hits, 1.15M tokens).** Leak share among redundant questions: synthetic v4 `167/167 = 1.000`; ScienceQA expand `72/552 = 0.130` `[0.105, 0.161]`; confirm `15/116 = 0.129` `[0.080, 0.202]`; MMQA SAQG `14/107 = 0.131` `[0.080, 0.208]`; MMQA plain control `9/103 = 0.087` `[0.047, 0.158]`; MMQA refer-v2 `5/122 = 0.041` `[0.018, 0.092]`. Everything else is `anchor`; the judge returned `other` zero times, which the manual sheet must test. Leak concentrates in the `both_suff` cells (control `7/11`, refer-v2 `5/13`, SAQG `5/18`, sq-expand `65/235`) and is negligible inside image-redundant units (control `2/89`). The programmatic screen is a weak proxy in both directions (sq-expand `0.049` vs judge `0.130`; MMQA control `0.136` vs judge `0.087`; refer-v2 `0.131` vs `0.041`), so only judge labels are quoted and the screen stays a diagnostic. Decision rule (a) fires: MMQA leak share is below `0.20`, so E4 becomes a synthetic-set analysis and E3 carries the method. Redundancy mix on decided units confirms the real-data failure is anchoring, not copying: MMQA control image redundancy `0.806` vs text redundancy `0.113`, refer-v2 `0.835` vs `0.137`; ScienceQA expand is closer to balanced (`0.553` vs `0.592`).

**Generalization outcome (2026-11-15, after R1-R4; redundancy fields corrected on 2026-11-15 after E1).** The MMQA feasibility test of `plain-refer-v2` **failed**: two-judge consensus dual `0.136 -> 0.072`, official cascade dual `0.064 -> 0.028`, inline `0.322 -> 0.467` (`+0.144 [+0.044, +0.250]`), image redundancy `0.812 -> 0.895` (`+0.083 [-0.006, +0.181]`), text redundancy `0.104 -> 0.124` (`+0.020 [-0.044, +0.087]`); only answerability improved slightly (`0.819 -> 0.856`). Earlier versions of this paragraph reported the text-sufficiency share as "text redundancy"; that inverted `docs/necessity_formal_argument.md` §9 and is superseded. The prompt is a synthetic threshold-copying fix, not a general cross-modal composition fix; on MMQA it increases copying and makes the question answerable from the text alone. The interval mechanism probe found the text interval **is used**: text ablation changes 24/24 answers, image-only reproduces 0/24 wrong counts, and the wrong count locks in between layers 22-26 while the correct count never becomes decodable; the defect is boundary precision, not attention routing. Attention knockout is not the high-value next step; a per-bound comparison prompt is. The interval result remains a controlled synthetic probe (R4: 6/25 hard MMQA duals are rule-on-value). The measurement paper remains the fallback.

### 7.15 E2a composition grid (2026-11-15; design and decision rules written before the n=100 run)

**Question.** Is the cross-modal composition gap a *binding* deficit (the two facts come from different modalities) or a *numeric/aggregation* deficit (the values must be read and combined)? Pre-registered contingency: if the all-text cell shows the same pattern of failure, the mechanism wording changes to numeric aggregation/precision and E3 (repaired self-verification) is untouched.

**Design.** One material supplies four distinct item values and a threshold strictly between the two smallest, so exactly three items always count. The same values, threshold and questions are rendered in four cells: `TT` both facts in text, `TI` rule in text + values in an image table, `IT` values in text + rule in an image strip, `II` both in a single image (rule strip stacked over the table). Cells are paired by material, so the contrast is within-material. Five question types: `full` = "Which items count?" (the conditional judgment), `count`, `sum` ("What is the total value of the items that count?"), `rule`, `values`. Scoring is deterministic: order-insensitive letter sets for `full`, exact integers for `count`/`sum`, order-insensitive value lists for `values`. The prompt (`grid-answer-v1`) is modality-neutral and receives an empty text block in `II`, so no cell is advantaged by wording.

**Readouts, fixed in advance.** (a) per-cell accuracy with Wilson intervals; (b) paired material-bootstrap contrasts `cross − same`, `TI − TT`, `IT − TT`, `II − TT`, `IT − TI`; (c) a psychometric curve of per-item conditional accuracy against the signed distance `value − threshold`, split by cell, with the miss/extra direction reported per bucket; (d) the error anatomy of `sum` (arithmetic vs listing the values instead of summing).

**Decision rules.** If `cross − same` excludes zero but `II` is at or above `TT`, the mechanism is *not* binding and the wording becomes aggregation/precision. If the all-text cell collapses on `sum` as well, the aggregation defect is modality-independent. If the near-threshold buckets are not worse than the far ones, the sharp boundary-width defect seen in the interval probe is specific to interval endpoints and must not be generalised to threshold comparisons.

**Outcome (2026-11-15, `outputs/grid-e2a-v3/`, 100 materials, 2,000 answers, Qwen3-VL-8B greedy, no API calls).** Conditional judgment: `TT 0.840`, `TI 0.810`, `IT 0.890`, `II 0.990`; `cross − same` `−0.065` `[−0.120, −0.010]`, `II − TT` `+0.150` `[+0.080, +0.230]`, `IT − TT` `+0.050` `[−0.030, +0.130]`, `TI − TT` `−0.030` `[−0.140, +0.080]`. Psychometric curve: below threshold `≤−4`, `−3`, `−2` all `1.000`, `−1` `0.957` (7 extras); above `+1` `0.950`, `+2` `0.914`, `+3` `0.917`, `≥+4` `0.987`, errors are omissions (9/12/10). `missed_one_A` is 17 of 47 full-question failures and per-item accuracy rises from A `0.950` to D `0.988`. `Sum`: `TT 0.580`, `TI 0.320`, `IT 0.430`, `II 0.230`, with 100% single-hop accuracy in every cell; `arithmetic_error` 157, `listed_values_not_summed` 82, correct 156. The n=40 replication `grid-e2a-v2` reproduces the ordering with wider intervals. Decision: the binding wording is withdrawn, the mechanism is aggregation over read values (worst when the values come from an image), the interval boundary-width defect stays interval-specific, and E3 remains the main line. The `count` question is not usable as a dependent variable (format violations and cell-flipping off-by-one answers).

### 7.16 E3 repaired self-verification (2026-11-15; rule and pools fixed before the runs)

**Question.** Can the local 8B act as the verifier that PVS currently buys from an API — select iff the material-level answer is correct and both single-side answers are wrong — and does repairing its composition ability raise recall? The paper's prediction was that unrepaired recall would be very low because genuine dual questions fail under TI and get discarded, and that repair would lift recall toward API PVS.

**Design.** `src/egvqg/self_verification.py`; CLI `self-verify-run`, `self-verify-report`, plus `pvs-select` (the API PVS selection stage, previously only callable from Python). Rule: `TI correct and T wrong and I wrong`. Isolation is by construction — the removed side is not passed at all (`T` receives no image, `I` receives an empty text block), not masked. Tiers: `direct` (`composition-answer-v1`) and `two-step` (`composition-two-step-v1`, state rule → state values → combine), the behavioural repair. Gold is computed from the synthetic material: exact normalised match against the three designed questions first, then rules for threshold / which-items / count / sum / item value / alpha; anything uncovered is dropped and counted (`dropped.jsonl`), never guessed. `self-verify-report` re-scores from `raw_output`, applies the rule per unique `(material_id, question)`, and can score an API PVS `selection.jsonl` on the same questions.

**Pools, fixed in advance.** Pool A = the 8B's own generations (`outputs/synth-comp-qwen3vl8b-direct-v4/generated.jsonl`, 720 rows → 231 unique questions: 64 `dual_bridging`, 167 `single_image`). Pool B = the 60 designed aggregation duals (`evaluation.full` of the `derived` materials), i.e. the dual type where the composition gap lives; pool B questions are template-designed, not generated, and this is reported as a limitation rather than hidden.

**Readouts.** Per-tier accuracy under TI/T/I, selection recall, precision, false-positive rate on non-duals, per-label recall, wiring (`selected` / `ti_wrong` / `single_side_answers` / `incomplete`) split by dual vs non-dual, and a paired material bootstrap of the tier recall difference. API PVS is scored on the identical question set for a head-to-head.

**Outcome (2026-11-15, no judge calls).** Pool A: `direct` TI `0.904`, T `0.011`, I `0.575`, recall `1.000 (64/64)`, precision `0.831 (64/77)`, FPR `0.078 (13/167)`; `two-step` TI `0.894`, recall `1.000`, precision `0.889 (64/72)`, FPR `0.048`; API PVS `1.000 / 1.000 / 0.000`, 336 calls. Pool B: `direct` TI `0.517`, T `0.000`, I `0.033`, recall `0.483 (29/60)`, precision `1.000`; `two-step` TI `0.850`, recall `0.850 (51/60)`, precision `1.000`; API PVS recall `0.983 (59/60, 1 undetermined)`, 140 calls; paired `two-step − direct recall` `+0.367` `[+0.217, +0.517]`. Decision: verifier recall factorises as P(TI correct) × P(both single sides wrong), so it is bounded by answerer TI accuracy. The pre-registered "unrepaired recall is very low" holds only on aggregation duals; on the model's own bridging duals repair buys precision instead of recall, because those questions are already TI-easy. The method claim becomes: self-verification is viable locally, and its coverage on hard duals is exactly what answerer repair improves. Mechanistic (patching) repair and the real-data tier remain open; the latter needs a reference-answer budget.

### 7.17 E3 repair ladder, early stop, coverage, MMQA real tier (2026-11-15; tiers and readouts fixed before the runs)

**Question.** §7.16 located the verifier's limit in the answerer's full-material (TI) accuracy. If the defect is the *comparison* step rather than the arithmetic, then making the comparison explicit per item — without moving the arithmetic out of the model — should lift TI to ceiling and recall to API level. How far up the ladder does that go, what breaks, and does the same signal transfer to real data?

**Tiers, fixed in advance (all share the selection rule `TI correct and T wrong and I wrong`).**
- `direct` — one answer from the joined material (`composition-answer-v1`).
- `two-step` — state the condition, state the values, then combine (`composition-two-step-v1`).
- `delegated` — the model returns the *qualifying* values and the rule it applied; the harness sums them (`composition-delegated-v1`). Delegates arithmetic only.
- `delegated-compare` — the model returns the rule, the threshold and the full item→value table; the harness performs the comparison and the aggregation (`composition-delegated-compare-v1`). Delegates the comparison.
- `per-item` — the model returns one row per item with its value and a `counts` flag, plus the rule and threshold; the harness aggregates the model's own flags (`composition-per-item-v1`). Delegates nothing but the bookkeeping.

`per-item` is the mechanism-targeted tier under the E2a/E2b reading (the defect is applying `at least N` at each item), while `delegated*` are the controls that ask whether *any* externalisation helps. TI failures are attributed by searching the below-threshold / qualifying subsets whose sum equals the error: `included_threshold_minus_one`, `included_other_below`, `dropped_qualifying`, `other`, `unparsed_or_item_set`.

**Readouts.** Per-tier TI/T/I accuracy and recall/precision/FPR; a paired material bootstrap for each consecutive tier pair; the TI error taxonomy; **coverage** (share of materials with at least one selected dual, mean selected, unique questions per material) and a k-sample `sampling_curve`; early-stop savings.

**Early stop.** Answer all TI slots first, then the single sides only for TI-correct groups. A wrong TI is a decided rejection (not undetermined), so recall cannot rise; the accuracy of T/I is then measured on the TI-correct subset and the report says so. The saved cost is `2·P(TI wrong)` answers per question.

**Outcome (2026-11-15, `outputs/self-verify-derived-{v2,v3,compare-v1}`, `self-verify-synth-v2`, `self-verify-es-check-v1`; no judge calls).** Pool B recall by tier: `direct 0.483` → `delegated 0.783` → `two-step 0.850` → `delegated-compare 0.933` → `per-item 0.983 (59/60)`, precision `1.000` throughout; `per-item − direct` `+0.500` `[+0.367, +0.633]`, `two-step − per-item` `−0.133` `[−0.233, −0.050]`. Per-item TI is `1.000 (60/60)` against API PVS recall `0.983` at zero API calls. The taxonomy explains why: 10 of delegated's 13 errors include the value exactly one below the threshold and then compute the sum correctly, i.e. `at least N` is applied as `at least N−1` — the question-level form of the E2b width-≈-1 defect; `per-item` has zero such failures. Pool A (`self-verify-synth-v2`, 6,480 answers): `direct` precision `0.865`/FPR `0.060`, `two-step` `0.914`/`0.036`, `per-item` `0.985`/`0.006`, recall `1.000` in all three — on easy duals the repair converts into precision because both single sides become reliably correct. Early stop: 124 instead of 180 answers (−31%), recall `0.500` vs `0.483`. Coverage: 60/180 materials, mean selected 0.36–0.41 with 1.28 unique questions per material, so **generator dual supply is now the binding constraint**; `_sampling_curve` prices a best-of-n budget and needs a `sample_index` field in the pool.

**MMQA real tier (`self-verify-mmqa-v1`).** 120 sampled rows → 89 unique questions, 7 consensus duals, 6 undetermined; references `gpt-6-luna` (`solve-v1`), equivalence `gpt-5.6-terra` (`answer-equivalence-v1`), 408 API requests. Local TI `0.843`, T `0.795`, I `0.120`; recall `0.143 (1/7)` `[0.026, 0.513]`, precision `0.250 (1/4)`, FPR `0.039`. The rule behaves correctly (T acceptable on `text_suff` questions `50/54`, which independently validates those labels); the failure is n and refusals — 5 of 7 duals are refused under every condition because they are open-ended "significance/connection" questions, and 2 have no reference. One material was blocked by provider image content policy (400; non-2xx bodies and a collapsed `error_detail` are now recorded in the call ledger). Decision: report this as an **underpowered negative transfer**, not a verifier failure; scaling it needs ≥4–6 unique questions per material and a selection rule that treats refusal as undetermined rather than wrong.

**Files.** Module `src/egvqg/self_verification.py`; prompts `composition-{delegated,per-item,delegated-compare}-v1`, `solve-v1`, `answer-equivalence-v1`; CLI `self-verify-run --early-stop`, `self-verify-real-run`, `self-verify-real-report`, `pvs-select`; tests `tests/test_self_verification.py` (12 tests).


### 7.18 E3b best-of-n supply check (2026-11-15; prediction fixed before the run)

**Question.** §7.17 shows the verifier is no longer the constraint: with 1.28 unique questions per material, only 33% of materials yield a selected dual. Is that a *sampling* deficit (more draws per material would surface a dual) or a *shape* deficit (the generator only has one dual shape)?

**Prediction fixed in advance.** If dual yield grows with k, buy supply with samples and report a priced best-of-n budget. If `materials_with_dual` is flat in k, supply is a generator property and must be addressed on the generation side.

**Design.** `outputs/bon-synth-v1/materials.jsonl` (60 materials, 20 each `bridging`/`derived`/`stitching`, seed 20261115), 8 samples per material with `plain-v2` at temperature 0.7 (the same protocol as the v4 pool), then `self-verify-run --tiers direct,two-step`. `_sampling_curve` reports per-k questions, materials with a dual, materials with a *selected* dual, selected count, precision and raw dual rate; it requires `sample_index` in the pool, which `composition-generate` writes.

**Outcome (2026-11-15, `outputs/bon-synth-v1/`, `outputs/bon-synth-verify-v1/`, 480 + 2,880 local answers, no API calls).** 480 generations → **86 unique questions**, 1–3 distinct per material (`single_image` 63 unique / 320 samples, `dual_bridging` 23 unique / 160 samples). Dual-bearing materials: **20/60 at k=1, 20/60 at k=8**, all `bridging`; `derived` and `stitching` never produce a dual — the model rewrites the rule into the question ("Which items have a value of at least 4?"), i.e. the leakage E1 measured. Curve: questions 59→86, selected 24→29, precision 0.833→0.793 (direct) / 0.958 (two-step, 24 selected), raw dual rate 0.339→0.267. Decision: the prediction fires on the flat branch — best-of-n buys +27 unique questions and +5 selected duals for 8× the cost, so **sampling is not an optimisation lever**; the binding constraint is a mode-collapsed generator with a single dual shape. Consequence for the plan: the next generation-side attempt on the synthetic set is a circuit-level intervention on the copied threshold (E4), not a larger k, and the paper must not present coverage as an inference-time budget question.

### 7.19 E4 copy-source circuit: attribution, head ablation, attention blocking (2026-11-16; readouts fixed before the runs)

**Question.** E1 puts 100% of the synthetic threshold-set redundancy on `leak`, and §7.18 shows the generator is mode-collapsed on that leak. Is the copied threshold a localizable copy circuit that can be attenuated during generation, or is the leak a property of the question schema?

**Design (fixed before the runs).** `data/processed/synthetic-composition-v1/materials.jsonl`, category `derived` (the text holds only the rule, the four values live only in the image, so a numeral in the question can only come from the text), `plain-v2`, 20 materials, greedy 24 tokens for attribution/ablation and 4 samples at temperature 0.7 for blocking.
1. `copy-attribute`: attention mass from the first generated numeral token to the context numeral token over all 36 layers × 32 heads; heads ranked by materials-in-top-10 then mean mass.
2. `copy-ablate`: zero the head-output slice for the top 8, top 16, a random 8, or all heads on decode steps only (prefill untouched), then generate.
3. `copy-mask`: additive bias (−1e4) forbidding generated positions from attending to the context numeral tokens, or to the whole text span.
Readouts fixed in advance: copy rate (`contains_number(text)`), threshold mention, `classify_generated_question` label, unique questions, and the **faithful-threshold rate** (first numeral equals the material threshold).

**Decision rule.** E4 is worth building into the method only if an intervention reduces the copy rate *and* leaves (i) a faithful rule or (ii) a genuine dual question. A copy rate that falls while questions become unfaithful or image-only is a failure of the intervention, not a partial success.

**Outcome (2026-11-16, `outputs/copy-heads-v1/`, `copy-ablate-all-v1`, `copy-mask-e4-v1`; local GPU only, no API calls).** Attribution finds the copy in 20/20 materials; top heads L02H31 19/20, L29H11 17, L28H23 15, L28H00 14, L20H08 13. Head ablation is a null: top-8, top-16 and random-8 leave every question byte-identical (copy 1.000, dual 0.000), while ablating all heads destroys the model, so the copy is redundant across heads. Attention blocking works at the mechanism (one numeral token at prompt position 172, bias applied) and removes the copy (**1.000 → 0.000**), but the first numeral is then the true threshold in **0/80** questions (10 in 44, 5 in 12, 9 in 8, 13 in 8, 12 in 4, 16 in 3, 15 in 1), the question still suffices from the image alone (single-image 1.000, dual 0.000) and is unfaithful. Blocking the whole span removes the criterion (threshold mention 0.000, "Which is the largest value?", single-image 0.963) and the model starts answering its own question in the output. **Decision: E4 closes negative** — the copy is load-bearing for the model's chosen schema, and no attention-level intervention on the redundant side can produce a dual, because a dual must use the rule without quoting it. This is the mechanistic counterpart of the E1 gate (MMQA leak < 0.20) and it points the generation-side lever at schema control rather than circuits.

**Bug fixed on the way.** `_KeyBlock.bias()` built its row slice as `slice(query_start, query_len)`; under cached decoding HF passes one query row per step, so the slice was empty, `bias()` returned `None`, and two earlier runs silently compared identical conditions. The decode-step branch is now explicit, verified against a single-material prompt (one numeral token at position 172), and covered by `tests/test_copy_circuit.py` (`bias(1, 8)` applies at the key positions, `bias(8, 8)` is `None`).

**Files.** Module `src/egvqg/copy_circuit.py`; CLI `copy-attribute`, `copy-ablate`, `copy-mask`, `copy-mask-report`; tests `tests/test_copy_circuit.py` (4 tests, including the decode-step mask and the merged three-condition report). Offline suite 179 passed.

**Open.** Masked-threshold faithfulness is 0/80 at 20 materials × 4 samples; a 60-material replication is needed before publishing a rate. Path patching on the top heads was not run (the ablation null deprioritises it). The only generation-side lever the negative leaves open is schema control — a prompt or constraint that requires using the rule without restating it — which must be tested on the same face-valid readouts before any real-data work.

### 7.20 Local text-answerability screen vs API PVS on real MMQA (2026-11-16; rule and readouts fixed before the run)

**Question.** E1 shows that MMQA redundancy is single-source anchoring (leak ≤ 13.1% lenient, ≤ 3.6% strict), and §7.17 shows the repaired self-verifier is the wrong instrument there (89 unique questions, 7 duals, recall 1/7). The matching question is the *anchoring* one: **does the passage alone already answer the question?** Can a purely local screen of that form — no reference answers, no judge, no API — match or beat the API PVS selector on real data?

**Design (fixed before the run).** Universe: the 149 unique `(material, question)` pairs of `outputs/vqag-mmqa-pvs60-control-label-v1/candidates.jsonl` (360 rows, 60 materials; `prepare_real_pool` now deduplicates and treats `sample_per_material<=0` as all rows). Gold: the existing two-judge consensus `label_consensus.verdict == 'dual_necessary'` (21 duals, base rate 0.141). Local answerer Qwen3-VL-8B; conditions `TI` and `T`; the text-only material is passed **without** the image.
- Rule `reject_if_text_answers` (primary): select iff the model **fails/abstains** on `T` and answers on `TI`.
- Rule `reject_if_text_answers_no_ti` (robustness): the same without the `TI` requirement.
- Comparison: API PVS `pvs-select` (`gpt-6.1-sol`) on exactly the same 149 questions.
- Readouts fixed in advance: recall, precision, FPR on the 128 non-duals, material-clustered bootstrap of precision and of lift over the base rate, coverage, and the local/PVS intersections.

**Command.**
```bash
E=".venv/bin/python -m egvqg"; M=/hpc2hdd/home/fye374/models/Qwen3-VL-8B-Instruct
CUDA_VISIBLE_DEVICES=0 $E answerability-audit --run-id mmqa-answerability-v1 \
  --candidates outputs/vqag-mmqa-pvs60-control-label-v1/candidates.jsonl \
  --materials outputs/mmqa-refer-v2-v1/materials.jsonl --local-model-path $M --device cuda:0 --max-batch 8
$E pvs-select --run-id pvs-mmqa-aa-v1 --candidates outputs/mmqa-answerability-v1/candidates_unique.jsonl \
  --materials outputs/mmqa-refer-v2-v1/materials.jsonl --workers 6
$E answerability-report --run-id mmqa-answerability-v1 --bootstrap 10000 \
  --pvs-selection outputs/pvs-mmqa-aa-v1/selection.jsonl
```

**Outcome (2026-11-16, `outputs/mmqa-answerability-v1/`, `outputs/pvs-mmqa-aa-v1/`).** 298 local answers (~5 min GPU, **0 API calls**) vs 149 PVS calls. Local text-answered rate 0.846 (126/149). PVS verdicts: text_suff 111, image_suff 6, stitching 15, dual 16, undetermined 1.

| selector | selected | recall | precision | FPR |
|---|---:|---:|---:|---:|
| local text-answerability | 23 | **0.667 (14/21)** [0.454, 0.828] | 0.609 (14/23) | 0.070 (9/128) |
| API PVS | 16 | 0.476 (10/21) [0.283, 0.676] | 0.625 (10/16) | 0.047 (6/128) |
| local ∧ API | 10 | 0.381 (8/21) | **0.800 (8/10)** | 0.016 (2/128) |
| local ∨ API | 29 | **0.762 (16/21)** | 0.552 (16/29) | 0.102 (13/128) |

Material-clustered bootstrap: local precision **0.609 [+0.385, +0.750]**, lift over the base rate **+0.468 [+0.296, +0.611]**. Coverage 11/60 materials (18.3% [10.6, 29.9]) at 0.383 selections per question. Both rules select the same 23 because `TI` abstains on only 8/149 and none of those overlap the 23 text-abstentions, so the signal is effectively "text-only fails". The two selectors are complementary: the intersection is the high-confidence set, the union raises recall to 0.762. PVS's extra rejections are mostly its 15 `stitching` calls (it penalizes questions that look like two independent sub-answers), while the local screen only asks whether the passage suffices.

**Decision (revised after §7.20b).** Keep the local answerability gate as the real-data selector; it is derived from the E1 diagnosis, costs zero API calls, and its point estimates are equal-or-better than PVS on both axes. Do **not** claim it is a dual-ness test when only one side is screened: the certificate is "that side alone is insufficient", not "both sides are necessary". Report the two-sided rule as the portable form and the single-side rule as the sharp form when the direction is known.

### 7.20b Two-sided rules and the ScienceQA direction check (2026-11-16)

**Design.** The same audit run now returns three rules, so one run tests the direction claim instead of assuming it: `reject_if_text_answers` (reject when the passage alone answers → detects image redundancy), `reject_if_image_answers` (reject when the image alone answers → detects text redundancy), `reject_if_either_side_answers` (both single sides must fail). `answerability-audit --conditions` now defaults to `TI,T,I`; reports also emit `text_answered_rate`, `image_answered_rate`, and a material-clustered paired bootstrap of each local rule minus API PVS (`_selector_comparison`, precision and recall).

**ScienceQA pool.** `outputs/qwen3vl8b-cand-confirm-v2/labels_consensus_full.jsonl` → 173 unique `(material, question)` pairs (480 rows, 24 duals, one conflicting duplicate), verdicts mapped from `decided/redundant/direction` to the formal labels.

**Outcome.**

| dataset | rule | selected | recall | precision | FPR | T answers | I answers |
|---|---|---:|---:|---:|---:|---:|---:|
| MMQA (149 q, 21 duals) | text | 23 | 0.667 | 0.609 | 0.070 | 0.846 | 0.195 |
| MMQA | image | 120 | 0.905 | 0.158 | 0.789 | | |
| MMQA | either-side | 15 | 0.571 | **0.800** | 0.023 | | |
| MMQA | API PVS | 16 | 0.476 | 0.625 | 0.047 | | |
| ScienceQA (173 q, 24 duals) | text | 105 | 0.708 | 0.162 | 0.591 | 0.393 | 0.590 |
| ScienceQA | image | 71 | 0.750 | 0.254 | 0.356 | | |
| ScienceQA | either-side | 30 | 0.458 | **0.367** | 0.128 | | |

The answered rates invert between datasets (MMQA text-anchored 0.846/0.195; ScienceQA image-anchored 0.393/0.590), and the single-side rule's precision follows the direction (0.609 → 0.162), so the audit doubles as a direction diagnostic. The two-sided rule stays above the base rate in both (lift on ScienceQA +0.228 [+0.036, +0.455]). Against PVS on the same MMQA questions the paired bootstrap still includes zero (precision [−0.019, +0.475], recall [−0.222, +0.421]), so the claim is bounded to "equal-or-better point estimates at zero API cost and 3 local answers per question".

**Open.** No PVS run on the ScienceQA pool; the two-sided rule costs recall on both datasets; a schema- or length-conditioned rule may recover it; every interval is wide at 21–24 duals.

**Files.** `src/egvqg/self_verification.py` (`run_answerability_audit`, `report_answerability_audit`, `_answered`, `_material_bootstrap_answerability`, `_interval`); CLI `answerability-audit` (`aar`), `answerability-report` (`aap`); test `test_answerability_audit_scores_text_screen` in `tests/test_self_verification.py`; offline suite **180 passed**.

**Open.** One dataset and 60 materials with AI-consensus labels; recall interval wide at n=21; the 14 true positives and 9 false positives have no human review yet; ScienceQA replication not run.

### 7.21 Completing the selector tables, blind control, and the performance law (2026-11-16; readouts fixed before the runs)

**Question.** The gate's table was MMQA-only, without a ScienceQA judge control, without a sampling control, and without a control for prior knowledge. Can the table be completed so that the method line is self-consistent, and can the gate's operating point be derived rather than described?

**Design, fixed before the runs.**
1. API PVS on the same 173 ScienceQA questions (`pvs-sq-aa-v1`), so the local gate and the judge are compared on one population per corpus.
2. A **blind condition** (`--conditions ...,0`: neither text nor image) added to both greedy pools. Prediction to test: if the level-3 "false positives" are questions answerable from prior knowledge, then on genuine duals the blind condition should answer at a rate close to the level-3 agreement rate.
3. A 3-sample, temperature-0.7 replication on both corpora (`*-answerability-v2`) to test whether the support/stability levels move the frontier favourably.
4. Derive the gate's recall and precision as functions of two measurable rates and check the derived expression against the observed selections; check the two-sided independence formula against the observed dual rejection rate.

**Outcome.**

| corpus | rule | selected | recall | precision | FPR |
|---|---|---:|---:|---:|---:|
| MMQA (149 q, 21 duals) | text | 23 | 0.667 | 0.609 | 0.070 |
| MMQA | either side | 15 | 0.571 | 0.800 | 0.023 |
| MMQA | API PVS | 16 | 0.476 | 0.625 | 0.047 |
| ScienceQA (173 q, 24 duals) | either side | 30 | 0.458 | 0.367 | 0.128 |
| ScienceQA | API PVS | 44 | **0.958** | **0.523** | 0.141 |
| ScienceQA | either side ∨ PVS | 53 | 0.958 | 0.434 | 0.201 |

- **Blind control: hypothesis refuted.** The blind condition answers 18/149 and 19/173 rows overall, but only 1/21 and 0/24 of genuine duals, with 0 and 0 agreements with the full-material answer. Prior knowledge is therefore not the source of the level-3 agreements; the side itself answered, so those rows are label errors or missed derivations. The blind probe is retained as a specificity certificate, not a filter.
- **Sampling is counterproductive** at the support level: three samples replace luck ℓ by 1−(1−ℓ)³; MMQA text-rule `stable` precision 0.349 vs 0.609 at k=1, either-side 0.366 vs 0.786. The gate is a k=1 instrument.
- **The law.** With prevalence π, luck ℓ and sensitivity s, recall = 1−ℓ and precision = π(1−ℓ)/[π(1−ℓ)+(1−π)(1−s)]; the gate beats the base rate **iff s > ℓ**. All four side rules satisfy it (MMQA text 0.930 vs 0.333; MMQA image 0.211 vs 0.095; ScienceQA text 0.409 vs 0.292; ScienceQA image 0.617 vs 0.250), the two-sided independence formula predicts the observed dual rejection within 0.03–0.05, and the recall ceiling holds exactly. ScienceQA's weakness is one number: s_T = 0.409, the answerer cannot read the lecture passages well enough to notice that they already answer the question.
- **Paired bootstrap.** MMQA either-side − API: precision [−0.035, +0.455], recall [−0.222, +0.300]. ScienceQA either-side − API: precision [−0.358, +0.043], recall [−0.778, −0.154]. The cross-corpus claim is therefore "the gate is the cheaper instrument when s_T is high; the judge wins when the answerer cannot read the side".

**Decision.** The method line is: (i) diagnose the redundancy direction and the leak share, (ii) run the gate where a labelled pilot shows s > ℓ on the side that carries the redundancy, (iii) use the repaired local verifier for the positive term. The gate is not claimed to replace the judge universally; the judge remains the correct instrument for corpora where the local answerer's sensitivity is low, and the union rule (gate ∨ judge) is the high-recall fallback (ScienceQA 0.958 at 0.434).

**Theory.** `docs/necessity_formal_argument.md`: §11 (why the device is the definition, the closed loop, the gate as hypothesis test, the exact decomposition, the $s>\ell$ theorem and stop rule, the sampling result, the verifier's feasibility cap, the measurement table) and §10.7 (Propositions 16–17).

**Files.** `src/egvqg/pvs.py` (`_cand_id`), `src/egvqg/self_verification.py` (blind condition, `reject_if_prior_answers`, `reject_if_either_side_or_prior_answers`, `_selector_comparison`, ladder with the `0` side), CLI `--samples/--temperature/--conditions TI,T,I,0`, test `test_answerability_audit_prior_condition_rejects_blind_answers`; offline suite **183 passed**.

**Open.** Human review of the ~28 level-3 false positives (label error vs missed derivation); the 60-material, AI-consensus labels; the pilot-based estimate of s on a new corpus; `story.md`/`VQG_Proposal.md` not yet rewritten.
