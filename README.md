# Evidence-Grounded Visual Question Generation

Educational image-text question generation: given a passage of text and its accompanying image, generate one question that requires jointly using information from the materials. The core question is: **of the questions a model generates, how many truly require using both the text and the image to be answered.**

## Current Status

- **Research route:** probe route P0–P7 ([section 7 of the implementation plan](docs/implementation_plan.md)): where inside the generator, and on what grounds, the decision is made about which modality the question depends on; only question Q is produced, answer A is out of scope. See [PROGRESS.md](PROGRESS.md) for progress, numbers, and items pending cleanup.
- **Two protocols:** the old protocol `plain-v1` (Chinese, with wording such as "educational text", Qwen3-VL-8B, transformers 4.57.6); the new protocol `plain-v2` (a fixed neutral English prompt, unified transformers 5.22.2, with the same criteria). Data and conclusions from the old protocol are retained and labeled as the old protocol, and are not mixed with new results.
- **Evaluation results under the old protocol (ScienceQA test, 1,257 materials):** Qwen3-VL-8B's baseline has a verified dual-necessity lower bound of 19 questions (1.5%); luna's baseline has a verified lower bound of about 23/1257 (1.8%). See `outputs/qwen3vl8b-necessity-v2/probe_report.md` and `outputs/necessity-v3-full-v1/probe_report.md`.
- **Redundancy sources (E1, 2026-11-15):** copying the unnecessary side into the question (`leak`) explains **100%** of the synthetic threshold-set redundancy but only **4–13%** of real redundant questions (ScienceQA expand 13.0%, confirm 12.9%, MMQA SAQG 13.1%, plain control 8.7%, refer-v2 4.1%); the rest are `anchor` questions whose answer the retained side already contains. On MMQA the dominant failure is text anchoring (image redundancy 0.81–0.84). AI labels only; the 50-row manual sheet is pending. Artifact: `outputs/e1-redsrc-v1/source_report.md`.
- **Reporting categories are only two kinds:** verified dual-necessity and the remainder. "The remainder" does not mean all are unimodal questions: it includes questions that can be answered unimodally, questions that do not pass on their own terms, and questions not yet adjudicated, listed separately.
- **English evaluator translation validated:** re-evaluating the 194 questions of the old pilot with the English evaluator gave consistent redundancy labels on 142/152 (0.934, Wilson 95% lower bound 0.883), see `outputs/qwen3vl8b-cand-pilot-en-eval-v1/translation_validation.md`.
- **Not yet done:** all formal data and G1 adjudication under plain-v2, a fair comparison of the proposed generation method, and overall repeated-run evaluation stability.

## Documentation Entry Points

| Document | Purpose | Status |
|---|---|---|
| [VQG_Proposal.md](VQG_Proposal.md) | Research definitions: task, metrics, evaluation, and experiment boundaries | Research definitions follow this document |
| [story.md](story.md) | Paper narrative: research questions, section skeleton, proposed contributions | Revised as research progresses |
| [docs/necessity_formal_argument.md](docs/necessity_formal_argument.md) | **Mathematical reference**: definitions, propositions, and proofs of bimodal necessity; Section 9 is the evaluation implementation convention | Current |
| [docs/references.md](docs/references.md) | **Paper and data reference** index, with an index of mathematical propositions | Current |
| [docs/implementation_plan.md](docs/implementation_plan.md) | Implementation route and stage acceptance | Current |
| [PROGRESS.md](PROGRESS.md) | Sole progress log: current status, old-protocol results, milestones, pending cleanup, work log | Current |
| [AGENTS.md](AGENTS.md) | Project conventions for the coding assistant to follow | Current |

## Environment and Offline Checks

The project uses a single `.venv/` and `uv.lock`. By default it does not call real APIs, download data, or train models.

```bash
uv sync
.venv/bin/python -m pytest          # offline tests, all using mocks
```

The container is limited to 8 CPU cores, while `os.cpu_count()` still reports 64. Importing `egvqg` therefore caps OMP/MKL/OpenBLAS/NumExpr and torch at 8 threads; override with `EGVQG_NUM_THREADS=<n>`. Without the cap, torch defaults to 64 threads and BLAS/OMP thrashing makes state loading roughly 70× slower. Ad-hoc scripts must import `egvqg` (or call `egvqg.configure_threads()`) before importing numpy/torch.

If the default `uv` cache directory on this machine is not writable, first designate a writable cache, for example `UV_CACHE_DIR=/tmp/uv-cache uv sync`.

### Model and Gateway Configuration

Model calls read `.env` or environment variables of the same name (environment variables take precedence, and `.env` does not enter Git):

| Variable | Meaning |
|---|---|
| `LLM_JUDGE_MODEL_NAME` / `LLM_JUDGE_BASE_URL` / `LLM_JUDGE_API_KEY` | Default model, API root path (or complete `/chat/completions` or `/responses` address), key |
| `LLM_JUDGE_API` | `auto` (default), `responses`, `chat_completions`; auto selects Responses for `gpt-*`/`luna*` and Chat Completions for everything else |
| `LLM_JUDGE_CONNECT_IP` / `LLM_JUDGE_LOCAL_ADDRESS` | Optional: fixed TCP target address and source address (needed by intranet gateways); the URL, Host, and TLS/SNI still use the original domain names and certificates are still validated, and on failure the network is not switched silently |
| `LLM_JUDGE_TIMEOUT` / `LLM_JUDGE_MAX_RETRIES` | Timeout in seconds and the number of technical retries |
| `LLM_JUDGE_CONCURRENCY` | Concurrency cap (used by some entry points) |

Model names on the gateway are case- and hyphen-sensitive (for example `DeepSeek-V4.1-Flash`); `/v1/models` can list the names actually available. Model outputs, adjudication rationales, calls, and billing uncertainties are recorded in `calls.jsonl`, `requests.jsonl`, and `usage.json` respectively; unknown tokens or costs are recorded as `null` rather than 0. Responses are cached in `outputs/cache/judge/`.

## Workflow and Commands

Commands have the form `.venv/bin/python -m egvqg <command> --run-id <run> ...`; `--run-id`, `--project-root`, and `--output-root` are written after the subcommand, and specific arguments are viewed with `<command> -h`. Rerunning the same `run_id` with the same configuration resumes, filling in only missing records; changing the configuration or prompts requires a new `run_id`.

| Stage | Command | Description |
|---|---|---|
| Data | `download-data`, `fetch-dataset`, `normalize-materials`, `import-mmqa-materials` | ScienceQA/MultiModalQA import and material normalization; see the next section |
| baseline generation | `simple-baseline-init/run/report` | Sees only the hint and the original image, generates questions and performs independent full image-text answering; asynchronous concurrency, with a hard size limit on images. `run` with `--local-model-path <Qwen3-VL directory> --device cuda:0 --max-batch 16` can switch to a local model (single-GPU dynamic batching, designate the GPU with `CUDA_VISIBLE_DEVICES`) |
| Basic evaluation | `simple-baseline-eval-init/run/report` | Can be created only after full-batch generation is complete; separately evaluates answerability/faithfulness, support by the original answer, and dual necessity |
| Basic evaluation (cascade) | `simple-baseline-eval-init --cascade` → `eval-run` → `eval-report` | Validity → unimodal screening → generate counterexamples only for surviving questions; about half the cost of the full pipeline |
| **Necessity evaluation (recommended)** | `necessity-ensemble-init/run/report` | See the next section |
| Candidates and labels | `candidates-capture`, `candidate-source-init/run`, `candidate-labels`, `label-audit-prepare/finish` | P1: stratified material×candidate sampling, white-box capture, writing candidates as baseline source runs, concatenating cascade labels and sampled audit |
| White-box and probes | `whitebox-capture`, `whitebox-compare`, `modality-distances`, `probe-run`, `p5-readout`, `p5-robust`, `geometry-run`, `geometry-summary` | P0/P3/P5: hidden states at three positions, per-layer probes, modality displacement readouts, and geometry |
| Selection and steering | `steer-prepare`, `steer-analyze` | P4 steering screening and G3 adjudication, depending on A-stream labels; P6's `selection_eval.py` is implemented but has no CLI entry yet |
| VQAG diagnostics | `diag-template`, `diag-curve` | 1b/H2 variance shares and centering levels; 1c/H2–H3 probes after none/global/skill/material centering, plus `w_dir.npy` |
| VQAG labels and solutions | `label-repair`, `vqag-select`, `vqag-steer`, `filter-baseline` | Stage-0 evidence labels with arbitration; 2a outlier/balance/min_shift selection; 2b projection clamping with controls (GPU); 2c counterfactual filter baseline |
| VQAG facts | `factsplit` | 1a/P2 fact extraction, A/B split, four-condition materials (render image is OCR-shortcut flagged) |
| Synthetic composition | `synthetic-build`, `composition-answer`, `composition-generate`, `composition-report`, `composition-compare`, `composition-compare-prompts` | Week-1 mechanism route: programmatic derived/bridging/stitching materials with split single-hop questions; local-model answering and dual-question generation; material-bootstrap composition deficits and the both-hops-correct-full-wrong list |
| Composition grid (E2) | `grid-build`, `grid-answer`, `grid-report` | 2×2 modality placement grid (rule text/image × values text/image) on paired materials; programmatic conditional/sum scoring, paired material bootstraps, and a psychometric curve by signed distance from the threshold |
| Self-verification (E3) | `self-verify-run`, `self-verify-report`, `self-verify-real-run`, `self-verify-real-report`, `pvs-select` | answers every candidate under TI/T/I with the local model, scores against computable synthetic gold, and selects iff TI correct and both single sides wrong; reports recall/precision per repair tier (`direct`, `two-step`, `delegated`, `delegated-compare`, `per-item`), coverage and a k-sample curve, the TI error taxonomy, `--early-stop` savings, and optional API-PVS scoring on the same questions; the real-data path builds references with an API model and judges local answers for equivalence |
| Local answerability screen | `answerability-audit`, `answerability-report` | Questions the local model under the full material, text-only, image-only and blind (no material) conditions; scores four rules — reject when the passage answers (detects image anchoring), reject when the image answers (detects text anchoring), reject when either side answers (portable, direction-agnostic), reject when the question is answerable blind (prior knowledge) — with no references, no judge and 0 API calls; reports recall/precision/FPR, per-side answered rates, an evidence ladder, the `s > ℓ` performance check, material-clustered precision/lift and paired differences against API PVS, and coverage; `--samples/--temperature` run the multi-sample control |
| Copy-source circuit (E4) | `copy-attribute`, `copy-ablate`, `copy-mask`, `copy-mask-report` | Attributes the generated-numeral → context-numeral attention over all 36×32 heads, ablates top/random head sets on decode steps, and blocks attention to the context numerals or the whole text span during generation; reports copy rate, threshold mention, faithful-threshold rate, dual and single-image rates. Meets `copy-mask-report` to merge several masked runs into one three-condition report |
| Inline, validity, redundancy source | `inline-report`, `local-validity`, `validity-report`, `redundancy-source` | R2 inline numbers/rules vs text redundancy (synthetic + MMQA); R3 local AI validity review of generated questions; E1 splits every redundant question into `leak` / `anchor` / `other` (programmatic screen + optional judge, plus a manual sheet) |
| Generalization checks | `material-generate`, `mmqa-refer-compare`, `interval-mechanism` | Generic MMQA generation for the refer-prompt feasibility test; behavioural ablations + logit lens for the interval-count failures |
| Accounting | `api-ledger` | Aggregates call counts and known tokens by run-name pattern |

## VQAG Mainline (code built; no real runs yet)

The mechanism/solution route is implemented as CLI subcommands and was only exercised with `--dry-run` on existing confirm-run features. No paid call or GPU job was made in this round. All new commands accept `--dry-run`; API-facing ones also accept `--max-calls` and share `outputs/cache/vqag/` (key = model + prompt + image hash). Intervals are paired bootstraps over parent materials in `stats.py`.

```bash
E=".venv/bin/python -m egvqg"; C=qwen3vl8b-cand-confirm-v2; L=labels_${C}-eval.jsonl
$E diag-template --run-id vqag-dry --capture-run $C --label-file $L --dry-run
$E diag-curve    --run-id vqag-dry --capture-run $C --label-file $L --dry-run
$E label-repair  --run-id vqag-dry --capture-run $C --label-file $L --dry-run   # prints 60/320-parent call estimates
$E vqag-select   --run-id vqag-dry --capture-run $C --label-file $L --dry-run   # old labels only: pipeline check
$E vqag-steer    --run-id vqag-dry --capture-run $C --label-file $L --dry-run   # CPU hook check + GPU/API estimate
$E factsplit     --run-id vqag-dry --dry-run
$E filter-baseline --run-id vqag-dry --capture-run $C --label-file $L --eval-run ${C}-eval --dry-run
.venv/bin/python scripts/sanity.py    # the only sanity check (synthetic, CPU, seconds)
```

## Synthetic Composition Route (week-1 mechanism check)

The controlled set `data/processed/synthetic-composition-v1` has 60 materials per category. `derived` is serial (text rule + image values), `bridging` is serial (text label + image column), `stitching` is parallel (independent text and image hops). Each material carries the full question and two single-hop questions with programmatic answers. `synthetic-build --derived-full count` builds the boundary-clean count variant (`synthetic-composition-count-v1`) that isolates conditional filtering from summation.

```bash
E=".venv/bin/python -m egvqg"
$E synthetic-build --per-category 60
$E composition-answer --run-id synth-comp-qwen3vl8b-direct-v2 --local-model-path /path/to/Qwen3-VL-8B-Instruct --device cuda:0
$E composition-generate --run-id synth-comp-qwen3vl8b-direct-v2 --local-model-path /path/to/Qwen3-VL-8B-Instruct --device cuda:0 --n-questions 4
$E composition-report --run-id synth-comp-qwen3vl8b-direct-v2 --bootstrap 10000
$E composition-compare --run-id synth-comp-cross-size-v2 --runs 2b=<run>,4b=<run>,8b=<run>
$E composition-compare-prompts --run-id synth-comp-prompt-compare-v2 --direct-run <run> --two-step-run <run>
$E synthetic-build --per-category 60 --output-dir data/processed/synthetic-composition-confirm-sum-v1 \
   --categories derived,bridging,stitching,interval --value-min 5 --value-max 41 --seed 20261116
$E inline-report --run-id inline-r2-v1 --synthetic 8b=<run> --mmqa saqg=<candidates.jsonl> \
   --mmqa-materials data/processed/multimodalqa/mmqa-pool-v1/pvs60-materials.jsonl
$E local-validity --run-id validity-orig-8b-v1 --source-run <run> --local-model-path /path/to/Qwen3-VL-8B-Instruct
$E validity-report --run-id validity-orig-8b-v1
```

No judge is used for the composition or inline analyses: answers are scored against programmatic answers, and generated questions get fixed grammar labels. The local validity check is a local AI review, not the official evaluator. Week-1 results are in `outputs/mechanism-w1-v1/week1_report.md`; the confirmation, R2, R3 and R4 reports are under `outputs/mechanism-w1-confirm-v1/` and `outputs/mechanism-w1-v1/`.

## Information Necessity Evaluation Entry Point (Recommended)

Two model groups (default `gpt-6-luna` and `DeepSeek-V4.1-Flash`) each perform unimodal sufficiency screening and counterexample recomputation, and the arbitration model (default `gpt-5.6-terra`) only handles questions that the two groups did not settle; all adjudications are produced by code according to fixed rules; answer equivalence is determined by answer slots generated only from the question stem. See [Section 9 of the mathematical reference](docs/necessity_formal_argument.md) for definitions and conventions. A completed `simple-baseline-eval` run is required as the source beforehand.

```bash
R=necessity-v3-full-v1; E=".venv/bin/python -m egvqg"
$E necessity-ensemble-init   --run-id $R --source-run luna-api-baseline-test1257-eval-v1   # freeze questions, prompts, models, and caps; idempotent under the same configuration
$E necessity-ensemble-run    --run-id $R --limit 10     # small real pre-check first; remove --limit for the full run, --phase screen|arbitrate|all
$E necessity-ensemble-report --run-id $R                # fixed-denominator report; offline, writes results.json and final_items.json
```

- Reference answers are by default given independently by the two evaluation groups using the full materials (a reference only when they agree); when the evaluated model is weak, do not use its own answer as the reference (`init --baseline-reference` would do so).
- Concurrency defaults to 32, each call is retried at most once, and the hard HTTP cap defaults to 16000; failed calls remain in `results.jsonl` but are not cached, and are rerun on resume.
- The report distinguishes only "confirmed dual-necessity" and "unconfirmed"; confirmed items count only after manual review, because the evaluator itself will let some through (symmetric grids, swap questions bound by presuppositions, and so on).
- Only questions for which the original evaluator gives hypothetical situations in both directions can be confirmed as dual-necessity, so by default only these questions are evaluated (about 7% of answerable questions, yet the entire source of confirmed items); `init --all-items` evaluates the remaining questions, for diagnostics only, at about 10 times the cost.

## Data

```bash
.venv/bin/python -m egvqg download-data --split train --limit 3          # ScienceQA official source; --offline allows offline re-import
.venv/bin/python -m egvqg fetch-dataset --dataset multimodalqa --workers 2
.venv/bin/python -m egvqg fetch-dataset --dataset webqa --workers 8 --proxy http://127.0.0.1:7890 --image-mirror
```

- **ScienceQA**: [official download source](https://github.com/lupantech/ScienceQA#-download-the-dataset), metadata pinned to commit `2cbf8318e07b9ece895bb2ae605e71e38d623264`; by default only non-empty `hint` and the original image are used, and the original question, options, answer, `lecture`, and `solution` do not enter the materials. Raw data is in `data/raw/scienceqa/<revision>/`, and artifacts are in `data/processed/scienceqa/`. Non-empty image-text is only a rule-based pre-screen and cannot be used to conclude that dual necessity holds.
- **MultiModalQA / WebQA**: [AllenAI MultiModalQA](https://github.com/allenai/multimodalqa), [WebQnA WebQA](https://github.com/WebQnA/WebQA). `fetch-dataset` only downloads, extracts, and verifies; the source URL, size, SHA256, and record count are written into each directory's `manifest.json`; `.part` files are resumable. Both have been fully downloaded (MultiModalQA about 4.6GiB, WebQA about 119GiB, whose images are still unextracted split archives). MultiModalQA has material import integrated: `import-mmqa-materials` generates materials of the form "Wikipedia passage + same-named entity image"; WebQA is not yet integrated.
- For flat JSONL from other datasets, use `normalize-materials` to map to the shared T/I material contract.

## Prompts and Configuration

`prompts/plain-v2.txt` is the fixed generation prompt of the new protocol (English, neutral). The other 9 templates are evaluator and answering prompts (`answer-slots-*`, `ensemble-*`, `open-necessity-v2`, `review-question-faithfulness-v1`, `solve-v1`, `verify-necessity-evidence-v1/v2`), all of which have been translated into English. Prompt templates are maintained independently; a missing `$variable` placeholder raises an error immediately, and each run snapshots the templates it uses into `outputs/<run>/prompts/`. `configs/necessity_assessment_v2.json` stores the knowledge scope K, analysis/verification template names, and token caps used by basic evaluation (the faithfulness template is fixed in code); other run parameters use command-line defaults or override arguments.

## Currently Not Implemented

Formal candidate data and G1 adjudication under plain-v2 (`scripts/gpu_queue.sh` and `scripts/api_queue.sh`), a formal comparison of the proposed generation method (data-side selection/rewriting), the CLI for P6 selection evaluation, second-model re-checking with Qwen3.5, formal runs and evaluation of steering, independent human annotation and scoring calibration, and overall repeated-run stability of the evaluator. The VQAG mainline (plan 7.11) is code-complete but has only been dry-run: real label repair, fact splitting, filtering and steering await explicit API/GPU launch.
