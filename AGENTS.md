# Project Collaboration Rules

## Project Entry Points

First read `README.md`, `PROGRESS.md`, and `docs/implementation_plan.md`, then read the sections of `VQG_Proposal.md` relevant to the current task. Research definitions follow `VQG_Proposal.md`; the user's latest instruction in the session takes precedence.

## Directories and New Files

| Location | Sole Responsibility |
|---|---|
| `src/egvqg/` | Reusable research code and the unified CLI; generation, evaluation, and other logic are not duplicated in standalone scripts |
| `configs/` | Model, data, and run configurations that contain no credentials; prefer a default configuration plus override arguments, and do not copy configuration files for every run |
| `prompts/` | Prompt templates that require independent maintenance; simple fixed strings do not need to be split into separate files |
| `data/raw/<dataset>/` | Downloaded raw data, kept read-only; processing results are stored separately |
| `data/processed/<dataset>/` | Standardized materials and split/filter manifests; record provenance and processing versions; reference images rather than copying them |
| `outputs/<run_id>/` | A single experiment's raw outputs, effective configuration/prompt snapshots, scores, manual review, reports, and run logs |
| `outputs/cache/` | Reusable model request cache, validated for validity against the inputs and model configuration |
| `tests/` | Lean checks of critical logic and integration; small synthetic fixtures go in `tests/fixtures/` |
| `docs/` | Implementation notes of long-term value; daily progress does not get separate documentation |
| `tmp/` | One-off debugging intermediates and temporary exports; when the task ends, clean up what was created in this round and is no longer needed |
| `.venv/` | The single default project environment; dependencies are managed by `pyproject.toml` and one lock file |

- Create directories and modules as needed; do not pre-create empty directories, `.gitkeep`, empty implementations, or placeholder documents.
- Before creating a new file, first check whether an existing file can be updated; create a new file only when the responsibility is genuinely independent and existing files are unsuitable for carrying it. No need to ask one by one for ordinary new files.
- The root directory holds only project entry documents and necessary tool configuration, not temporary scripts, experiment outputs, or per-round summaries.
- Do not create parallel versions such as `final_v2`, `new`, or `backup`; source code and protocol changes are left to Git to record, and formal runs are distinguished by `run_id` and protocol version.
- When resuming an interrupted run with the same configuration, reuse the original `run_id` and append the missing records after checking that the configuration is consistent; changing the experiment configuration requires creating a new run, which must not overwrite old results.
- Experiment reports go into the corresponding `outputs/<run_id>/`, not stored three times in `docs/`, the root directory, and `reports/`.
- Large data, model weights, caches, outputs, and environments do not enter Git. Shared model caches may be referenced directly; do not copy another one for this project.
- Prefer the unified CLI; create `scripts/` only when independent maintenance is genuinely needed, and scripts are only responsible for calling in-package logic.

## Documentation and Progress Maintenance

- `VQG_Proposal.md` maintains research definitions; `docs/implementation_plan.md` maintains implementation order; `README.md` maintains entry points and currently runnable commands; `story.md` maintains the paper narrative. Modify them only when the corresponding content changes, and do not fully rewrite them every round.
- `PROGRESS.md` is the sole daily progress entry point: update "current status" and milestones in place; append brief entries for this round's substantive progress under "Work Log" at the end, without deleting or rewriting existing records.
- Each progress entry contains the date, items actually completed, verification command/result, artifact locations, unresolved issues, and next steps; omit items that do not apply, and do not copy entire terminal logs or chat transcripts.
- Do not additionally create `SUMMARY.md`, `TODO.md`, `NEXT_STEPS.md`, date-based reports, or another set of planning/progress documents. Temporary plans stay in the session; information that needs handoff is written into the existing progress or implementation plan.
- Modify the protocol version only when the research protocol changes; mere implementation details do not trigger copying the entire document set. If progress does become too long in the future, first compress repetitive descriptions, keep key decisions, failures, and evidence indices, and do not automatically split it into multiple archives.

## Testing and Verification Scope

- Adopt a small number of high-value tests, prefer merging boundary cases with parameterization, do not mechanically pair a test file with every function, and do not pursue coverage numbers.
- Must test: BDR and effective-rate denominators, invalid/undetermined propagation, isolation of the four-condition materials, correspondence between answers and questions, cache invalidation, and no double counting after resume. Add the smallest regression test that prevents recurrence for a real defect.
- For documentation, log wording, low-risk formatting, and simple configuration changes, link/syntax/targeted smoke checks are sufficient; do not add unit tests for these changes.
- Run the affected checks first; at stage delivery, run the lean full offline check suite. After they pass, expand verification only when new modifications, failures, or still-unexplained risks appear, and do not repeatedly rerun identical checks.
- By default, tests run offline using mocks, without triggering paid APIs, model downloads, or large-data reads; real-model smoke runs separately and records the actual cost. Passing mocks does not mean the model or the metrics are valid.

## Efficiency and Implementation Pace

- Complete the current stage in small steps; get the minimal pipeline working before scaling up. For real integration, first use 1–3 groups of materials, then 30 groups, and only enter the formal scale after protocol review.
- Batch entry points support sample caps, budget estimates, and checkpoint resume; set concurrency caps, timeouts, and limited retries, avoiding infinite retries or occupying all resources at once.
- Reuse verified model responses and intermediate results; rerun only the stages affected by this configuration/prompt/data change, and do not reuse scores that have been invalidated.
- Independent reads, writes, and requests may be parallelized in a limited way; keep a one-to-one correspondence among sample ID, condition, and output. Record by material group so that asynchronous results do not cross items.
- Record elapsed time, call counts, tokens, and obtainable costs, and optimize only after identifying bottlenecks. Prefer reducing duplicate calls, repeated model loading, and repeated image processing over building a complex optimization framework in advance.
- Prefer existing dependencies and simple data structures; do not introduce a plugin system, an abstract base class hierarchy, a database, or a scheduling service for one-off needs.
- Preserve raw research results and failure evidence; efficiency optimization must not be achieved by reducing necessary verification, filtering out bad questions, or changing denominators.

## Implementation Principles

- Currently implementing educational image-text QG from scratch; implement step by step according to the handoff plan, without presupposing existing data, code, or model results.
- The coding assistant itself does not automatically become the experiment model. The generation, answer annotation, answering, and scoring roles are configured separately, and their identities are recorded.
- All methods use the same materials; the generator only outputs Q to the outside. The reference answer is bound to the current Q and does not reuse the dataset's original answer by default.
- Preserve all generation attempts, failures, and undetermined items; do not improve metrics by deleting bad questions or retrying silently.
- The four-condition answering is mutually isolated; removed materials and their captions, OCR, and evidence summaries must not leak to the answerer through other fields.
- BDR is computed per question; undetermined is not disguised as 0; the accuracy of answering with the full image-text is not equal to the effective question rate.
- Run offline mocks and a small number of real samples first, then proceed at the scale arranged by the user. Implementation, testing, and fixes within the current milestone are completed autonomously.
- Do not output `.env` or API key contents. Credentials are managed through environment variables or the tool's own authentication, and run logs do not store authentication headers.
- Prefer a simple Python package, configuration files, and JSONL data; do not build a training framework, a web service, or a complex multi-agent system for now.
- Tests focus on issues that affect conclusions, such as denominators, undetermined semantics, material isolation, checkpoint resume, and duplicate records; do not manufacture irrelevant tests just to increase the number of tests.
- Update `PROGRESS.md` at each stage, stating items actually completed, verification commands, failures, and next steps; experiments that were not executed must not be written as completed.

## Performance Optimization

- Measure before optimizing: record wall-clock time, call counts, token usage, and memory at the real data size, then optimize the bottleneck the measurement identifies rather than the one that looks suspicious.
- Reduce duplicate model calls, repeated model loading, and repeated image decoding before introducing caching layers, batching frameworks, or concurrency machinery.
- Reuse verified intermediate artifacts (processed image paths, OCR text, cached model responses); recompute only the stages downstream of a changed input, prompt, or configuration.
- Every optimization stays bounded: batch entry points keep sample caps, concurrency limits, timeouts, and checkpoint resume, and speed must never be bought with skipped verification, filtered-out hard questions, or altered denominators.
- Prefer vectorized operations and existing dependencies over new heavy frameworks; do not add a training framework, scheduler, or database to accelerate a one-off step.
- When an optimization is kept, record the measured before/after numbers beside the change in `PROGRESS.md`.

## Online Research

- Research when the task involves an unfamiliar API, a library behavior whose version is uncertain, a performance technique, a known failure mode, or a framework best practice. Routine local edits, clear bug fixes, and logic already readable in the repository do not require searching.
- Source priority: official documentation, then authoritative issue trackers or upstream release notes, then maintained technical write-ups. Treat forum snippets, aggregated answer sites, and undated blog posts as leads only.
- Use the configured `context7` MCP for library and API documentation instead of generic web search; reserve general search for problems that documentation cannot answer.
- Verify version fit before adopting anything: check the versions pinned in `pyproject.toml` and the lock file, and confirm the documented behavior matches them. Do not paste a solution written for a different major version.
- After researching, state which sources were consulted and classify each as applicable here or not applicable, with the reason. Research output is a recommendation, not an automatic change.
- Research never overrides this project's definitions: `VQG_Proposal.md`, the data splits, four-condition isolation, BDR semantics, and the result-truthfulness requirements outrank anything found online.
- Keep the balance: do not search on every step. Searching by default inflates cost and invites mismatched solutions; the default remains local, repository-grounded reasoning, with research triggered by genuine uncertainty.

## Skill Usage

Skills used by the coding assistant may be installed in a project-level or global skill directory. Read their descriptions first, then use them for specific coding tasks; generic skills must not override this project's data splits, evaluation definitions, or result-truthfulness requirements.
