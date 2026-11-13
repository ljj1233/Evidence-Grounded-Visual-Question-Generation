"""Run orchestration: config resolution, resumable JSONL records and derived reporting."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .evaluation import (
    annotate_and_verify_answer,
    answer_question,
    score_response,
    validate_question,
)
from .generation import GenerationFailure, METHODS, run_generation
from .metrics import compute_metrics, render_summary
from .prompts import PromptSet
from .providers import CallRecorder, JudgeProvider, MockProvider, Provider, ProviderError
from .records import (
    CONDITIONS,
    GenerationAttemptRecord,
    JsonlTable,
    QuestionRecord,
    QuestionStatus,
    canonical_json,
    read_jsonl,
    sha256_file,
    sha256_text,
    stable_id,
    utc_now,
    write_json,
    write_jsonl,
)

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
RECORD_FILES = {
    "generation_attempts": "generation_attempts.jsonl",
    "questions": "questions.jsonl",
    "validity": "validity.jsonl",
    "answers": "answers.jsonl",
    "responses": "responses.jsonl",
    "scores": "scores.jsonl",
    "calls": "calls.jsonl",
}
TABLE_KEYS = {
    "generation_attempts": "attempt_id",
    "questions": "question_id",
    "validity": "validity_id",
    "answers": "answer_id",
    "responses": "response_id",
    "scores": "score_id",
    "calls": "call_id",
}
MATERIAL_FIELDS = (
    "material_id",
    "source_group_id",
    "split",
    "text",
    "image_path",
    "image_sha256",
    "provenance",
    "text_source",
)


class RunConfigMismatch(RuntimeError):
    pass


@dataclass
class RunResult:
    run_dir: Path
    metrics: dict[str, Any]
    planned: int
    new_calls: int
    new_records: int


@dataclass
class PreparedRun:
    project_root: Path
    config_path: Path
    config: dict[str, Any]
    run_id: str
    run_dir: Path
    materials: list[dict[str, Any]]
    plan: list[dict[str, Any]]
    prompts: PromptSet | None
    prompt_snapshot: dict[str, str]
    mock_script_path: Path
    mock_script_sha256: str
    config_sha256: str
    materials_sha256: str
    prompts_sha256: str


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_materials(project_root: Path, materials_path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(materials_path)
    if not rows:
        raise ValueError(f"no materials found in {materials_path}")
    seen: set[str] = set()
    materials: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=2):
        missing = [field for field in MATERIAL_FIELDS if field not in row]
        if missing:
            raise ValueError(f"{materials_path}:{index} is missing fields: {', '.join(missing)}")
        material_id = str(row["material_id"])
        if not material_id or material_id in seen:
            raise ValueError(f"duplicate or empty material_id {material_id!r} in {materials_path}")
        seen.add(material_id)
        if not str(row["text"]).strip():
            raise ValueError(f"material {material_id} has empty text")
        image_path = _resolve_path(project_root, str(row["image_path"]))
        if not image_path.exists():
            raise FileNotFoundError(f"image for {material_id} not found: {image_path}")
        actual_sha = sha256_file(image_path)
        if actual_sha != row["image_sha256"]:
            raise ValueError(
                f"image hash mismatch for {material_id}: expected {row['image_sha256']}, got {actual_sha}"
            )
        materials.append(dict(row))
    return materials


def build_plan(materials: list[Mapping[str, Any]], methods: list[str], run_id: str) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for material in materials:
        for method in methods:
            plan.append(
                {
                    "question_id": stable_id("q", run_id, material["material_id"], method),
                    "material_id": material["material_id"],
                    "source_group_id": material["source_group_id"],
                    "split": material["split"],
                    "method": method,
                }
            )
    return plan


def _material_image_path(project_root: Path, material: Mapping[str, Any]) -> Path:
    return _resolve_path(project_root, str(material["image_path"]))


def _validate_run_id(run_id: str) -> None:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run_id must start with a letter/number and contain only letters, numbers, '.', '_' or '-'"
        )


def _prepare_run(
    *,
    config_path: str | Path,
    project_root: str | Path,
    run_id: str | None,
    limit: int | None,
    output_root: str | Path | None,
) -> PreparedRun:
    root = Path(project_root).resolve()
    resolved_config_path = _resolve_path(root, str(config_path))
    if not resolved_config_path.exists():
        raise FileNotFoundError(f"config not found: {resolved_config_path}")
    config = json.loads(resolved_config_path.read_text(encoding="utf-8"))

    methods = config.get("methods")
    if not isinstance(methods, list) or not methods:
        raise ValueError("config.methods must be a non-empty list")
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise ValueError(f"unsupported generation methods: {unknown}")
    if config.get("provider") != "mock":
        raise ValueError("Stage A only supports the mock provider; real providers belong to Stage B")

    run_config = config.get("run", {})
    if not isinstance(run_config, dict):
        raise ValueError("config.run must be an object")
    resolved_run_id = run_id or run_config.get("default_run_id")
    if not isinstance(resolved_run_id, str) or not resolved_run_id:
        raise ValueError("a run_id is required (CLI --run-id or config.run.default_run_id)")
    _validate_run_id(resolved_run_id)

    resolved_limit = limit if limit is not None else run_config.get("limit")
    if resolved_limit is not None:
        if not isinstance(resolved_limit, int) or resolved_limit < 1:
            raise ValueError("limit must be a positive integer or null")
    output_root_value = output_root if output_root is not None else run_config.get("output_root", "outputs")
    resolved_output_root = _resolve_path(root, str(output_root_value))

    max_attempts = run_config.get("max_attempts")
    if not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("config.run.max_attempts must be a positive integer")

    models = config.get("models")
    if not isinstance(models, dict):
        raise ValueError("config.models must be an object")
    for role in ("generator", "annotator", "verifier", "validator", "answerer", "scorer"):
        if not isinstance(models.get(role), str) or not models[role]:
            raise ValueError(f"config.models.{role} must be a non-empty string")

    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("config.paths must be an object")
    materials_path = _resolve_path(root, str(paths["materials"]))
    prompt_dir = _resolve_path(root, str(paths["prompts"]))
    mock_script_path = _resolve_path(root, str(paths["mock_script"]))
    if not mock_script_path.exists():
        raise FileNotFoundError(f"mock script not found: {mock_script_path}")

    materials = load_materials(root, materials_path)
    if resolved_limit is not None:
        materials = materials[:resolved_limit]

    try:
        output_root_snapshot = str(resolved_output_root.relative_to(root))
    except ValueError:
        output_root_snapshot = str(resolved_output_root)
    config_snapshot = copy.deepcopy(config)
    config_snapshot["run_id"] = resolved_run_id
    config_snapshot.setdefault("run", {})
    config_snapshot["run"]["limit"] = resolved_limit
    config_snapshot["run"]["output_root"] = output_root_snapshot
    prompts = PromptSet(prompt_dir)
    plan = build_plan(materials, methods, resolved_run_id)

    prompts_sha256 = prompts.snapshot_sha256()
    materials_sha256 = sha256_text(canonical_json(materials))
    mock_script_sha256 = sha256_file(mock_script_path)
    config_sha256 = sha256_text(
        canonical_json(
            {
                "config": config_snapshot,
                "materials": materials,
                "prompts": prompts.snapshot(),
                "mock_script_sha256": mock_script_sha256,
            }
        )
    )
    return PreparedRun(
        project_root=root,
        config_path=resolved_config_path,
        config=config_snapshot,
        run_id=resolved_run_id,
        run_dir=resolved_output_root / resolved_run_id,
        materials=materials,
        plan=plan,
        prompts=prompts,
        prompt_snapshot=prompts.snapshot(),
        mock_script_path=mock_script_path,
        mock_script_sha256=mock_script_sha256,
        config_sha256=config_sha256,
        materials_sha256=materials_sha256,
        prompts_sha256=prompts_sha256,
    )


def _load_tables(run_dir: Path) -> dict[str, JsonlTable]:
    return {
        name: JsonlTable(run_dir / filename, TABLE_KEYS[name])
        for name, filename in RECORD_FILES.items()
    }


def _ensure_run_state(prepared: PreparedRun) -> dict[str, Any]:
    run_dir = prepared.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "run.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("config_sha256") != prepared.config_sha256:
            raise RunConfigMismatch(
                f"run_id {prepared.run_id!r} already has a different config, prompt or material snapshot; "
                "use a new run_id instead of overwriting the existing run"
            )
        state["status"] = "running"
        state["updated_at"] = utc_now()
        state["n_planned"] = len(prepared.plan)
    else:
        existing_records = [name for name in RECORD_FILES.values() if (run_dir / name).exists()]
        if existing_records:
            raise RunConfigMismatch(
                f"run directory {run_dir} has record files but no run.json: {existing_records}; "
                "use a new run_id"
            )
        state = {
            "run_id": prepared.run_id,
            "protocol_version": prepared.config["protocol_version"],
            "config_sha256": prepared.config_sha256,
            "materials_sha256": prepared.materials_sha256,
            "prompts_sha256": prepared.prompts_sha256,
            "mock_script_sha256": prepared.mock_script_sha256,
            "evidence_class": prepared.config.get("evidence_class", "unspecified"),
            "provider": prepared.config.get("provider"),
            "project_root": str(prepared.project_root),
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "status": "running",
            "n_planned": len(prepared.plan),
        }
    write_json(state_path, state)
    write_json(run_dir / "config.json", prepared.config)
    write_json(run_dir / "prompts.json", prepared.prompt_snapshot)
    write_jsonl(run_dir / "materials.jsonl", prepared.materials)
    return state


def _attempts_for(attempts: JsonlTable, question_id: str) -> list[dict[str, Any]]:
    rows = [row for row in attempts.values() if row["question_id"] == question_id]
    rows.sort(key=lambda row: int(row["attempt_no"]))
    return rows


def _finalize_question(
    questions: JsonlTable,
    *,
    prepared: PreparedRun,
    slot: Mapping[str, Any],
    status: str,
    question: str | None,
    failure_reason: str | None,
    attempt_count: int,
) -> None:
    questions.add(
        QuestionRecord(
            run_id=prepared.run_id,
            protocol_version=prepared.config["protocol_version"],
            question_id=str(slot["question_id"]),
            material_id=str(slot["material_id"]),
            source_group_id=str(slot["source_group_id"]),
            split=str(slot["split"]),
            method=str(slot["method"]),
            status=status,
            question=question,
            attempt_count=attempt_count,
            failure_reason=failure_reason,
            generator_model=prepared.config["models"]["generator"],
            provider=str(prepared.config.get("provider")),
            created_at=utc_now(),
        ).to_dict()
    )


def _run_generation_phase(
    prepared: PreparedRun,
    provider: Provider,
    recorder: CallRecorder,
    tables: Mapping[str, JsonlTable],
) -> None:
    attempts = tables["generation_attempts"]
    questions = tables["questions"]
    materials = {material["material_id"]: material for material in prepared.materials}
    max_attempts = int(prepared.config["run"]["max_attempts"])
    generator_model = prepared.config["models"]["generator"]

    for slot in prepared.plan:
        question_id = str(slot["question_id"])
        if questions.get(question_id) is not None:
            continue
        existing = _attempts_for(attempts, question_id)
        success = next((row for row in existing if row["status"] == QuestionStatus.OK), None)
        if success is not None:
            _finalize_question(
                questions,
                prepared=prepared,
                slot=slot,
                status=QuestionStatus.OK,
                question=success["parsed_question"],
                failure_reason=None,
                attempt_count=len(existing),
            )
            continue
        generation_failure = next(
            (
                row
                for row in reversed(existing)
                if row["status"] == QuestionStatus.GENERATION_FAILURE
            ),
            None,
        )
        if generation_failure is not None:
            _finalize_question(
                questions,
                prepared=prepared,
                slot=slot,
                status=QuestionStatus.GENERATION_FAILURE,
                question=None,
                failure_reason=generation_failure["failure_reason"],
                attempt_count=len(existing),
            )
            continue

        material = materials[str(slot["material_id"])]
        image_path = _material_image_path(prepared.project_root, material)
        next_attempt = max((int(row["attempt_no"]) for row in existing), default=0) + 1
        last_error = existing[-1]["failure_reason"] if existing else None
        finalized = False
        while next_attempt <= max_attempts:
            try:
                outcome = run_generation(
                    provider,
                    prepared.prompts,
                    recorder,
                    material,
                    str(slot["method"]),
                    material_image_path=image_path,
                    run_id=prepared.run_id,
                    protocol_version=prepared.config["protocol_version"],
                    question_id=question_id,
                    generator_model=generator_model,
                )
            except ProviderError as exc:
                last_error = str(exc)
                attempts.add(
                    GenerationAttemptRecord(
                        attempt_id=stable_id("attempt", prepared.run_id, question_id, next_attempt),
                        run_id=prepared.run_id,
                        protocol_version=prepared.config["protocol_version"],
                        question_id=question_id,
                        material_id=str(slot["material_id"]),
                        method=str(slot["method"]),
                        attempt_no=next_attempt,
                        status=QuestionStatus.TECHNICAL_FAILURE,
                        raw_output=None,
                        parsed_question=None,
                        failure_reason=str(exc),
                        generator_model=generator_model,
                        usage={},
                        created_at=utc_now(),
                    ).to_dict()
                )
                next_attempt += 1
                continue
            except GenerationFailure as exc:
                attempts.add(
                    GenerationAttemptRecord(
                        attempt_id=stable_id("attempt", prepared.run_id, question_id, next_attempt),
                        run_id=prepared.run_id,
                        protocol_version=prepared.config["protocol_version"],
                        question_id=question_id,
                        material_id=str(slot["material_id"]),
                        method=str(slot["method"]),
                        attempt_no=next_attempt,
                        status=QuestionStatus.GENERATION_FAILURE,
                        raw_output=exc.raw_output,
                        parsed_question=None,
                        failure_reason=exc.reason,
                        generator_model=generator_model,
                        usage={},
                        created_at=utc_now(),
                    ).to_dict()
                )
                _finalize_question(
                    questions,
                    prepared=prepared,
                    slot=slot,
                    status=QuestionStatus.GENERATION_FAILURE,
                    question=None,
                    failure_reason=exc.reason,
                    attempt_count=next_attempt,
                )
                finalized = True
                break
            else:
                attempts.add(
                    GenerationAttemptRecord(
                        attempt_id=stable_id("attempt", prepared.run_id, question_id, next_attempt),
                        run_id=prepared.run_id,
                        protocol_version=prepared.config["protocol_version"],
                        question_id=question_id,
                        material_id=str(slot["material_id"]),
                        method=str(slot["method"]),
                        attempt_no=next_attempt,
                        status=QuestionStatus.OK,
                        raw_output=outcome.raw_output,
                        parsed_question=outcome.question,
                        failure_reason=None,
                        generator_model=generator_model,
                        usage={},
                        created_at=utc_now(),
                    ).to_dict()
                )
                _finalize_question(
                    questions,
                    prepared=prepared,
                    slot=slot,
                    status=QuestionStatus.OK,
                    question=outcome.question,
                    failure_reason=None,
                    attempt_count=next_attempt,
                )
                finalized = True
                break
        if not finalized:
            _finalize_question(
                questions,
                prepared=prepared,
                slot=slot,
                status=QuestionStatus.TECHNICAL_FAILURE,
                question=None,
                failure_reason=last_error or "technical_retries_exhausted",
                attempt_count=next_attempt - 1,
            )


def _by_question(table: JsonlTable, key: str = "question_id") -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in table.values()}


def _run_evaluation_phase(
    prepared: PreparedRun,
    provider: Provider,
    recorder: CallRecorder,
    tables: Mapping[str, JsonlTable],
) -> None:
    questions = tables["questions"]
    validity = tables["validity"]
    answers = tables["answers"]
    responses = tables["responses"]
    scores = tables["scores"]
    validity_by_question = _by_question(validity)
    answers_by_question = _by_question(answers)
    materials = {material["material_id"]: material for material in prepared.materials}
    models = prepared.config["models"]
    source = str(prepared.config.get("provider"))

    for slot in prepared.plan:
        question_id = str(slot["question_id"])
        question_row = questions.get(question_id)
        if question_row is None or question_row["status"] != QuestionStatus.OK:
            continue
        material = materials[str(slot["material_id"])]
        image_path = _material_image_path(prepared.project_root, material)
        question = question_row["question"]

        if validity_by_question.get(question_id) is None:
            record = validate_question(
                provider,
                prepared.prompts,
                recorder,
                material=material,
                method=str(slot["method"]),
                question=question,
                material_image_path=image_path,
                run_id=prepared.run_id,
                protocol_version=prepared.config["protocol_version"],
                question_id=question_id,
                validator_model=models["validator"],
                source=source,
            )
            validity.add(record.to_dict())
            validity_by_question[question_id] = record.to_dict()

        if answers_by_question.get(question_id) is None:
            record = annotate_and_verify_answer(
                provider,
                prepared.prompts,
                recorder,
                material=material,
                method=str(slot["method"]),
                question=question,
                material_image_path=image_path,
                run_id=prepared.run_id,
                protocol_version=prepared.config["protocol_version"],
                question_id=question_id,
                annotation_model=models["annotator"],
                verifier_model=models["verifier"],
                source=source,
            )
            answers.add(record.to_dict())
            answers_by_question[question_id] = record.to_dict()

        answer_row = answers_by_question.get(question_id)
        for condition in CONDITIONS:
            response_id = stable_id("response", prepared.run_id, question_id, condition)
            response_row = responses.get(response_id)
            if response_row is None:
                response_row = answer_question(
                    provider,
                    prepared.prompts,
                    recorder,
                    material=material,
                    method=str(slot["method"]),
                    question=question,
                    condition=condition,
                    material_image_path=image_path,
                    run_id=prepared.run_id,
                    protocol_version=prepared.config["protocol_version"],
                    question_id=question_id,
                    answerer_model=models["answerer"],
                ).to_dict()
                responses.add(response_row)
            score_id = stable_id("score", prepared.run_id, response_id)
            if scores.get(score_id) is None:
                score_row = score_response(
                    response=response_row,
                    answer=answer_row,
                    run_id=prepared.run_id,
                    protocol_version=prepared.config["protocol_version"],
                    scorer=models["scorer"],
                ).to_dict()
                scores.add(score_row)


def _compute_and_write(prepared: PreparedRun, tables: Mapping[str, JsonlTable]) -> dict[str, Any]:
    metrics = compute_metrics(
        prepared.plan,
        tables["questions"].values(),
        tables["validity"].values(),
        tables["answers"].values(),
        tables["responses"].values(),
        tables["scores"].values(),
        run_id=prepared.run_id,
        protocol_version=prepared.config["protocol_version"],
        evidence_class=str(prepared.config.get("evidence_class", "unspecified")),
    )
    write_json(prepared.run_dir / "metrics.json", metrics)
    (prepared.run_dir / "summary.md").write_text(render_summary(metrics), encoding="utf-8")
    return metrics


def _update_state(prepared: PreparedRun, state: dict[str, Any], status: str) -> None:
    state["status"] = status
    state["updated_at"] = utc_now()
    state["n_planned"] = len(prepared.plan)
    write_json(prepared.run_dir / "run.json", state)


def execute_run(
    *,
    config_path: str | Path = "configs/stage_a.json",
    project_root: str | Path = ".",
    run_id: str | None = None,
    limit: int | None = None,
    output_root: str | Path | None = None,
    provider: Provider | None = None,
) -> RunResult:
    prepared = _prepare_run(
        config_path=config_path,
        project_root=project_root,
        run_id=run_id,
        limit=limit,
        output_root=output_root,
    )
    state = _ensure_run_state(prepared)
    tables = _load_tables(prepared.run_dir)
    if provider is None:
        provider = MockProvider(prepared.mock_script_path)
    recorder = CallRecorder(
        tables["calls"],
        run_id=prepared.run_id,
        protocol_version=prepared.config["protocol_version"],
        default_model=prepared.config["models"]["generator"],
    )

    _run_generation_phase(prepared, provider, recorder, tables)
    _run_evaluation_phase(prepared, provider, recorder, tables)
    metrics = _compute_and_write(prepared, tables)
    _update_state(prepared, state, "complete")

    return RunResult(
        run_dir=prepared.run_dir,
        metrics=metrics,
        planned=len(prepared.plan),
        new_calls=recorder.added_count,
        new_records=sum(table.added_count for table in tables.values()),
    )


def summarize_run(
    *,
    run_id: str,
    project_root: str | Path = ".",
    output_root: str | Path = "outputs",
) -> RunResult:
    _validate_run_id(run_id)
    root = Path(project_root).resolve()
    resolved_output_root = _resolve_path(root, str(output_root))
    run_dir = resolved_output_root / run_id
    state_path = run_dir / "run.json"
    if not state_path.exists():
        raise FileNotFoundError(f"run not found: {run_dir}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    materials = read_jsonl(run_dir / "materials.jsonl")
    plan_path = run_dir / "plan.jsonl"
    plan = read_jsonl(plan_path) if plan_path.exists() else build_plan(materials, config["methods"], run_id)
    prepared = PreparedRun(
        project_root=root,
        config_path=run_dir / "config.json",
        config=config,
        run_id=run_id,
        run_dir=run_dir,
        materials=materials,
        plan=plan,
        prompts=None,
        prompt_snapshot={},
        mock_script_path=root / str(config["paths"]["mock_script"]),
        mock_script_sha256=state.get("mock_script_sha256", ""),
        config_sha256=state["config_sha256"],
        materials_sha256=state["materials_sha256"],
        prompts_sha256=state["prompts_sha256"],
    )
    tables = _load_tables(run_dir)
    metrics = _compute_and_write(prepared, tables)
    return RunResult(
        run_dir=run_dir,
        metrics=metrics,
        planned=len(plan),
        new_calls=0,
        new_records=0,
    )


def rescore_run(*, source_run: str, run_id: str, project_root: str | Path = '.',
                output_root: str | Path = 'outputs', limit: int | None = None,
                max_requests: int = 12, max_tokens: int = 768,
                dry_run: bool = False, judge=None) -> RunResult | dict:
    """Score saved responses in a new run, preserving original attempts and inputs."""
    _validate_run_id(source_run)
    _validate_run_id(run_id)
    if source_run == run_id:
        raise ValueError('scoring must use a new run_id; source records are immutable')
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    root = Path(project_root).resolve()
    output = _resolve_path(root, str(output_root))
    source_dir, run_dir = output / source_run, output / run_id
    source_state = json.loads((source_dir / 'run.json').read_text())
    if source_state.get('status') != 'complete':
        raise ValueError('source run must be complete before rescoring')
    source_config = json.loads((source_dir / 'config.json').read_text())
    materials = read_jsonl(source_dir / 'materials.jsonl')
    if limit is not None:
        materials = materials[:limit]
    material_ids = {row['material_id'] for row in materials}
    source_plan = (read_jsonl(source_dir / 'plan.jsonl') if (source_dir / 'plan.jsonl').exists()
                   else build_plan(read_jsonl(source_dir / 'materials.jsonl'), source_config['methods'], source_run))
    plan = [row for row in source_plan if row['material_id'] in material_ids]
    question_ids = {row['question_id'] for row in plan}
    copied = {name: [row for row in read_jsonl(source_dir / RECORD_FILES[name])
                     if row['question_id'] in question_ids]
              for name in ('questions', 'generation_attempts', 'validity', 'answers', 'responses')}
    judge = judge or JudgeProvider.from_env(root=root, run_dir=run_dir,
                                           max_requests=max_requests, max_tokens=max_tokens)
    prompts = PromptSet(root / 'prompts', names=('score_response',))
    protocol = source_config['protocol_version'] + '+model-score-v1'
    verified = {row['question_id'] for row in copied['answers']
                if row['status'] == 'verified' and row.get('answer')}
    eligible = sum(row['status'] == 'answered' and bool(row.get('answer'))
                   and row['question_id'] in verified for row in copied['responses'])
    if dry_run:
        return {'n_planned': len(plan), 'scorable_responses': eligible,
                'max_http_attempts_without_cache': eligible * judge.max_attempts,
                'request_budget': max_requests, 'max_output_tokens_per_request': max_tokens,
                'concurrency': 1, 'estimated_cost': None,
                'note': 'No API calls. Provider pricing unavailable; budget is a hard request cap.'}
    evidence_class = ('synthetic_fixture_model_scoring' if 'synthetic' in
                      source_config.get('evidence_class', '') else 'model_scoring_uncalibrated')
    config = copy.deepcopy(source_config)
    config.update(run_id=run_id, protocol_version=protocol, source_run=source_run,
                  evidence_class=evidence_class, scoring={'backend': 'llm', **judge.identity})
    config['models']['scorer'] = judge.model
    config['run'].update(limit=limit, output_root=str(output))
    identity = {'config': config, 'source_records': copied, 'materials': materials,
                'prompts': prompts.snapshot(), 'plan': plan}
    fingerprint = sha256_text(canonical_json(identity))
    state_path = run_dir / 'run.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get('config_sha256') != fingerprint:
            raise RunConfigMismatch('source data, scoring model, prompt or settings changed; use a new run_id')
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise RunConfigMismatch('scoring output directory is not empty and has no run state')
        state = {'run_id': run_id, 'source_run': source_run, 'created_at': utc_now(),
                 'config_sha256': fingerprint, 'materials_sha256': sha256_text(canonical_json(materials)),
                 'prompts_sha256': prompts.snapshot_sha256(), 'protocol_version': protocol,
                 'evidence_class': evidence_class, 'provider': 'mixed_source_and_judge', 'n_planned': len(plan)}
    state.update(status='running', updated_at=utc_now())
    state['execution'] = {'max_requests': max_requests, 'max_attempts': judge.max_attempts,
                          'timeout_seconds': judge.timeout, 'concurrency': 1}
    write_json(state_path, state)
    write_json(run_dir / 'config.json', config)
    write_json(run_dir / 'prompts.json', prompts.snapshot())
    write_jsonl(run_dir / 'materials.jsonl', materials)
    write_jsonl(run_dir / 'plan.jsonl', plan)
    for name, rows in copied.items():
        write_jsonl(run_dir / RECORD_FILES[name], [{**row, 'run_id': run_id,
                    'source_run_id': source_run, 'source_protocol_version': row['protocol_version']}
                    for row in rows])
    tables = _load_tables(run_dir)
    recorder = CallRecorder(tables['calls'], run_id=run_id, protocol_version=protocol,
                            default_model=judge.model)
    questions = _by_question(tables['questions'])
    answers = _by_question(tables['answers'])
    for response in tables['responses'].values():
        score_id = stable_id('score', run_id, response['response_id'])
        if tables['scores'].get(score_id) is not None:
            continue
        question = questions[response['question_id']]
        score = score_response(response=response, answer=answers.get(response['question_id']),
                               run_id=run_id, protocol_version=protocol, scorer=judge.model,
                               question=question['question'], provider=judge,
                               prompts=prompts, recorder=recorder)
        tables['scores'].add(score.to_dict())
    calls = tables['calls'].values()
    ledger = read_jsonl(run_dir / 'requests.jsonl')
    network = [row for row in ledger if row['event'] == 'started']
    completed_usage = [row['usage'] for row in ledger if row['event'] == 'completed']
    usage = [row.get('usage', {}) for row in calls if row['status'] == 'ok']
    accounting = {'scope': 'scoring only; source generation/answering calls excluded',
                  'http_attempts': len(network), 'cache_hits': sum(bool(u.get('cache_hit')) for u in usage),
                  'attempts_without_usage': len(network) - len(completed_usage),
                  'total_tokens': sum(u.get('total_tokens') or 0 for u in completed_usage),
                  'tokens_complete': all(u.get('total_tokens') is not None for u in completed_usage)
                                     and len(network) == len(completed_usage),
                  'reported_cost': sum(u.get('cost') or 0 for u in completed_usage),
                  'cost_complete': all(u.get('cost') is not None for u in completed_usage)
                                   and len(network) == len(completed_usage)}
    if not accounting['cost_complete']:
        accounting['reported_cost'] = None
    if not accounting['tokens_complete']:
        accounting['total_tokens'] = None
    write_json(run_dir / 'usage.json', accounting)
    state.update(status='complete', updated_at=utc_now())
    write_json(state_path, state)
    result = summarize_run(run_id=run_id, project_root=root, output_root=output)
    result.new_calls = recorder.added_count
    result.new_records = sum(table.added_count for table in tables.values())
    return result
