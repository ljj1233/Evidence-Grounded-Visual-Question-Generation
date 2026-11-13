"""Stable JSONL record schemas shared by generation, evaluation and reporting."""

from __future__ import annotations

import hashlib
import json
import os
import fcntl
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


# ---------------------------------------------------------------------------
# Status vocabularies. Keep these as strings so JSONL stays readable.

CONDITIONS = ("TI", "T", "I", "0")


class QuestionStatus:
    OK = "ok"
    GENERATION_FAILURE = "generation_failure"
    TECHNICAL_FAILURE = "technical_failure"


class ValidityStatus:
    VALID = "valid"
    INVALID = "invalid"
    UNRESOLVED = "unresolved"


class AnswerStatus:
    CANDIDATE = "candidate"
    VERIFIED = "verified"
    UNRESOLVED = "unresolved"


class ResponseStatus:
    ANSWERED = "answered"
    REFUSED = "refused"
    UNPARSEABLE = "unparseable"
    TECHNICAL_FAILURE = "technical_failure"


class ScoreStatus:
    SCORED = "scored"
    UNRESOLVED = "unresolved"


class BDRState:
    PASS = "pass"
    FAIL = "fail"
    UNRESOLVED = "unresolved"


# ---------------------------------------------------------------------------
# Helpers


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(prefix: str, *parts: Any) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row at {path}:{line_no} is not an object")
            rows.append(row)
    return rows


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def file_lock(path: Path):
    """Advisory process lock; callers re-read state after acquiring it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def atomic_upsert(path: Path, key: str, record: Mapping[str, Any], *,
                  allow_insert: bool = True, expected_sha256: str | None = None) -> str:
    """Replace one row, preserving all other rows and rejecting stale updates."""
    with file_lock(path.with_suffix(path.suffix + '.lock')):
        rows = read_jsonl(path)
        ids = [row[key] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f'duplicate {key} in {path}')
        value = record[key]
        if value not in ids:
            if not allow_insert:
                raise ValueError(f'unplanned {key}: {value}')
            rows.append(dict(record))
            action = 'inserted'
        else:
            index = ids.index(value)
            if expected_sha256 and sha256_text(canonical_json(rows[index])) != expected_sha256:
                raise RecordConflict('stale record update')
            if rows[index] == dict(record):
                return 'noop'
            rows[index] = dict(record)
            action = 'updated'
        write_jsonl(path, rows)
        return action


class RecordConflict(RuntimeError):
    """Raised when the same stable record ID is appended with different content."""


class JsonlTable:
    """Append-only JSONL table with in-memory indexing and ID-based deduplication."""

    def __init__(self, path: Path, key: str):
        self.path = path
        self.key = key
        self.rows: dict[str, dict[str, Any]] = {}
        self.added_count = 0
        for row in read_jsonl(path):
            row_key = row.get(key)
            if not isinstance(row_key, str) or not row_key:
                raise ValueError(f"record in {path} is missing key {key!r}")
            existing = self.rows.get(row_key)
            if existing is not None and existing != row:
                raise RecordConflict(f"conflicting duplicate {key}={row_key} in {path}")
            self.rows[row_key] = row

    def get(self, key_value: str) -> dict[str, Any] | None:
        return self.rows.get(key_value)

    def values(self) -> list[dict[str, Any]]:
        return list(self.rows.values())

    def __len__(self) -> int:
        return len(self.rows)

    def add(self, record: Mapping[str, Any]) -> bool:
        key_value = record.get(self.key)
        if not isinstance(key_value, str) or not key_value:
            raise ValueError(f"record is missing key {self.key!r}")
        existing = self.rows.get(key_value)
        if existing is not None:
            if existing != dict(record):
                raise RecordConflict(f"conflicting content for {self.key}={key_value} in {self.path}")
            return False
        append_jsonl(self.path, record)
        self.rows[key_value] = dict(record)
        self.added_count += 1
        return True


# ---------------------------------------------------------------------------
# Record schemas. IDs are stable within a run; question_id links all downstream
# records, and answer_id / response_id keep scoring traceable.


@dataclass
class MaterialRecord:
    material_id: str
    source_group_id: str
    split: str
    text: str
    image_path: str
    image_sha256: str
    provenance: str
    text_source: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationAttemptRecord:
    attempt_id: str
    run_id: str
    protocol_version: str
    question_id: str
    material_id: str
    method: str
    attempt_no: int
    status: str
    raw_output: str | None
    parsed_question: str | None
    failure_reason: str | None
    generator_model: str
    usage: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QuestionRecord:
    run_id: str
    protocol_version: str
    question_id: str
    material_id: str
    source_group_id: str
    split: str
    method: str
    status: str
    question: str | None
    attempt_count: int
    failure_reason: str | None
    generator_model: str
    provider: str
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidityRecord:
    validity_id: str
    run_id: str
    protocol_version: str
    question_id: str
    material_id: str
    method: str
    status: str
    reasons: list[str]
    source: str
    validator_model: str
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnswerRecord:
    answer_id: str
    run_id: str
    protocol_version: str
    question_id: str
    material_id: str
    method: str
    status: str
    candidate_answer: str | None
    candidate_acceptable_answers: list[str]
    answer: str | None
    acceptable_answers: list[str]
    rubric: str | None
    evidence_text: list[str]
    evidence_image: list[str]
    annotation_model: str
    verifier_model: str
    verification_notes: str | None
    source: str
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResponseRecord:
    response_id: str
    run_id: str
    protocol_version: str
    question_id: str
    material_id: str
    method: str
    condition: str
    status: str
    answer: str | None
    answerer_model: str
    input_manifest: dict[str, Any]
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScoreRecord:
    score_id: str
    run_id: str
    protocol_version: str
    question_id: str
    response_id: str
    condition: str
    status: str
    correct: bool | None
    rationale: str
    answer_id: str | None
    scorer: str
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CallRecord:
    call_id: str
    run_id: str
    protocol_version: str
    role: str
    task: str
    material_id: str | None
    question_id: str | None
    condition: str | None
    model: str
    prompt_sha256: str
    input_manifest: dict[str, Any]
    status: str
    error: str | None
    raw_output: str | None
    usage: dict[str, Any]
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentStore:
    """One initialized row per planned attempt, with optimistic atomic updates."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.path = run_dir / 'records.jsonl'
        self.manifest = json.loads((run_dir / 'manifest.json').read_text())
        self.plan = read_jsonl(run_dir / 'plan.jsonl')
        ids = [row['attempt_id'] for row in self.plan]
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate attempt_id in plan')
        if sha256_text(canonical_json(self.plan)) != self.manifest['plan_sha256']:
            raise ValueError('frozen plan changed')
        self.ids = set(ids)

    def rows(self) -> list[dict]:
        rows = read_jsonl(self.path)
        if self.manifest.get('evaluation_status') in ('frozen_human_reviewed', 'frozen_delegated_reviewed') and (
                sha256_file(self.path) != self.manifest.get('frozen_records_sha256')):
            raise ValueError('frozen evaluation records changed')
        ids = [row['attempt_id'] for row in rows]
        if len(ids) != len(set(ids)) or set(ids) != self.ids:
            raise ValueError('records must contain exactly the planned attempts')
        return rows

    def get(self, attempt_id: str) -> dict:
        return next(row for row in self.rows() if row['attempt_id'] == attempt_id)

    def put(self, record: dict, *, previous: dict) -> str:
        if self.manifest.get('evaluation_status') in ('frozen_human_reviewed', 'frozen_delegated_reviewed') and record != previous:
            raise ValueError('evaluation is frozen; use a new run_id for revisions')
        if record['attempt_id'] not in self.ids:
            raise ValueError('unplanned attempt')
        for key in ('attempt_id', 'run_id', 'protocol_version', 'material', 'method', 'plan_index'):
            if record[key] != previous[key]:
                if key == 'material' and record[key] == material_metadata(previous[key]):
                    # Deterministic metadata aliases may enrich imported legacy rows;
                    # source text, image, IDs and provenance remain immutable.
                    continue
                raise ValueError(f'cannot change frozen {key}')
        return atomic_upsert(self.path, 'attempt_id', record, allow_insert=False,
                             expected_sha256=sha256_text(canonical_json(previous)))


def material_metadata(material):
    """Expose explicit metadata aliases while retaining the shared loader contract."""
    m = dict(material); scienceqa = m['material_id'].startswith('scienceqa_')
    provenance = m.get('provenance', '')
    m.setdefault('dataset', 'scienceqa' if scienceqa else 'development_fixture')
    m.setdefault('dataset_revision', provenance.split('@', 1)[1].split(':', 1)[0] if '@' in provenance else None)
    m.setdefault('dataset_split', m['split'])
    m.setdefault('text_field', m.get('text_source'))
    m.setdefault('selection_version', m.get('processing_version'))
    return m


def review_source(review):
    """Keep human and explicitly authorized AI review provenance distinct."""
    source = review.get('source')
    if source == 'human':
        return source
    if source == 'user_delegated_ai' and review.get('user_authorized') is True and review.get('authorization'):
        return source
    raise ValueError('review requires a human source or explicitly user-authorized delegated AI provenance')

