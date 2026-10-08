#!/usr/bin/env python3
"""Validate a new_vqg JSONL dataset against its frozen JSON Schema.

Usage:
    python scripts/new_vqg/check_dataset.py --kind sft --path <file.jsonl>
    python scripts/new_vqg/check_dataset.py --kind calibration --path <file.jsonl>
    python scripts/new_vqg/check_dataset.py --kind rollout --path <file.jsonl>

Schema defaults to src/vvqg/data/<kind>/schema.json and can be overridden
with --schema. Schema validation only: no training, no API calls, no data
generation. Exit codes: 0 = all records valid, 1 = validation errors,
2 = usage/IO error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_ROOT = REPO_ROOT / "src" / "vvqg" / "data"
KINDS = ("sft", "calibration", "rollout")


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _type_ok(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def validate_instance(value: Any, schema: dict, path: str = "$") -> list[str]:
    errors: list[str] = []

    expected = schema.get("type")
    if expected is not None:
        types = expected if isinstance(expected, list) else [expected]
        if not any(_type_ok(value, t) for t in types):
            return [f"{path}: expected type {expected!r}, got {_json_type(value)!r}"]

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value {value!r} not in enum {schema['enum']!r}")

    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required field {key!r}")
        if schema.get("additionalProperties") is False:
            for key in sorted(set(value) - set(props)):
                errors.append(f"{path}: unexpected field {key!r}")
        for key, sub_schema in props.items():
            if key in value:
                errors.extend(validate_instance(value[key], sub_schema, f"{path}.{key}"))

    if isinstance(value, str) and "minLength" in schema:
        if len(value) < schema["minLength"]:
            errors.append(
                f"{path}: string shorter than minLength {schema['minLength']}"
            )

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: fewer than minItems {schema['minItems']}")
        items = schema.get("items")
        if isinstance(items, dict):
            for i, item in enumerate(value):
                errors.extend(validate_instance(item, items, f"{path}[{i}]"))

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value!r} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value!r} > maximum {schema['maximum']}")

    return errors


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_file(dataset_path: Path, schema: dict) -> tuple[int, list[tuple[int, str]]]:
    total = 0
    errors: list[tuple[int, str]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append((lineno, f"invalid JSON: {exc.msg}"))
                continue
            for message in validate_instance(record, schema):
                errors.append((lineno, message))
    return total, errors


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a new_vqg JSONL dataset against its frozen schema."
    )
    parser.add_argument("--kind", required=True, choices=KINDS)
    parser.add_argument("--path", required=True, type=Path)
    parser.add_argument("--schema", type=Path, default=None)
    parser.add_argument("--max-errors", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    schema_path = args.schema or (SCHEMA_ROOT / args.kind / "schema.json")

    if not schema_path.is_file():
        print(f"error: schema not found: {schema_path}", file=sys.stderr)
        return 2
    if not args.path.is_file():
        print(f"error: dataset not found: {args.path}", file=sys.stderr)
        return 2

    try:
        schema = load_json(schema_path)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot load schema {schema_path}: {exc}", file=sys.stderr)
        return 2

    total, errors = validate_file(args.path, schema)

    for lineno, message in errors[: max(args.max_errors, 0)]:
        print(f"{args.path}:{lineno}: {message}")
    if len(errors) > max(args.max_errors, 0):
        print(f"... {len(errors) - args.max_errors} more error(s) suppressed")

    if errors:
        print(f"FAIL: {args.kind}: {len(errors)} error(s) in {total} record(s) [{args.path}]")
        return 1
    print(f"OK: {args.kind}: {total} record(s) valid [{args.path}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
