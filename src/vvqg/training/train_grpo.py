#!/usr/bin/env python3
"""GRPO entry point retained as a schema-only blocked dry-run.

Phase 2.3 does not authorize GRPO, reward weighting, rollout generation, or
model training. The dry-run validates rollout schema files without loading a
model or importing a reward implementation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vvqg.paths import local_model_path

DATA_ROOT = SRC_ROOT / "vvqg" / "data"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "new_vqg" / "grpo.yaml"
DEFAULT_DATA = DATA_ROOT / "rollout"
DEFAULT_OUTPUT = REPO_ROOT / "checkpoint" / "grpo"
DEFAULT_SCHEMA = DATA_ROOT / "rollout" / "schema.json"
DEFAULT_BASE_MODEL = Path(local_model_path('Qwen3-8B')).expanduser()
DRY_RUN_LIMIT = 200


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return payload


def _resolve_path(value: str, default: Path) -> Path:
    if not value:
        return default
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else (REPO_ROOT / candidate)


def _load_validator() -> Any:
    path = REPO_ROOT / "scripts" / "new_vqg" / "check_dataset.py"
    spec = importlib.util.spec_from_file_location("new_vqg_check_dataset", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def discover_jsonl(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.glob("*.jsonl") if not path.name.endswith("_manifest.jsonl")
    )


def scan_records(files: list[Path], schema: dict) -> tuple[int, list[str]]:
    validator = _load_validator()
    total = 0
    errors: list[str] = []
    for path in files:
        count, file_errors = validator.validate_file(path, schema)
        total += count
        errors.extend(f"{path.name}:{line}: {message}" for line, message in file_errors)
    return total, errors


def _read_records(files: list[Path], limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
                if len(records) >= limit:
                    return records
    return records


def dry_run(cfg: dict[str, Any], data_dir: Path, output_dir: Path, schema_path: Path) -> int:
    base_model = _resolve_path(cfg.get("base_model", ""), DEFAULT_BASE_MODEL)
    schema = json.loads(schema_path.read_text(encoding="utf-8")) if schema_path.is_file() else {}
    files = discover_jsonl(data_dir)
    total, errors = scan_records(files, schema) if files else (0, [])
    print("== train_grpo dry-run ==")
    print(f"base_model   : {base_model} ({'found' if base_model.exists() else 'MISSING'})")
    print(f"data_dir     : {data_dir} ({len(files)} jsonl file(s), {total} record(s))")
    print(f"output_dir   : {output_dir}")
    print("rewards      : BLOCKED (no calibrated GRPO reward is active)")
    print(f"lora         : {lora_plan(cfg)}")

    if errors:
        for message in errors[:20]:
            print(f"schema-error : {message}")
        print(f"schema       : FAIL ({len(errors)} error(s))")
        return 1
    print("schema       : OK")

    if not files:
        print("note         : no rollout files found (dry-run tolerates empty input)")
        print("DRY RUN — no model loaded, no training performed.")
        return 0

    records = _read_records(files, DRY_RUN_LIMIT)
    print(f"records      : {len(records)} inspected")
    print("DRY RUN — no model loaded, no reward or training performed.")
    return 0


def lora_plan(cfg: dict[str, Any]) -> dict[str, Any]:
    lora = cfg.get("lora") or {}
    if not lora.get("enabled", False):
        return {"enabled": False}
    return {
        "enabled": True,
        "r": int(lora.get("r", 16)),
        "alpha": float(lora.get("alpha", 32)),
        "dropout": float(lora.get("dropout", 0.05)),
        "target_modules": list(lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
    }


def train(cfg: dict[str, Any], data_dir: Path, output_dir: Path) -> int:
    raise SystemExit(
        "GRPO training is not implemented in this phase. "
        "Only --dry-run is supported (no custom loss is provided; a GRPO "
        "optimizer from `trl` is required and `trl` is not installed)."
    )


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO entry point (dry-run by default).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="validate only (default)")
    mode.add_argument("--train", action="store_true", help="run real training (not implemented)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = _load_yaml(args.config)
    data_dir = args.data_dir or _resolve_path(cfg.get("rollout_data", ""), DEFAULT_DATA)
    output_dir = args.output_dir or _resolve_path(cfg.get("output_dir", ""), DEFAULT_OUTPUT)
    schema_path = data_dir / "schema.json" if data_dir.is_dir() else DEFAULT_SCHEMA

    if args.train:
        return train(cfg, data_dir, output_dir)
    return dry_run(cfg, data_dir, output_dir, schema_path)


if __name__ == "__main__":
    raise SystemExit(main())
