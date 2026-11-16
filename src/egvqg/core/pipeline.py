"""Helpers of the baseline and evaluation workflow (the remaining part of an older module; everything else was removed)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from .records import read_jsonl, sha256_file


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

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

def _validate_run_id(run_id: str) -> None:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run_id must start with a letter/number and contain only letters, numbers, '.', '_' or '-'"
        )
