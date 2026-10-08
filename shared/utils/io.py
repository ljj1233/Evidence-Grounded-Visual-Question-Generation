from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL records using the repository's UTF-8 convention."""
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_hashes_from_jsonl(paths: Iterable[Path], *, require: bool = False) -> set[str]:
    """Collect image hashes from flat or canonical VVQG JSONL records."""
    hashes: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            image_hash = row.get("image_hash") or row.get("item", {}).get("image_hash")
            if require and (not isinstance(image_hash, str) or not image_hash):
                raise ValueError(f"missing image_hash in manifest: {path}")
            if isinstance(image_hash, str) and image_hash:
                hashes.add(image_hash)
    return hashes


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
