#!/usr/bin/env python3
"""Phase 4A.5 execute the dry-run plan: delete only proven-regenerable Phase 4A
intermediate checkpoints, log every action. No canonical artifact is touched."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/repo_cleanup"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def guards(path: Path) -> None:
    rel = path.relative_to(ROOT)
    parts = rel.parts
    assert parts[0] == "outputs" and parts[1].startswith("phase4a"), f"refuse: {rel}"
    assert "phase4b" not in rel.parts, f"refuse phase4b: {rel}"
    assert path.name.startswith("checkpoint-") and path.name != "checkpoint-final", f"refuse checkpoint name: {path.name}"
    assert path.suffix == "" and path.is_dir(), f"refuse non-dir: {rel}"


def main() -> int:
    plan = json.loads((OUT / "cleanup_plan.json").read_text())
    canonical = json.loads((OUT / "canonical_hashes_pre.json").read_text())
    actions = []
    for item in tqdm(plan["delete_regenerable"], desc="Execute cleanup", dynamic_ncols=True):
        path = ROOT / item["path"]
        guards(path)
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        shutil.rmtree(path)
        actions.append({"old_path": item["path"], "action": "DELETE_REGENERABLE", "new_path": None, "reason": item["reason"], "size_bytes": size, "timestamp": datetime.now(timezone.utc).isoformat()})
    (OUT / "cleanup_actions.jsonl").write_text("".join(json.dumps(a) + "\n" for a in actions), encoding="utf-8")
    # verify canonical datasets unchanged
    post = {k: sha256(ROOT / v["path"]) for k, v in canonical.items()}
    drift = {k: {"expected": v["sha256"], "actual": post[k]} for k, v in canonical.items() if post[k] != v["sha256"]}
    (OUT / "canonical_hashes_post.json").write_text(json.dumps({"post": post, "drift": drift}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"deleted": len(actions), "bytes_freed": sum(a["size_bytes"] for a in actions), "canonical_drift": drift}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
