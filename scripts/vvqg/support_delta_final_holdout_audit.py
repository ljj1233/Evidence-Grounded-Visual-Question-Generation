#!/usr/bin/env python3
"""Final-holdout confirmation — Phase A read-only audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vvqg.paths import local_model_path

ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def numeric_id(value: str) -> str:
    return str(int(Path(str(value)).stem.split("_")[-1]))


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", type=Path, required=True)
    ap.add_argument("--holdout", type=Path, default=ROOT / "src/vvqg/data/evaluation/final_holdout_manifest.jsonl")
    ap.add_argument("--experiment-dir", type=Path, default=ROOT / "reports/support_delta_grpo_20261006T022616Z")
    ap.add_argument("--train-manifest", type=Path, default=ROOT / "reports/support_delta_grpo_20261006T022616Z/resolved_train_manifest.jsonl")
    ap.add_argument("--dev-manifest", type=Path, default=ROOT / "outputs/same_backbone_baselines/manifests/development1000_images.jsonl")
    ap.add_argument("--pilot256", type=Path, default=ROOT / "reports/vc_qse_20261005T035840Z/dev256_manifest.jsonl")
    ap.add_argument("--base-checkpoint", type=Path, default=ROOT / "outputs/phase5c_qse_relativeguard/checkpoints/step_final")
    args = ap.parse_args()
    R = args.report_dir.resolve()
    E = args.experiment_dir.resolve()
    step300 = E / "pilot_final/checkpoints/step_0300"

    for path in (args.holdout, args.base_checkpoint / "adapter_model.safetensors", step300 / "adapter_model.safetensors"):
        if not path.is_file():
            print(json.dumps({"STATUS": "HOLDOUT_INPUT_MISSING", "missing": str(path)}))
            return 1

    from PIL import Image
    rows = read_jsonl(args.holdout)
    ids = [str(r["image_id"]) for r in rows]
    id_counts = Counter(ids)
    hashes, missing, corrupt, sizes = {}, [], [], []
    for row in rows:
        p = ROOT / row["image_path"]
        if not p.is_file():
            missing.append(str(row["image_id"])); continue
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                sizes.append(list(im.size)); hashes[str(row["image_id"])] = sha(p)
        except Exception as exc:  # corrupt image is a real external boundary
            corrupt.append({"image_id": str(row["image_id"]), "error": type(exc).__name__})

    train_ids = {numeric_id(r["image_id"]) for r in read_jsonl(args.train_manifest)}
    dev_ids = {numeric_id(r.get("image_path") or r["image_id"]) for r in read_jsonl(args.dev_manifest)}
    pilot_ids = {numeric_id(r["image_id"]) for r in read_jsonl(args.pilot256)}
    holdout_ids = set(ids)
    hash_counts = Counter(hashes.values())

    holdout_audit = {
        "holdout_count": len(rows),
        "train_overlap_count": len(holdout_ids & train_ids),
        "development_overlap_count": len(holdout_ids & dev_ids),
        "pilot256_overlap_count": len(holdout_ids & pilot_ids),
        "duplicate_image_id_count": sum(c - 1 for c in id_counts.values() if c > 1),
        "duplicate_image_hash_count": sum(c - 1 for c in hash_counts.values() if c > 1),
        "missing_image_count": len(missing),
        "corrupted_image_count": len(corrupt),
        "unique_image_ids": len(set(ids)) == len(ids),
        "image_size_min": [min(s[0] for s in sizes), min(s[1] for s in sizes)] if sizes else None,
        "image_size_max": [max(s[0] for s in sizes), max(s[1] for s in sizes)] if sizes else None,
        "missing_ids": missing[:20], "corrupted": corrupt[:20],
    }
    if holdout_audit["development_overlap_count"] or holdout_audit["train_overlap_count"]:
        holdout_audit["status"] = "HOLDOUT_OVERLAP_WARNING"
    else:
        holdout_audit["status"] = "OK"
    (R / "HOLDOUT_AUDIT.json").write_text(json.dumps(holdout_audit, indent=2) + "\n", encoding="utf-8")

    input_audit = {
        "holdout_manifest": str(args.holdout.relative_to(ROOT)), "holdout_manifest_sha256": sha(args.holdout),
        "train_manifest": str(args.train_manifest.relative_to(ROOT)), "dev_manifest": str(args.dev_manifest.relative_to(ROOT)),
        "pilot256_manifest": str(args.pilot256.relative_to(ROOT)),
        "holdout_audit": holdout_audit, "final_holdout_accessed": True,
        "authorization": "explicit_user_authorization_for_final_holdout_read_only_evaluation",
    }
    (R / "INPUT_AUDIT.json").write_text(json.dumps(input_audit, indent=2) + "\n", encoding="utf-8")

    model_audit = {
        "base_checkpoint": str(args.base_checkpoint), "base_adapter_sha256": sha(args.base_checkpoint / "adapter_model.safetensors"),
        "step300_checkpoint": str(step300), "step300_adapter_sha256": sha(step300 / "adapter_model.safetensors"),
        "base_model": local_model_path('Qwen3-VL-8B-Instruct'),
        "instruction": "Generate a question about this image.",
        "decoding": {"do_sample": False, "temperature": 0.0, "top_p": 1.0, "max_new_tokens": 32},
        "seed": 42, "lora_only": True,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        "worktree_clean": subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip() == "",
    }
    (R / "MODEL_AUDIT.json").write_text(json.dumps(model_audit, indent=2) + "\n", encoding="utf-8")
    (R / "AUTHORIZATION_RECORD.md").write_text(
        "# Authorization record\n\nThe user explicitly authorized reading and evaluating `final_holdout` in this task.\n"
        "Task is read-only over the holdout: no training, no optimizer step, no model/reward/decoding change, no label edits,\n"
        "no sample-dependent selection, no re-query of unfavorable items.\n\n"
        f"- holdout manifest: `{args.holdout.relative_to(ROOT)}`\n- holdout images: {len(rows)}\n"
        f"- git commit: {model_audit['git_commit']}\n", encoding="utf-8")
    (R / "GIT_STATE.txt").write_text(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout, encoding="utf-8")
    (R / "GIT_DIFF.patch").write_text(subprocess.run(["git", "diff"], cwd=ROOT, capture_output=True, text=True).stdout, encoding="utf-8")
    print(json.dumps(holdout_audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
