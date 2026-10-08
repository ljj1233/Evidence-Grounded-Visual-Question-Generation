#!/usr/bin/env python3
"""dev1000 confirmation — input/split/model audit and fixed-output merge."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vvqg.paths import local_model_path

ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric_id(value: str) -> int:
    name = Path(str(value)).stem
    return int(name.split("_")[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", type=Path, required=True)
    ap.add_argument("--experiment-dir", type=Path, default=ROOT / "reports/support_delta_grpo_20261006T022616Z")
    ap.add_argument("--dev-manifest", type=Path, default=ROOT / "outputs/same_backbone_baselines/manifests/development1000_images.jsonl")
    ap.add_argument("--pilot256", type=Path, default=ROOT / "reports/vc_qse_20261005T035840Z/dev256_manifest.jsonl")
    ap.add_argument("--base-checkpoint", type=Path, default=ROOT / "outputs/phase5c_qse_relativeguard/checkpoints/step_final")
    args = ap.parse_args()
    R = args.report_dir.resolve()
    E = args.experiment_dir.resolve()
    (R / "generation").mkdir(parents=True, exist_ok=True)

    base_pred = {str(row["image_id"]): row for row in read_jsonl(E / "pilot_eval/predictions_step0_original.jsonl")}
    sd_pred = {str(row["image_id"]): row for row in read_jsonl(E / "pilot_eval/predictions_step300_original.jsonl")}
    pilot = read_jsonl(args.pilot256)
    dev = read_jsonl(args.dev_manifest)
    step300_checkpoint = E / "pilot_final/checkpoints/step_0300"

    existing = set(base_pred)
    pilot_ids = {str(row["image_id"]) for row in pilot}
    dev_by_id = {str(numeric_id(row.get("image_path") or row["image_id"])): row for row in dev}
    dev_ids = set(dev_by_id)
    remaining = sorted(dev_ids - existing, key=int)

    hashes = {
        "base_checkpoint": str(args.base_checkpoint.relative_to(ROOT)),
        "base_adapter_sha256": sha_file(args.base_checkpoint / "adapter_model.safetensors"),
        "step300_checkpoint": str(step300_checkpoint.relative_to(ROOT)),
        "step300_adapter_sha256": sha_file(step300_checkpoint / "adapter_model.safetensors"),
        "pilot256_manifest": str(args.pilot256.relative_to(ROOT)),
        "pilot256_manifest_sha256": sha_file(args.pilot256),
        "dev1000_manifest": str(args.dev_manifest.relative_to(ROOT)),
        "dev1000_manifest_sha256": sha_file(args.dev_manifest),
    }
    (R / "BASE_AND_STEP300_HASHES.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")

    model_audit = {
        "instruction": "Generate a question about this image.",
        "decoding": {"do_sample": False, "temperature": 0.0, "top_p": 1.0, "max_new_tokens": 32},
        "seed": 42,
        "generation_batch_size": 8,
        "image_preprocessing": "generate_questions / inference.py (Qwen processor, min_pixels=4096, max_pixels=147456, left padding restored)",
        "chat_template": "processor.apply_chat_template([user image+text])",
        "base_model": local_model_path('Qwen3-VL-8B-Instruct'),
        "base_checkpoint": hashes["base_checkpoint"], "base_adapter_sha256": hashes["base_adapter_sha256"],
        "step300_checkpoint": hashes["step300_checkpoint"], "step300_adapter_sha256": hashes["step300_adapter_sha256"],
        "lora_only": True,
    }
    (R / "MODEL_CONFIG_AUDIT.json").write_text(json.dumps(model_audit, indent=2) + "\n", encoding="utf-8")

    existing_question_hash_base = hashlib.sha256("\n".join(base_pred[i]["question"] for i in sorted(existing, key=int)).encode()).hexdigest()
    existing_question_hash_sd = hashlib.sha256("\n".join(sd_pred[i]["question"] for i in sorted(existing, key=int)).encode()).hexdigest()
    split = {
        "existing256_n": len(existing),
        "remaining744_n": len(remaining),
        "full1000_n": len(existing | set(remaining)),
        "existing256_equals_pilot256": existing == pilot_ids,
        "existing256_is_subset_of_dev1000": existing <= dev_ids,
        "remaining744_is_subset_of_dev1000": set(remaining) <= dev_ids,
        "existing256_disjoint_remaining744": len(existing & set(remaining)) == 0,
        "base_pred_ids_equal_step300_pred_ids": set(base_pred) == set(sd_pred),
        "existing_question_hash_base": existing_question_hash_base,
        "existing_question_hash_step300": existing_question_hash_sd,
        "final_holdout_accessed": False,
    }
    (R / "IMAGE_SPLIT_AUDIT.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    input_audit = {"paths": {"experiment_dir": str(E.relative_to(ROOT)), "dev_manifest": hashes["dev1000_manifest"], "pilot256": hashes["pilot256_manifest"]},
                   "hashes": hashes, "split": split, "final_holdout_accessed": False}
    (R / "INPUT_AUDIT.json").write_text(json.dumps(input_audit, indent=2) + "\n", encoding="utf-8")
    (R / "generation/remaining744_ids.txt").write_text("\n".join(remaining) + "\n", encoding="utf-8")
    print(json.dumps(split, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
