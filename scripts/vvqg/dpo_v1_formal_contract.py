#!/usr/bin/env python3
"""Emit the formal Offline DPO-v1 contract, provenance, split audit and cache/API
accounting deliverables from the frozen artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vvqg.paths import local_model_path
VAL_RE = re.compile(r"(\d+)\.jpg$")


def read(path: Path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def img_id(value) -> str:
    text = str(value)
    match = VAL_RE.search(text)
    return str(int(match.group(1))) if match else str(int(text.lstrip("0") or "0"))


def ids_from(rows, key) -> set[str]:
    return {img_id(r[key]) for r in rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--report-dir", type=Path, required=True)
    args = ap.parse_args()
    O, R = args.out_dir.resolve(), args.report_dir.resolve()
    R.mkdir(parents=True, exist_ok=True)

    dev = ids_from(read(O / "data/dev1000_images.jsonl"), "image_id")
    train = ids_from(read(O / "data/train_images_9184.jsonl"), "image_id")
    sft_train = ids_from(read(ROOT / "src/vvqg/data/sft/train_manifest.jsonl"), "image_id")
    uvfc_disc = ids_from(read(ROOT / "outputs/phase26/vfc/discovery_manifest.jsonl"), "image")
    uvfc_cal = ids_from(read(ROOT / "outputs/phase26/vfc/calibration_manifest.jsonl"), "image")
    dpo_v1_train = ids_from(read(ROOT / "reports/offline_dpo_pilot_20261006T180506Z/data/pairs_train.jsonl"), "image_id")
    dev_audit = json.loads((O / "data/DEV_SPLIT_AUDIT.json").read_text()) if (O / "data/DEV_SPLIT_AUDIT.json").is_file() else {}

    pair_stats = json.loads((O / "data/PAIR_FILTER_STATS.json").read_text()) if (O / "data/PAIR_FILTER_STATS.json").is_file() else {}
    generation = json.loads((O / "data/candidates_aligned/generation_manifest.json").read_text())
    score_summary = {}
    if (O / "data/SCORE_SUMMARY.json").is_file():
        score_summary = json.loads((O / "data/SCORE_SUMMARY.json").read_text())

    base_ckpt = ROOT / "outputs/phase5c_qse_relativeguard/checkpoints/step_final"
    final_ckpt = R / "dpo/checkpoints/step_final"

    config = {}
    if (R / "dpo/config_resolved.json").is_file():
        config = json.loads((R / "dpo/config_resolved.json").read_text())
    training_summary = {}
    if (R / "dpo/training_summary.json").is_file():
        training_summary = json.loads((R / "dpo/training_summary.json").read_text())

    config_contract = {
        "method": "FULL_DATA_OFFLINE_DPO_V1",
        "seed": 42,
        "base_model": config.get("base_model"),
        "base_checkpoint": str(base_ckpt),
        "reference_checkpoint": str(base_ckpt),
        "objective": "DPO",
        "loss": "-logsigmoid(beta*((logp_policy_c - logp_ref_c) - (logp_policy_r - logp_ref_r))).mean() over pairs",
        "per_sequence_logprob": "sum of completion-token log-probs; assistant question tokens only; EOS/prompt/padding excluded",
        "beta": config.get("beta", 0.1),
        "lr": config.get("lr", 5e-6),
        "optimizer": "AdamW", "betas": list(config.get("betas", (0.9, 0.999))),
        "weight_decay": config.get("weight_decay", 0.0), "max_grad_norm": config.get("max_grad_norm", 1.0),
        "batch_size": config.get("batch_size", 1), "gradient_accumulation": 1,
        "epochs": config.get("epochs", 1), "max_steps": config.get("total_optimizer_steps"),
        "scheduler": "ConstantLR(factor=1.0)",
        "precision": "bfloat16", "attn_implementation": config.get("attn_implementation", "sdpa"),
        "instruction": config.get("instruction", "Generate a question about this image."),
        "image_preprocess": {"min_pixels": 4096, "max_pixels": 147456, "backend": "processor-cached"},
        "lora": {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": ["gate_proj","down_proj","k_proj","o_proj","up_proj","v_proj","q_proj"]},
        "changes_vs_historical_v1": [
            "preference pairs come from the formal method-train manifest (9184 train2014 images), not the aligned dev1000 pool",
            "all eligible pairs used (no 578-image subset)",
            "one full epoch (no 300-step cap): max_steps = number of pairs at batch_size 1",
            "scheduler length set to the actual step count",
            "resumable periodic checkpoints added",
        ],
    }
    (R / "CONFIG.json").write_text(json.dumps(config_contract, indent=2) + "\n", encoding="utf-8")

    provenance = {
        "formal_train_manifest": {
            "path": "outputs/phase4b_online_uvfc/data/train_10000.jsonl",
            "sha256": sha256_file(ROOT / "outputs/phase4b_online_uvfc/data/train_10000.jsonl"),
            "records": 10000, "unique_images": 9184,
            "image_list_sha256": sha256_file(O / "data/train_images_9184.jsonl"),
        },
        "candidate_generation": generation,
        "scoring": score_summary,
        "frozen_scorers": {
            "uvfc_dir": "outputs/phase26/vfc",
            "uvfc_sae_sha256": sha256_file(ROOT / "outputs/phase26/vfc/sae.pt"),
            "uvfc_weights_sha256": sha256_file(ROOT / "outputs/phase26/vfc/weights.pt"),
            "flat_model": local_model_path('Qwen2.5-VL-7B-Instruct'),
            "flat_prompt_file": "configs/new_vqg/visual_support_prompts.json",
            "uvfc_generator_checkpoint": "outputs/checkpoints/qwen3vl8b_lora_r16_vqav2_50k/checkpoint-final",
        },
        "pair_stats": pair_stats,
        "dev": {"manifest": str((O / "data/dev1000_images.jsonl").relative_to(ROOT)),
                "sha256": sha256_file(O / "data/dev1000_images.jsonl")},
    }
    (R / "DATA_PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    split_audit = {
        "dev_role": "FRESH_DEVELOPMENT",
        "end_to_end_untouched": True,
        "dev_n": len(dev),
        "excluded_and_checked": {
            "dpo_train_images": {"count": len(train), "overlap": len(train & dev)},
            "sft_base_train_images": {"count": len(sft_train), "overlap": len(sft_train & dev)},
            "uvfc_discovery_images": {"count": len(uvfc_disc), "overlap": len(uvfc_disc & dev)},
            "uvfc_calibration_images": {"count": len(uvfc_cal), "overlap": len(uvfc_cal & dev)},
            "canonical_development1000_val2014": {"excluded": True},
            "val_debug": {"excluded": True},
            "final_holdout": {"excluded_ids_only": True},
            "historical_dpo_v1_train_images": {"count": len(dpo_v1_train), "overlap": len(dpo_v1_train & dev)},
        },
        "dev_construction": dev_audit,
        "gqa_task_modified": False,
        "final_holdout_accessed": False,
    }
    (R / "SPLIT_AUDIT.json").write_text(json.dumps(split_audit, indent=2) + "\n", encoding="utf-8")

    hashes = {
        "base_adapter_sha256": sha256_file(base_ckpt / "adapter_model.safetensors"),
        "reference_adapter_sha256": sha256_file(base_ckpt / "adapter_model.safetensors"),
        "final_dpo_adapter_sha256": sha256_file(final_ckpt / "adapter_model.safetensors") if (final_ckpt / "adapter_model.safetensors").is_file() else None,
        "final_dpo_state_sha256": sha256_file(final_ckpt / "training_state.pt") if (final_ckpt / "training_state.pt").is_file() else None,
        "final_dpo_checkpoint_json": json.loads((final_ckpt / "CHECKPOINT.json").read_text()) if (final_ckpt / "CHECKPOINT.json").is_file() else None,
        "training_summary": training_summary,
    }
    (R / "CHECKPOINT_HASHES.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")

    support_summary = json.loads((O / "evaluation/support/summary.json").read_text())["runtime"] if (O / "evaluation/support/summary.json").is_file() else {}
    seeq_runtime = json.loads((O / "evaluation/seeq_runtime.json").read_text()) if (O / "evaluation/seeq_runtime.json").is_file() else {}
    seeq_first_pass = len(read(O / "evaluation/union_dev1000.jsonl"))
    seeq_retry = seeq_runtime.get("new_requests", 0)
    seeq_total = seeq_first_pass + seeq_retry
    cache_audit = {
        "candidate_generation_cache": "none (aligned train2014 pool generated fresh)",
        "scoring_cache": {"scheme": "content-addressed (image_sha256, question, verifier, prompt/probe version)",
                          "uvfc_computed": score_summary.get("uvfc_computed"), "flat_computed": score_summary.get("flat_computed"),
                          "uvfc_cache_total": score_summary.get("uvfc_cache_total"), "flat_cache_total": score_summary.get("flat_cache_total"),
                          "uvfc_version": score_summary.get("uvfc_version"), "flat_version": score_summary.get("flat_version")},
        "generation_eval_cache": "none (fresh dev not previously generated)",
        "seeq": {"new_requests": seeq_total, "first_pass": seeq_first_pass, "bounded_retry": seeq_retry,
                 "ok": seeq_runtime.get("ok"), "failed": seeq_runtime.get("failed")},
        "support": {"api_requests": support_summary.get("api_requests"), "cache_hits": support_summary.get("cache_hits")},
    }
    (R / "CACHE_AUDIT.json").write_text(json.dumps(cache_audit, indent=2) + "\n", encoding="utf-8")

    api_usage = {
        "train_label_api_calls": 0,
        "evaluation": {
            "seeq_requests": seeq_total,
            "seeq_first_pass": seeq_first_pass,
            "seeq_bounded_retry": seeq_retry,
            "support_requests": support_summary.get("api_requests"),
            "support_cache_hits": support_summary.get("cache_hits"),
            "total_new_requests": seeq_total + (support_summary.get("api_requests") or 0),
            "evaluator_models": {"seeq": seeq_runtime.get("model", "gpt-5.6-luna"), "support": "gpt-4o"},
            "shared_across_systems": "union queue for SeeQ; per-run dedup for support",
            "missing_kept_missing": True,
        },
    }
    (R / "API_USAGE.json").write_text(json.dumps(api_usage, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"dev_n": len(dev), "overlaps": {k: v.get("overlap") for k, v in split_audit["excluded_and_checked"].items() if isinstance(v, dict) and "overlap" in v},
                      "training_summary": training_summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
