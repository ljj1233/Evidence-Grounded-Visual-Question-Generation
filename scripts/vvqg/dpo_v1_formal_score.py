#!/usr/bin/env python3
"""Formal DPO-v1 candidate scoring: frozen UVFC + requirement-level Flat.

Canonical frozen scorers (``vvqg.rewards.visual_information.uvfc`` and
``vvqg.rewards.support.structured_flat``) exactly as in
``scripts/vvqg/phase5b_score_candidates.py``.  Adds a content-addressed cache
keyed by (image content hash, question, verifier identity, prompt/probe version),
so scores can never be silently reused across a different prompt/checkpoint.
Resumable: every computed score is appended to the cache; re-running skips hits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vvqg.paths import local_model_path

from vvqg.rewards.support.frozen_verifier import frozen_prompts  # noqa: E402
from vvqg.rewards.support.structured_flat import StructuredFlatVerifier, support_summary  # noqa: E402
from vvqg.rewards.visual_information import uvfc as uvfc_mod  # noqa: E402

BASE = local_model_path('Qwen3-VL-8B-Instruct')
SFT_CHECKPOINT = ROOT / "outputs/checkpoints/qwen3vl8b_lora_r16_vqav2_50k/checkpoint-final"
UVFC_DIR = ROOT / "outputs/phase26/vfc"
FLAT_MODEL = local_model_path('Qwen2.5-VL-7B-Instruct')


def read(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def content_key(*parts: str) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()


def load_cache(path: Path) -> dict[str, dict]:
    return {row["key"]: row for row in read(path)}


def append_cache(handle, key: str, value: dict) -> None:
    handle.write(json.dumps({"key": key, **value}, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--uvfc-dir", type=Path, default=UVFC_DIR)
    ap.add_argument("--flat-model", default=FLAT_MODEL)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--sft-checkpoint", type=Path, default=SFT_CHECKPOINT)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--uvfc-batch-size", type=int, default=32)
    args = ap.parse_args()

    import torch

    try:
        os.sched_setaffinity(0, set(range(8)))
    except Exception:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    rows = read(args.candidates)
    rows.sort(key=lambda row: (str(row["image_id"]), int(row["generation_id"])))

    uvfc_version = "uvfc-v1:" + sha256_file(args.uvfc_dir / "sae.pt")[:16] + ":" + sha256_file(args.uvfc_dir / "weights.pt")[:16] + ":" + str(args.sft_checkpoint)
    flat_version = "flat-v1:" + hashlib.sha256(frozen_prompts()["flat"].encode()).hexdigest()[:16] + ":" + args.flat_model
    image_hash = {str(p): sha256_file(ROOT / p if not Path(p).is_absolute() else Path(p)) for p in {str(r["image"]) for r in rows}}
    for row in rows:
        ih = image_hash[str(row["image"])]
        row["_uvfc_key"] = content_key(uvfc_version, ih, str(row["question"]))
        row["_flat_key"] = content_key(flat_version, ih, str(row["question"]))

    uvfc_cache = load_cache(args.cache_dir / "uvfc.jsonl")
    flat_cache = load_cache(args.cache_dir / "flat.jsonl")

    pending_uvfc = [r for r in rows if r["_uvfc_key"] not in uvfc_cache]
    if pending_uvfc:
        from vvqg.training.online_uvfc_grpo import build_frozen_model

        model, processor = build_frozen_model(args.base, str(args.sft_checkpoint), "sdpa")
        model.to(device)
        frozen = uvfc_mod.load_uvfc(args.uvfc_dir, device)
        with (args.cache_dir / "uvfc.jsonl").open("a", encoding="utf-8") as handle:
            for start in range(0, len(pending_uvfc), args.uvfc_batch_size):
                chunk = pending_uvfc[start : start + args.uvfc_batch_size]
                for row, score in zip(chunk, uvfc_mod.score_uvfc(model, processor, chunk, frozen, device)):
                    uvfc_cache[row["_uvfc_key"]] = {"uvfc": float(score["UVFC"])}
                    append_cache(handle, row["_uvfc_key"], {"uvfc": float(score["UVFC"])})
                handle.flush()
                print(f"uvfc {min(start + args.uvfc_batch_size, len(pending_uvfc))}/{len(pending_uvfc)}", flush=True)
        del model, processor
        torch.cuda.empty_cache()

    pending_flat = [r for r in rows if r["_flat_key"] not in flat_cache]
    if pending_flat:
        flat = StructuredFlatVerifier(args.flat_model, device, batch_size=args.batch_size)
        with (args.cache_dir / "flat.jsonl").open("a", encoding="utf-8") as handle:
            for start in range(0, len(pending_flat), args.batch_size):
                chunk = pending_flat[start : start + args.batch_size]
                for row, item in zip(chunk, flat.verify_structured(chunk)):
                    value = {"overall_visual_support": item.get("overall_visual_support"), **support_summary(item)}
                    flat_cache[row["_flat_key"]] = value
                    append_cache(handle, row["_flat_key"], value)
                handle.flush()
        del flat
        torch.cuda.empty_cache()

    done = {content_key(str(r["image_id"]), str(r["generation_id"])) for r in read(args.out)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as handle:
        for row in rows:
            if content_key(str(row["image_id"]), str(row["generation_id"])) in done:
                continue
            flat = flat_cache[row["_flat_key"]]
            handle.write(json.dumps({
                "image_id": str(row["image_id"]), "image": row["image"], "question": row["question"],
                "generation_id": int(row["generation_id"]), "UVFC": float(uvfc_cache[row["_uvfc_key"]]["uvfc"]),
                "overall_visual_support": flat["overall_visual_support"], "flat_state": flat["flat_state"],
                "support_floor": flat["support_floor"], "support_mean": flat["support_mean"],
                "n_requirements": flat["n_requirements"], "requirement_labels": flat["requirement_labels"],
            }, ensure_ascii=False) + "\n")
        handle.flush()
    if {content_key(str(r["image_id"]), str(r["generation_id"])) for r in read(args.out)} != {content_key(str(r["image_id"]), str(r["generation_id"])) for r in rows}:
        raise RuntimeError("scoring coverage mismatch")
    summary = {"candidates": len(rows), "uvfc_computed": len(pending_uvfc), "flat_computed": len(pending_flat),
               "uvfc_cache_total": len(uvfc_cache), "flat_cache_total": len(flat_cache),
               "uvfc_version": uvfc_version, "flat_version": flat_version, "out": str(args.out)}
    (args.out.parent / "SCORE_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
