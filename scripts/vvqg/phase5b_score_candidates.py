#!/usr/bin/env python3
"""Score a candidate pool with the frozen UVFC scorer and the requirement-level
Flat verifier (Phase 5B).  Reuses cached UVFC values when available; the frozen
prompt/checkpoint/calibration are unchanged.
"""

from __future__ import annotations

import argparse
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

from vvqg.rewards.support.structured_flat import StructuredFlatVerifier, support_summary  # noqa: E402
from vvqg.rewards.visual_information import uvfc as uvfc_mod  # noqa: E402

BASE = local_model_path('Qwen3-VL-8B-Instruct')
SFT_CHECKPOINT = ROOT / "outputs/checkpoints/qwen3vl8b_lora_r16_vqav2_50k/checkpoint-final"
UVFC_DIR = ROOT / "outputs/phase26/vfc"
FLAT_MODEL = local_model_path('Qwen2.5-VL-7B-Instruct')


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--uvfc-cache", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--uvfc-batch-size", type=int, default=32)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--sft-checkpoint", type=Path, default=SFT_CHECKPOINT)
    parser.add_argument("--uvfc-dir", type=Path, default=UVFC_DIR)
    parser.add_argument("--flat-model", default=FLAT_MODEL)
    args = parser.parse_args()

    import torch

    try:
        os.sched_setaffinity(0, set(range(8)))
    except Exception:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = read(args.candidates)
    rows.sort(key=lambda row: (str(row["image_id"]), int(row["generation_id"])))
    key = lambda row: (str(row["image_id"]), int(row["generation_id"]))

    cache: dict[tuple[str, int], dict] = {}
    if args.uvfc_cache and args.uvfc_cache.is_file():
        cache = {key(row): row for row in read(args.uvfc_cache)}

    done = {key(row) for row in read(args.out)} if args.out.is_file() else set()
    pending = [row for row in rows if key(row) not in done]
    if not pending:
        print(f"already scored {len(rows)} candidates -> {args.out}", flush=True)
        return 0

    if cache:
        for row in pending:
            cached = cache.get(key(row))
            if cached is None:
                raise RuntimeError(f"missing UVFC cache for {row['image_id']}#{row['generation_id']}")
            row["UVFC"] = float(cached["UVFC"])
    else:
        from vvqg.training.online_uvfc_grpo import build_frozen_model

        model, processor = build_frozen_model(args.base, str(args.sft_checkpoint), "sdpa")
        model.to(device)
        frozen = uvfc_mod.load_uvfc(args.uvfc_dir, device)
        for start in range(0, len(pending), args.uvfc_batch_size):
            chunk = pending[start : start + args.uvfc_batch_size]
            for row, score in zip(chunk, uvfc_mod.score_uvfc(model, processor, chunk, frozen, device)):
                row["UVFC"] = float(score["UVFC"])
            print(f"uvfc {min(start + args.uvfc_batch_size, len(pending))}/{len(pending)}", flush=True)
        del model, processor
        torch.cuda.empty_cache()

    flat = StructuredFlatVerifier(args.flat_model, device, batch_size=args.batch_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), args.batch_size):
            chunk = pending[start : start + args.batch_size]
            for row, item in zip(chunk, flat.verify_structured(chunk)):
                summary = support_summary(item)
                handle.write(json.dumps({
                    "image_id": str(row["image_id"]), "image": row["image"], "question": row["question"],
                    "generation_id": int(row["generation_id"]), "UVFC": float(row["UVFC"]),
                    "overall_visual_support": item.get("overall_visual_support"),
                    **summary,
                }, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"scored {min(start + args.batch_size, len(pending))}/{len(pending)}", flush=True)
    final = {key(row) for row in read(args.out)}
    if final != {key(row) for row in rows}:
        raise RuntimeError("scoring coverage mismatch")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
