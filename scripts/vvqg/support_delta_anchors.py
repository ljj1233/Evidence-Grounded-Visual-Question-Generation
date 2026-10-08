#!/usr/bin/env python3
"""Support-Delta GRPO — frozen-pi0 baseline anchors q0 on the full dedup train set."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vvqg.paths import local_model_path

from vvqg.generation.inference import generate_questions, model_prompt  # noqa: E402
from vvqg.rewards.support.structured_flat import StructuredFlatVerifier, support_summary  # noqa: E402
from vvqg.rewards.visual_information import uvfc as uvfc_mod  # noqa: E402
from vvqg.support_delta.reward import support_score  # noqa: E402
from vvqg.training.online_uvfc_relative_grpo import build_frozen_model  # noqa: E402

INSTRUCTION = "Generate a question about this image."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-model", default=local_model_path('Qwen3-VL-8B-Instruct'))
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/phase5c_qse_relativeguard/checkpoints/step_final")
    ap.add_argument("--flat-model", default=local_model_path('Qwen2.5-VL-7B-Instruct'))
    ap.add_argument("--uvfc-dir", type=Path, default=ROOT / "outputs/phase26/vfc")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import torch

    rows = [json.loads(line) for line in args.train_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        rows = rows[: args.limit]
    images = [str(row["image"]) for row in rows]
    out = Path(args.out)
    done = {}
    if out.is_file():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                done[record["image_id"]] = record
    handle = out.open("a", encoding="utf-8")
    started = time.perf_counter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = build_frozen_model(args.base_model, str(args.checkpoint), "sdpa")
    model.to(device).eval()
    prompt = model_prompt(processor, INSTRUCTION)
    flat = StructuredFlatVerifier(args.flat_model, device, batch_size=32)
    frozen = uvfc_mod.load_uvfc(args.uvfc_dir, device)

    n_written = 0
    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start : start + args.batch_size]
        pending = [row for row in chunk if str(row["image_id"]) not in done]
        if not pending:
            continue
        chunk_images = [str(row["image"]) for row in pending]
        generated = generate_questions(model, processor, chunk_images, prompt, num_samples=1, temperature=0.0,
                                       top_p=1.0, max_new_tokens=args.max_new_tokens, seed=args.seed,
                                       batch_size=args.batch_size)
        questions = [group[0].strip() for group in generated]
        flat_rows = [{"image_id": str(row["image_id"]), "image": str(row["image"]), "question": q}
                     for row, q in zip(pending, questions)]
        parsed = flat.verify_structured(flat_rows)
        scored = []
        for row, question, parse in zip(pending, questions, parsed):
            summary = support_summary(parse)
            token_count = len(processor.tokenizer(question, add_special_tokens=False)["input_ids"])
            scored.append({
                "image_id": str(row["image_id"]), "image": str(row["image"]), "question": question,
                "question_token_count": token_count,
                "flat_state": summary.get("flat_state"), "support_floor": summary.get("support_floor"),
                "support_mean": summary.get("support_mean"), "requirement_labels": summary.get("requirement_labels"),
                "n_requirements": summary.get("n_requirements"), "parser_status": parse.get("status"),
            })
        for index in range(0, len(scored), args.batch_size):
            group = scored[index : index + args.batch_size]
            uvfc = uvfc_mod.score_uvfc(model, processor, group, frozen, device)
            for record, score in zip(group, uvfc):
                record["UVFC"] = float(score["UVFC"])
                record["support_score"] = support_score(record.get("support_floor"), record.get("flat_state"),
                                                        len(record["question"].strip()) < 2 or record.get("parser_status") != "ok")
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1
        handle.flush()
        print(json.dumps({"processed": min(start + args.batch_size, len(rows)), "total": len(rows),
                          "written_this_run": n_written, "elapsed": round(time.perf_counter() - started, 1)}), flush=True)
    handle.close()
    print(json.dumps({"total_rows": len(rows), "written_this_run": n_written, "wall_seconds": time.perf_counter() - started}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
