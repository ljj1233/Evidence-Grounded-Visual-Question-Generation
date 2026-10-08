#!/usr/bin/env python3
"""Local frozen Flat verifier over candidate questions (no external evaluator).

Reuses the exact Phase-2.1 Flat prompt/parser/aggregation via
``frozen_verifier`` and keeps the requirement-level summary via
``structured_flat.support_summary``.
"""

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

from vvqg.rewards.support.frozen_verifier import frozen_prompts, parse_prediction  # noqa: E402
from vvqg.rewards.support.structured_flat import support_summary  # noqa: E402

BASE = local_model_path('Qwen2.5-VL-7B-Instruct')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=BASE)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    rows = [json.loads(l) for l in args.candidates.read_text().splitlines() if l.strip()]
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    model = (
        Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="cuda", local_files_only=True
        )
        .eval()
        .requires_grad_(False)
    )
    prompt = frozen_prompts()["flat"]

    def image(path: str):
        candidate = Path(path)
        with Image.open(candidate if candidate.is_absolute() else ROOT / candidate) as source:
            return source.convert("RGB")

    started = time.perf_counter()
    out = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start : start + args.batch_size]
            texts = [
                processor.apply_chat_template(
                    [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt + "\nQuestion: " + r["question"]}]}],
                    tokenize=False, add_generation_prompt=True,
                )
                for r in chunk
            ]
            inputs = processor(text=texts, images=[image(r["image"]) for r in chunk], padding=True, return_tensors="pt").to("cuda")
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            raws = processor.tokenizer.batch_decode(generated[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)
            for row, raw in zip(chunk, raws):
                parsed = parse_prediction(raw, "flat")
                summary = support_summary(parsed)
                out.append({**row, **summary, "overall_visual_support": parsed.get("overall_visual_support"),
                            "support_status": parsed.get("status"), "support_raw": raw})
    with args.out.open("w", encoding="utf-8") as handle:
        for record in out:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    runtime = {"n": len(out), "model": args.model, "seed": 42, "wall_seconds": time.perf_counter() - started,
               "parse_failure_rate": sum(r["support_status"] != "ok" for r in out) / len(out)}
    (args.out.parent / (args.out.stem + "_runtime.json")).write_text(json.dumps(runtime, indent=2) + "\n")
    print(json.dumps(runtime), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
