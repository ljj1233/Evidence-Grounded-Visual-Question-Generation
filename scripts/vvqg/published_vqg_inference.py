#!/usr/bin/env python3
"""Run a published LLaVA-based VQG checkpoint (LOVA3 / Genixer) on development1000.

Inference only. Uses the official repository's own ``llava`` package (passed via
--repo) and mirrors its official generation path: instruction + image -> text.
The GenQA instruction and decoding contract are supplied by the caller so no
generic prompt is invented here. One process = one checkpoint; successful rows
are cached by (image_id, sample_index) and never regenerated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "outputs/same_backbone_baselines/manifests/development1000_images.jsonl"

_QUESTION = re.compile(r"Question\s*:\s*(.*?)(?:\n\s*Answer\s*:|\Z)", re.S)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_question(raw: str) -> str:
    match = _QUESTION.search(raw)
    text = (match.group(1) if match else raw).replace("</s>", " ").strip().strip('"').strip("'").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    question = next((line for line in lines if line.endswith("?")), lines[0] if lines else "")
    for prefix in ("Question:", "Q:", "ASSISTANT:", "USER:"):
        if question.startswith(prefix):
            question = question[len(prefix):].strip()
    return question


def stable_seed(base: int, image_id: str, sample_index: int) -> int:
    return (base + int(hashlib.sha256(f"{image_id}:{sample_index}".encode()).hexdigest()[:8], 16)) % (2**31)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["lova3", "genixer"])
    parser.add_argument("--repo", type=Path, required=True, help="official code repo root containing the llava package")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--schema", choices=["greedy", "sampled4"], default="greedy")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--instruction-suffix", default="")
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo.resolve()))
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
    from PIL import Image

    disable_torch_init()
    model_path = str(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, get_model_name_from_path(model_path))
    model.eval()

    instruction = args.instruction + args.instruction_suffix
    records = read_jsonl(args.manifest)
    for row in records:
        row["image_id"] = str(row["image_id"])
    if args.limit is not None:
        records = sorted(records, key=lambda r: r["image_id"])[: args.limit]
    image_of = {r["image_id"]: r.get("image_path") or r["image"] for r in records}

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    schema_tag = "greedy" if args.schema == "greedy" else "sampled4"
    cache_path = out / f"cache_{schema_tag}.jsonl"
    cached = {row["key"]: row for row in read_jsonl(cache_path)} if cache_path.is_file() else {}

    tasks = [
        (image_id, sample_index)
        for image_id in sorted(image_of)
        for sample_index in range(args.num_samples)
        if args.retry_failures or f"{image_id}#{sample_index}" not in cached
    ]
    label = args.label or f"{args.method} dev1000"
    started = time.perf_counter()
    peak_gpu = 0

    def build_prompt() -> str:
        qs = instruction
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        return conv

    with cache_path.open("a", encoding="utf-8") as handle, torch.inference_mode():
        bar = tqdm(total=len(tasks), desc=label, unit="gen", dynamic_ncols=True)
        for image_id, sample_index in tasks:
            key = f"{image_id}#{sample_index}"
            row = {"key": key, "image_id": image_id, "sample_index": sample_index, "image_path": image_of[image_id],
                   "instruction": instruction, "conv_mode": args.conv_mode, "temperature": args.temperature,
                   "top_p": args.top_p, "max_new_tokens": args.max_new_tokens,
                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            try:
                conv = build_prompt()
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
                image = Image.open(ROOT / image_of[image_id] if not Path(image_of[image_id]).is_absolute() else Path(image_of[image_id])).convert("RGB")
                image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                stopping = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
                torch.manual_seed(stable_seed(args.seed, image_id, sample_index))
                output_ids = model.generate(
                    input_ids, images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=args.temperature > 0, temperature=args.temperature or None, top_p=args.top_p,
                    num_beams=1, max_new_tokens=args.max_new_tokens, use_cache=True, stopping_criteria=[stopping])
                raw = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
                if raw.endswith(stop_str):
                    raw = raw[: -len(stop_str)].strip()
                question = parse_question(raw)
                row.update({"status": "ok" if question else "invalid", "raw_output": raw, "question": question,
                            "error_type": None if question else "NoQuestion"})
            except Exception as error:  # noqa: BLE001 - one bad item must not abort the run
                row.update({"status": "error", "raw_output": "", "question": "", "error_type": type(error).__name__,
                            "error": str(error)[:300]})
            cached[key] = row
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            peak_gpu = max(peak_gpu, torch.cuda.max_memory_allocated() // (1024 * 1024))
            bar.update(1)
        bar.close()

    rows = [cached[f"{image_id}#{i}"] for image_id in sorted(image_of) for i in range(args.num_samples)]
    ok = [row for row in rows if row["status"] == "ok"]
    if args.schema == "greedy":
        raw_path, pred_path = out / "raw_outputs.jsonl", out / "predictions.jsonl"
    else:
        raw_path, pred_path = out / "raw_outputs_sampled4x.jsonl", out / "sampled4x.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"image_id": row["image_id"], "sample_index": row["sample_index"],
                                     "raw_output": row["raw_output"], "status": row["status"]}, ensure_ascii=False) + "\n")
    with pred_path.open("w", encoding="utf-8") as handle:
        for row in ok:
            handle.write(json.dumps({"image_id": row["image_id"], "sample_index": row["sample_index"],
                                     "image_path": row["image_path"], "question": row["question"]}, ensure_ascii=False) + "\n")

    seconds = time.perf_counter() - started
    integrity = {
        "schema": "published-vqg-integrity-v1", "method": args.method, "split": "development1000",
        "manifest": str(args.manifest), "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "expected_images": len(image_of), "unique_image_ids": len({r["image_id"] for r in rows}),
        "generations": len(rows), "ok": len(ok), "empty": sum(not r["question"] for r in rows),
        "malformed": sum(r["status"] != "ok" for r in rows),
        "failure_types": {name: sum(r.get("error_type") == name for r in rows) for name in sorted({r.get("error_type") for r in rows if r.get("error_type")})},
    }
    (out / "integrity.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    runtime = {
        "schema": "published-vqg-runtime-v1", "method": args.method, "schema_name": schema_tag,
        "requested": len(tasks), "new_requests": len(tasks), "cache_hits": len(image_of) * args.num_samples - len(tasks),
        "wall_seconds": round(seconds, 3), "generations_per_sec": round(len(tasks) / seconds, 3) if seconds else None,
        "peak_gpu_mib": peak_gpu, "decoding": {"temperature": args.temperature, "top_p": args.top_p,
        "num_samples": args.num_samples, "max_new_tokens": args.max_new_tokens, "seed": args.seed},
    }
    (out / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({
        "schema": "published-vqg-manifest-v1", "method": args.method, "model_path": model_path,
        "official_repo": str(args.repo), "conv_mode": args.conv_mode, "instruction": instruction,
        "input_manifest": str(args.manifest), "input_manifest_sha256": integrity["manifest_sha256"],
        "images": len(image_of), "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"integrity": integrity, "runtime": runtime}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
