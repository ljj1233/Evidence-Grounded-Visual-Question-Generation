#!/usr/bin/env python3
"""Run resumable local SeeQ stages (proposal, answer, rewrite, filter)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from vvqg.baselines.common import read_jsonl
from vvqg.baselines.seeq_style import Qwen3VLBackend, SeeQConfig, SeeQPipeline, check_stable_reference

def main():
    p = argparse.ArgumentParser(); p.add_argument("--images", type=Path, required=True); p.add_argument("--stage-dir", type=Path, required=True); p.add_argument("--base-model", required=True); p.add_argument("--adapter", help="proposer/answerer adapter (M0 for R1, M1 for R2)"); p.add_argument("--proposer-adapter", help="explicit alias for --adapter"); p.add_argument("--reference-adapter", help="frozen M0 adapter for rewrite/filter roles"); p.add_argument("--m0-hash", required=True); p.add_argument("--current-m0-hash", required=True); p.add_argument("--round", type=int, choices=(1,2), required=True); p.add_argument("--retained", type=int, default=10000); p.add_argument("--limit", type=int); p.add_argument("--batch-size", type=int, default=8); p.add_argument("--gpu-memory-utilization", type=float, default=0.7); p.add_argument("--backend", choices=("hf", "vllm"), default="hf"); p.add_argument("--turns", type=int, default=2); p.add_argument("--candidates-per-image", type=int, default=2); p.add_argument("--attn-implementation", default="flash_attention_2"); p.add_argument("--device", default="cuda"); args = p.parse_args()
    check_stable_reference(args.m0_hash, args.current_m0_hash)
    images = read_jsonl(args.images)
    proposer_adapter = args.proposer_adapter or args.adapter
    if args.backend == "vllm":
        from vvqg.baselines.seeq_style.vllm_backend import VLLMQwen3VLBackend
        backend = VLLMQwen3VLBackend(base_model=args.base_model, adapter=proposer_adapter, reference_adapter=args.reference_adapter, batch_size=args.batch_size, gpu_memory_utilization=args.gpu_memory_utilization)
    else:
        backend = Qwen3VLBackend(base_model=args.base_model, adapter=proposer_adapter, reference_adapter=args.reference_adapter, device=args.device, batch_size=args.batch_size, attn_implementation=args.attn_implementation)
    config = SeeQConfig(retained_per_round=args.retained, turns=args.turns, candidates_per_image=args.candidates_per_image)
    stats = SeeQPipeline(backend, config).run(images, args.stage_dir, round_id=args.round, limit=args.limit)
    (args.stage_dir / "runtime.json").write_text(json.dumps({**stats, "m0_hash": args.m0_hash}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(stats, sort_keys=True))
if __name__ == "__main__": main()
