#!/usr/bin/env python3
"""Run one faithful SeeQ round: multi-intent proposal -> answer -> M0 rewrite -> difficulty filter.

Usage
-----
  # R1 (proposer = M0, rewrite/filter = frozen M0)
  .venv-vllm-cu128-v011/bin/python scripts/vvqg/run_seeq_faithful.py \
      --round 1 --images <train_unlabeled.jsonl> \
      --stage-dir outputs/same_backbone_baselines/seeq_faithful/round1 \
      --base-model ~/models/Qwen3-VL-8B-Instruct \
      --proposer-adapter <M0> --reference-adapter <M0> \
      --m0-hash <hash> --current-m0-hash <hash> --backend vllm

  # R2 (proposer = M1, rewrite/filter = frozen M0)
  ... --round 2 --proposer-adapter <M1> --reference-adapter <M0> ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vvqg.paths import local_model_path

from vvqg.baselines.common import read_jsonl  # noqa: E402
from vvqg.baselines.seeq_faithful import FaithfulSeeQConfig, FaithfulSeeQPipeline, resolve_round_models  # noqa: E402
from vvqg.baselines.seeq_style.pipeline import check_stable_reference  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--round", type=int, choices=(1, 2), required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--stage-dir", type=Path, required=True)
    p.add_argument("--base-model", default=str(Path(local_model_path('Qwen3-VL-8B-Instruct')).expanduser()))
    p.add_argument("--proposer-adapter", required=True, help="M0 for round 1, M1 for round 2")
    p.add_argument("--reference-adapter", required=True, help="frozen M0 for both rounds")
    p.add_argument("--m0-hash", required=True)
    p.add_argument("--current-m0-hash", required=True)
    p.add_argument("--retained", type=int, default=10_000)
    p.add_argument("--samples-per-intent", type=int, default=2)
    p.add_argument("--limit", type=int)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    p.add_argument("--backend", choices=("vllm", "hf"), default="vllm")
    args = p.parse_args()

    check_stable_reference(args.m0_hash, args.current_m0_hash)
    proposer, reference = resolve_round_models(args.round, args.reference_adapter, args.proposer_adapter)
    assert proposer == args.proposer_adapter
    assert reference == args.reference_adapter, "round rewrite/filter reference must be the frozen M0 adapter"

    images = read_jsonl(args.images)
    if args.backend == "vllm":
        from vvqg.baselines.seeq_style.vllm_backend import VLLMQwen3VLBackend
        backend = VLLMQwen3VLBackend(
            base_model=args.base_model, adapter=args.proposer_adapter,
            reference_adapter=args.reference_adapter, batch_size=args.batch_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    else:
        from vvqg.baselines.seeq_style.pipeline import Qwen3VLBackend
        backend = Qwen3VLBackend(
            base_model=args.base_model, adapter=args.proposer_adapter,
            reference_base_model=args.base_model, reference_adapter=args.reference_adapter,
            batch_size=args.batch_size,
        )
    config = FaithfulSeeQConfig(retained_per_round=args.retained, samples_per_intent=args.samples_per_intent)
    pipeline = FaithfulSeeQPipeline(
        backend, config,
        proposer_checkpoint=args.proposer_adapter, reference_checkpoint=args.reference_adapter,
    )
    stats = pipeline.run(images, args.stage_dir, round_id=args.round, limit=args.limit)
    (args.stage_dir / "runtime.json").write_text(json.dumps({**stats, "m0_hash": args.m0_hash}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(stats, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
