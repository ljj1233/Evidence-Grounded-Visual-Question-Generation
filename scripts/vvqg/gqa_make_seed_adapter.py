#!/usr/bin/env python3
"""Create the GQA-native SeeQ seed M0 = base Qwen3-VL-8B-Instruct as a zero LoRA.

Section 6.1 requires GQA SeeQ to start from base Qwen3-VL-8B-Instruct and forbids
VQAv2-SFT/GQA-SFT initialization. The SeeQ pipeline serves/probes adapters via
LoRA, so we materialize a LoRA whose B matrices are zero; its forward pass is
mathematically identical to the base model. LoRA config matches the VQAv2 SFT
adapter (r=16, alpha=32, dropout=0.05, 7 target modules).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vvqg.paths import local_model_path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText

TARGETS = ["v_proj", "k_proj", "gate_proj", "o_proj", "down_proj", "q_proj", "up_proj"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default=local_model_path('Qwen3-VL-8B-Instruct'))
    p.add_argument("--output", type=Path, default=Path("outputs/gqa_native/published_baselines/seeq/seed_m0"))
    args = p.parse_args()

    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model, dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa")
    config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=TARGETS, task_type="CAUSAL_LM")
    model = get_peft_model(model, config)  # init_lora_weights=True -> B=0, delta=0
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    nonzero = sum(int((n, p.abs().sum().item()) == n and p.abs().sum().item() != 0) for n, p in model.named_parameters() if "lora_B" in n and p.numel())
    total = sum(p.abs().sum().item() for n, p in model.named_parameters() if "lora_B" in n)
    print({"output": str(args.output), "lora_B_sum": total, "nonzero_B": nonzero, "targets": TARGETS})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
