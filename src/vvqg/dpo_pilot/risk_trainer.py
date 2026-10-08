"""Risk-aware DPO-v2: weighted three-tier DPO (S>U 1.0, U>X 1.5, S>X 0.25).

Per-type mean loss is computed first, then weighted.  Reference is the frozen
Base Full Ours checkpoint.  LoRA-only.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

from vvqg.training.fixed_group_relative import build_trainable_model, norm_question
from vvqg.training.online_uvfc_relative_grpo import build_frozen_model, save_checkpoint
from vvqg.training.policy_logprobs import encode_items, gather_token_logprobs, to_device

DEFAULTS = {
    "instruction": "Generate a question about this image.",
    "beta": 0.1, "lr": 2.5e-6, "betas": (0.9, 0.999), "weight_decay": 0.0,
    "max_steps": 100, "seed": 42, "attn_implementation": "sdpa", "checkpoint_steps": [50, 100],
    "weights": {"S>U": 1.0, "U>X": 1.5, "S>X": 0.25},
}


def _logprobs(model: Any, processor: Any, pairs: list[dict], instruction: str, device: Any, with_grad: bool):
    import torch

    items = [(str(p["image"]), p["chosen"]) for p in pairs] + [(str(p["image"]), p["rejected"]) for p in pairs]
    encoded = encode_items(processor, items, instruction)
    inputs = to_device(encoded["inputs"], device)
    counts = [len(ids) for ids in encoded["completion_token_ids"]]
    if with_grad:
        logits = model(**inputs, use_cache=False).logits
    else:
        with torch.no_grad():
            logits = model(**inputs, use_cache=False).logits
    per_token = gather_token_logprobs(logits, inputs["input_ids"], encoded["completion_start"], counts)
    n = len(pairs)
    return [v.sum() for v in per_token[:n]], [v.sum() for v in per_token[n:]]


def run_risk_dpo(cfg: dict[str, Any]) -> dict[str, Any]:
    import torch

    resolved = {**DEFAULTS, **cfg}
    seed = int(resolved["seed"]); random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    out = Path(resolved["output_dir"]); (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "config_resolved.json").write_text(json.dumps({k: v for k, v in resolved.items()}, indent=2, default=str) + "\n", encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = [json.loads(line) for line in Path(resolved["pairs_train"]).read_text().splitlines() if line.strip()]
    by_type: dict[str, list[dict]] = {"S>U": [], "U>X": [], "S>X": []}
    for p in pairs:
        by_type[p["pair_type"]].append(p)
    for key in by_type:
        random.Random(seed).shuffle(by_type[key])

    policy, processor, _ = build_trainable_model(resolved["base_model"], resolved["base_checkpoint"], resolved["attn_implementation"])
    reference, _ = build_frozen_model(resolved["base_model"], resolved["base_checkpoint"], resolved["attn_implementation"])
    reference.to(device).eval(); reference.requires_grad_(False)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(resolved["lr"]), betas=tuple(resolved["betas"]), weight_decay=float(resolved["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)
    save_checkpoint(policy, processor, optimizer, scheduler, 0, out, "step_0000")

    beta = float(resolved["beta"]); weights = resolved["weights"]
    max_steps = int(resolved["max_steps"]); checkpoint_steps = set(int(s) for s in resolved["checkpoint_steps"])
    log = (out / "train_log.jsonl").open("w", encoding="utf-8")
    checkpoints = ["step_0000"]
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for step in range(1, max_steps + 1):
        batch = [by_type[key][step % len(by_type[key])] for key in ("S>U", "U>X", "S>X")]
        chosen, rejected = _logprobs(policy, processor, batch, resolved["instruction"], device, True)
        with torch.no_grad():
            ref_chosen, ref_rejected = _logprobs(reference, processor, batch, resolved["instruction"], device, False)
        per_type_loss, per_type_acc = {}, {}
        totals = []
        for index, key in enumerate(("S>U", "U>X", "S>X")):
            margin = beta * ((chosen[index] - ref_chosen[index]) - (rejected[index] - ref_rejected[index]))
            per_type_loss[key] = -torch.nn.functional.logsigmoid(margin)
            per_type_acc[key] = float(chosen[index].detach() > rejected[index].detach())
            totals.append(weights[key] * per_type_loss[key])
        loss = sum(totals)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        chosen_kl = sum(float(chosen[i].detach() - ref_chosen[i].detach()) for i in range(3)) / 3
        rejected_kl = sum(float(rejected[i].detach() - ref_rejected[i].detach()) for i in range(3)) / 3
        questions = [p["chosen"] for p in batch] + [p["rejected"] for p in batch]
        duplicate_rate = (len(questions) - len({norm_question(q) for q in questions})) / len(questions)
        entry = {"step": step, "loss": float(loss.detach()),
                 "loss_S>U": float(per_type_loss["S>U"].detach()), "loss_U>X": float(per_type_loss["U>X"].detach()),
                 "loss_S>X": float(per_type_loss["S>X"].detach()),
                 "acc_S>U": per_type_acc["S>U"], "acc_U>X": per_type_acc["U>X"], "acc_S>X": per_type_acc["S>X"],
                 "chosen_logp_mean": sum(float(c.detach()) for c in chosen) / 3,
                 "rejected_logp_mean": sum(float(r.detach()) for r in rejected) / 3,
                 "chosen_ref_kl": chosen_kl, "rejected_ref_kl": rejected_kl, "kl_total": (chosen_kl + rejected_kl) / 2,
                 "grad_norm": grad_norm, "duplicate_rate": duplicate_rate, "malformed_rate": 0.0,
                 "lr": float(resolved["lr"]),
                 "peak_gpu_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None}
        log.write(json.dumps(entry) + "\n"); log.flush()
        if step in checkpoint_steps:
            tag = f"step_{step:04d}"; save_checkpoint(policy, processor, optimizer, scheduler, step, out, tag); checkpoints.append(tag)
    log.close()
    save_checkpoint(policy, processor, optimizer, scheduler, max_steps, out, "step_final"); checkpoints.append("step_final")
    entries = [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines() if line.strip()]
    kl50 = next((e["kl_total"] for e in entries if e["step"] == 50), None)
    kl100 = next((e["kl_total"] for e in entries if e["step"] == 100), None)
    summary = {"steps": max_steps, "beta": beta, "lr": resolved["lr"], "checkpoints": checkpoints,
               "kl_step50": kl50, "kl_step100": kl100,
               "kl_guard_fail": bool((kl50 is not None and kl50 > 0.10) or (kl100 is not None and kl100 > 0.10)),
               "mean_acc_S>U": sum(e["acc_S>U"] for e in entries) / len(entries),
               "mean_acc_U>X": sum(e["acc_U>X"] for e in entries) / len(entries),
               "mean_acc_S>X": sum(e["acc_S>X"] for e in entries) / len(entries),
               "wall_seconds": time.perf_counter() - started,
               "peak_gpu_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None}
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
