"""Offline DPO (single objective: DPO).  No GRPO/IPO/DAPO.

Base/reference is the frozen Base Full Ours checkpoint.  LoRA-only training on
Flat-support preference pairs (chosen=Flat supported, rejected=Flat unsupported).

Faithful to the historical DPO-v1 pilot (`reports/offline_dpo_pilot_20261006T180506Z`):
same loss, masking, optimizer, beta and decoding of log-probs.  The only additions
are (a) an optional full-epoch mode (``max_steps`` <= 0), (b) resumable checkpoints
and (c) extra reference diagnostics.  ``epochs`` must be 1.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Sequence

from vvqg.training.fixed_group_relative import build_trainable_model, count_parameters, norm_question
from vvqg.training.online_uvfc_relative_grpo import (
    _restore_rng,
    _rng_state,
    build_frozen_model,
    save_checkpoint,
)
from vvqg.training.policy_logprobs import encode_items, gather_token_logprobs, to_device

DEFAULTS = {
    "instruction": "Generate a question about this image.",
    "beta": 0.1,
    "epochs": 1,
    "max_steps": 300,
    "batch_size": 4,
    "lr": 5e-6,
    "betas": (0.9, 0.999),
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "seed": 42,
    "attn_implementation": "sdpa",
    "checkpoint_steps": [100, 200, 300],
}


def _seq_logprobs(model: Any, processor: Any, pairs: Sequence[dict], instruction: str, device: Any, with_grad: bool) -> tuple[list[Any], list[Any]]:
    import torch

    items = [(str(p["image"]), p["chosen"]) for p in pairs] + [(str(p["image"]), p["rejected"]) for p in pairs]
    encoded = encode_items(processor, items, instruction)
    inputs = to_device(encoded["inputs"], device)
    counts = [len(ids) for ids in encoded["completion_token_ids"]]
    if any(count == 0 for count in counts):
        raise ValueError("empty assistant completion after tokenization")
    if with_grad:
        logits = model(**inputs, use_cache=False).logits
    else:
        with torch.no_grad():
            logits = model(**inputs, use_cache=False).logits
    per_token = gather_token_logprobs(logits, inputs["input_ids"], encoded["completion_start"], counts)
    n = len(pairs)
    chosen = [value.sum() for value in per_token[:n]]
    rejected = [value.sum() for value in per_token[n:]]
    return chosen, rejected


def run_dpo(cfg: dict[str, Any]) -> dict[str, Any]:
    import torch

    resolved = {**DEFAULTS, **cfg}
    if int(resolved["epochs"]) != 1:
        raise ValueError("offline DPO supports exactly 1 epoch")
    seed = int(resolved["seed"])
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    out = Path(resolved["output_dir"])
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = [json.loads(line) for line in Path(resolved["pairs_train"]).read_text().splitlines() if line.strip()]
    random.Random(seed).shuffle(pairs)
    bs = int(resolved["batch_size"])
    total_steps = math.ceil(len(pairs) / bs)
    max_steps = int(resolved["max_steps"]) if resolved.get("max_steps") else total_steps
    max_steps = min(max_steps, total_steps)
    resolved["max_steps"] = max_steps
    resolved["total_optimizer_steps"] = total_steps
    (out / "config_resolved.json").write_text(json.dumps({k: v for k, v in resolved.items()}, indent=2, default=str) + "\n", encoding="utf-8")

    resume_dir = resolved.get("resume")
    policy, processor, _ = build_trainable_model(resolved["base_model"], resume_dir or resolved["base_checkpoint"], resolved["attn_implementation"])
    reference, _ = build_frozen_model(resolved["base_model"], resolved["base_checkpoint"], resolved["attn_implementation"])
    reference.to(device).eval(); reference.requires_grad_(False)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(resolved["lr"]), betas=tuple(resolved["betas"]), weight_decay=float(resolved["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=max(1, total_steps))

    cursor, step, log_mode = 0, 0, "w"
    if resume_dir:
        state = torch.load(Path(resume_dir) / "training_state.pt", map_location="cpu", weights_only=False)
        step = int(state["global_step"]); cursor = step * bs
        optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        _restore_rng(state["rng"])
        log_mode = "a"
    checkpoints = [f"step_{step:04d}"] if step else []
    if step == 0:
        save_checkpoint(policy, processor, optimizer, scheduler, 0, out, "step_0000")
        checkpoints = ["step_0000"]

    beta = float(resolved["beta"])
    checkpoint_steps = set(int(s) for s in resolved["checkpoint_steps"])
    log = (out / "train_log.jsonl").open(log_mode, encoding="utf-8")
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    while step < max_steps and cursor < len(pairs):
        batch = pairs[cursor : cursor + bs]; cursor += bs; step += 1
        chosen, rejected = _seq_logprobs(policy, processor, batch, resolved["instruction"], device, True)
        with torch.no_grad():
            ref_chosen, ref_rejected = _seq_logprobs(reference, processor, batch, resolved["instruction"], device, False)
        margins = torch.stack([beta * ((c - rc) - (r - rr)) for c, r, rc, rr in zip(chosen, rejected, ref_chosen, ref_rejected)])
        dpo = -torch.nn.functional.logsigmoid(margins)
        if "weight" in batch[0]:
            weights = torch.tensor([float(p["weight"]) for p in batch], device=margins.device, dtype=margins.dtype)
            loss = (dpo * weights).sum() / weights.sum()
        else:
            loss = dpo.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, float(resolved["max_grad_norm"])))
        optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        preferences = sum(float(c.detach()) > float(r.detach()) for c, r in zip(chosen, rejected)) / len(batch)
        kl = sum(float(c.detach() - rc.detach()) for c, rc in zip(chosen, ref_chosen)) / len(batch)
        reject_kl = sum(float(r.detach() - rr.detach()) for r, rr in zip(rejected, ref_rejected)) / len(batch)
        chosen_tokens = [len(processor.tokenizer(p["chosen"], add_special_tokens=False)["input_ids"]) for p in batch]
        rejected_tokens = [len(processor.tokenizer(p["rejected"], add_special_tokens=False)["input_ids"]) for p in batch]
        ref_margin = sum(float((c.detach() - rc.detach()) - (r.detach() - rr.detach())) for c, r, rc, rr in zip(chosen, rejected, ref_chosen, ref_rejected)) / len(batch)
        groups = None
        if "weight" in batch[0]:
            chosen_lr = [float(c.detach() - rc.detach()) for c, rc in zip(chosen, ref_chosen)]
            rejected_lr = [float(r.detach() - rr.detach()) for r, rr in zip(rejected, ref_rejected)]
            dpo_v = [float(x.detach()) for x in dpo]
            margin_v = [float(x.detach()) for x in margins]
            pref_v = [float(a) > float(b) for a, b in zip(chosen, rejected)]
            groups = {}
            for name, keep in (("matched", [i for i, p in enumerate(batch) if float(p["weight"]) >= 0.999]),
                               ("fallback", [i for i, p in enumerate(batch) if float(p["weight"]) < 0.999])):
                if not keep:
                    groups[name] = {"n": 0}
                    continue
                groups[name] = {"n": len(keep),
                                "loss": sum(dpo_v[i] for i in keep) / len(keep),
                                "preference_accuracy": sum(pref_v[i] for i in keep) / len(keep),
                                "margin_mean": sum(margin_v[i] for i in keep) / len(keep),
                                "chosen_logratio": sum(chosen_lr[i] for i in keep) / len(keep),
                                "rejected_logratio": sum(rejected_lr[i] for i in keep) / len(keep),
                                "chosen_logratio_per_token": (sum(chosen_lr[i] for i in keep) / len(keep)) / (sum(chosen_tokens[i] for i in keep) / len(keep)),
                                "rejected_logratio_per_token": (sum(rejected_lr[i] for i in keep) / len(keep)) / (sum(rejected_tokens[i] for i in keep) / len(keep))}
        questions = [p["chosen"] for p in batch] + [p["rejected"] for p in batch]
        duplicate_rate = (len(questions) - len({norm_question(q) for q in questions})) / len(questions)
        entry = {"step": step, "loss": float(loss.detach()), "margin_mean": float(margins.detach().mean()),
                 "groups": groups,
                 "preference_accuracy": preferences, "kl_chosen_logratio": kl, "kl_rejected_logratio": reject_kl,
                 "chosen_logratio_per_token": kl / (sum(chosen_tokens) / len(batch)),
                 "rejected_logratio_per_token": reject_kl / (sum(rejected_tokens) / len(batch)),
                 "ref_adjusted_margin": ref_margin, "grad_norm": grad_norm,
                 "consumed_pairs": step * bs, "duplicate_rate": duplicate_rate, "lr": float(optimizer.param_groups[0]["lr"]),
                 "chosen_logp_mean": sum(float(c.detach()) for c in chosen) / len(batch),
                 "rejected_logp_mean": sum(float(r.detach()) for r in rejected) / len(batch),
                 "ref_chosen_logp_mean": sum(float(rc.detach()) for rc in ref_chosen) / len(batch),
                 "ref_rejected_logp_mean": sum(float(rr.detach()) for rr in ref_rejected) / len(batch),
                 "peak_gpu_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None}
        log.write(json.dumps(entry) + "\n"); log.flush()
        if step in checkpoint_steps:
            tag = f"step_{step:05d}"; save_checkpoint(policy, processor, optimizer, scheduler, step, out, tag); checkpoints.append(tag)
    log.close()
    save_checkpoint(policy, processor, optimizer, scheduler, step, out, "step_final"); checkpoints.append("step_final")
    entries = [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines() if line.strip()]
    window = entries[-min(100, len(entries)):]
    group_agg = {}
    for name in ("matched", "fallback"):
        gs = [e["groups"][name] for e in entries if e.get("groups") and e["groups"].get(name, {}).get("n", 0) > 0]
        if gs:
            group_agg[name] = {"steps": len(gs), "mean_loss": sum(g["loss"] for g in gs) / len(gs),
                               "mean_preference_accuracy": sum(g["preference_accuracy"] for g in gs) / len(gs),
                               "mean_margin": sum(g["margin_mean"] for g in gs) / len(gs),
                               "mean_chosen_logratio_per_token": sum(g["chosen_logratio_per_token"] for g in gs) / len(gs),
                               "mean_rejected_logratio_per_token": sum(g["rejected_logratio_per_token"] for g in gs) / len(gs)}
    summary = {"steps": step, "epochs": resolved["epochs"], "beta": beta, "checkpoints": checkpoints, "groups": group_agg,
               "parameters": count_parameters(policy), "train_pairs": len(pairs),
               "total_optimizer_steps": total_steps, "pair_exposures": step * bs,
               "final_preference_accuracy": entries[-1]["preference_accuracy"], "final_loss": entries[-1]["loss"],
               "final_kl_chosen_logratio": entries[-1]["kl_chosen_logratio"],
               "mean_preference_accuracy": sum(e["preference_accuracy"] for e in entries) / len(entries),
               "last100_preference_accuracy": sum(e["preference_accuracy"] for e in window) / len(window),
               "mean_duplicate_rate": sum(e["duplicate_rate"] for e in entries) / len(entries),
               "wall_seconds": time.perf_counter() - started,
               "peak_gpu_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None}
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
