"""CER supervised trainer: one LoRA adapter, paired original/shuffled anchor loss,
KL anchor to the frozen Base policy.  No GRPO / PPO / DAPO."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from vvqg.evidence_constrained.objective import (
    delta_counterfactual,
    kl_to_reference,
    margin_loss,
    split_pairs,
    token_normalized_logprob,
)
from vvqg.training.fixed_group_relative import build_trainable_model, count_parameters
from vvqg.training.online_uvfc_grpo import build_frozen_model
from vvqg.training.policy_logprobs import encode_items, gather_token_logprobs, to_device

DEFAULTS = {
    "instruction": "Generate a question about this image.",
    "margin": 0.5,
    "lambda_cf": 0.10,
    "lambda_kl": 0.02,
    "lr": 5e-6,
    "betas": (0.9, 0.999),
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "epochs": 1,
    "max_steps": 300,
    "seed": 42,
    "batch_size": 8,
    "attn_implementation": "sdpa",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _adapter_digest(checkpoint: Path) -> str:
    return _sha256(Path(checkpoint) / "adapter_model.safetensors")


def save_checkpoint(model: Any, processor: Any, optimizer: Any, step: int, output_dir: Path, tag: str) -> dict[str, Any]:
    target = Path(output_dir) / "checkpoints" / tag
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(target))
    processor.save_pretrained(str(target))
    torch.save({"optimizer": optimizer.state_dict(), "global_step": step}, target / "training_state.pt")
    files = {path.name: _sha256(path) for path in sorted(target.iterdir()) if path.is_file() and path.name != "CHECKPOINT.json"}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    manifest = {"tag": tag, "global_step": step, "files": files, "checkpoint_sha256": digest}
    (target / "CHECKPOINT.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _grad_norm(parameters: Sequence[Any]) -> float:
    total = torch.zeros((), device="cuda" if torch.cuda.is_available() else "cpu")
    for parameter in parameters:
        if parameter.grad is not None:
            total = total + parameter.grad.detach().float().pow(2).sum()
    return float(total.sqrt())


def load_eligible_anchors(path: str | Path, limit: int | None, seed: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    eligible = [row for row in rows if row.get("eligible_for_evidence_loss")]
    random.Random(seed).shuffle(eligible)
    if limit is not None:
        eligible = eligible[: int(limit)]
    return eligible


def _batch_items(batch: Sequence[dict[str, Any]]) -> list[tuple[str, str]]:
    plus = [(str(row["image"]), row["question"]) for row in batch]
    minus = [(str(row["shuffled_image_path"]), row["question"]) for row in batch]
    return plus + minus


def _pairing_ok(batch: Sequence[dict[str, Any]]) -> bool:
    return all(
        str(row["image"]) != str(row["shuffled_image_path"])
        and str(row["image"]).endswith(".jpg")
        and str(row["shuffled_image_path"]).endswith(".jpg")
        for row in batch
    )


def run_cer(cfg: dict[str, Any]) -> dict[str, Any]:
    resolved = {**DEFAULTS, **cfg}
    seed = int(resolved["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    output_dir = Path(resolved["output_dir"])
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.json").write_text(json.dumps({k: v for k, v in resolved.items()}, indent=2, default=str) + "\n", encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_eligible_anchors(resolved["anchors_manifest"], resolved.get("max_image_limit"), seed)
    if not records:
        raise ValueError("no eligible anchor records")

    base_digest_before = _adapter_digest(Path(resolved["base_checkpoint"]))
    policy, processor, _ = build_trainable_model(resolved["base_model"], resolved["base_checkpoint"], resolved["attn_implementation"])
    reference, _ = build_frozen_model(resolved["base_model"], resolved["base_checkpoint"], resolved["attn_implementation"])
    reference.to(device).eval()
    reference.requires_grad_(False)
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=float(resolved["lr"]), betas=tuple(resolved["betas"]), weight_decay=float(resolved["weight_decay"])
    )
    parameter_report = count_parameters(policy)

    checkpoint_steps = set(int(step) for step in resolved.get("checkpoint_steps", [0, 50, 100, 150, 200, 250, 300]))
    save_checkpoint(policy, processor, optimizer, 0, output_dir, "step_0000")

    max_steps = int(resolved["max_steps"])
    batch_size = int(resolved["batch_size"])
    epochs = int(resolved["epochs"])
    probe = bool(resolved.get("probe_gradients", False))
    log_handle = (output_dir / "train_log.jsonl").open("w", encoding="utf-8")
    cursor = 0
    order = list(range(len(records)))
    step = 0
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    checkpoints = ["step_0000"]
    probe_result: dict[str, Any] = {}

    try:
        while step < max_steps:
            if cursor + batch_size > len(order):
                random.Random(seed + step).shuffle(order)
                cursor = 0
            batch = [records[index] for index in order[cursor : cursor + batch_size]]
            cursor += batch_size
            step += 1

            items = _batch_items(batch)
            n_pairs = len(batch)
            encoded = encode_items(processor, items, resolved["instruction"])
            inputs = to_device(encoded["inputs"], device)
            counts = [len(ids) for ids in encoded["completion_token_ids"]]

            logits = policy(**inputs, use_cache=False).logits
            current = gather_token_logprobs(logits, inputs["input_ids"], encoded["completion_start"], counts)
            plus, minus = split_pairs(current, n_pairs)
            lp_plus = torch.stack([token_normalized_logprob(value) for value in plus])
            lp_minus = torch.stack([token_normalized_logprob(value) for value in minus])
            delta = delta_counterfactual(lp_plus, lp_minus)
            l_anchor = -lp_plus.mean()
            l_cf = margin_loss(delta, float(resolved["margin"]))

            if float(resolved["lambda_kl"]) > 0:
                with torch.no_grad():
                    ref_logits = reference(**inputs, use_cache=False).logits
                ref_tokens = gather_token_logprobs(ref_logits, inputs["input_ids"], encoded["completion_start"], counts)
                kl = torch.stack([kl_to_reference(c, r) for c, r in zip(current, ref_tokens)]).mean()
            else:
                kl = torch.zeros((), device=device)

            loss = l_anchor + float(resolved["lambda_cf"]) * l_cf + float(resolved["lambda_kl"]) * kl

            probe_param = None
            if probe and step == 1:
                probe_param = next(parameter for name, parameter in policy.named_parameters() if parameter.requires_grad and ".lora_B" in name)
                probe_before = probe_param.detach().float().clone()
                probe_result = _probe_grads(trainable, l_anchor, l_cf, kl, resolved)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, float(resolved["max_grad_norm"])))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if probe_param is not None:
                probe_result["probe_lora_delta_max"] = float((probe_param.detach().float() - probe_before).abs().max())
                probe_result["probe_lora_updated"] = probe_result["probe_lora_delta_max"] > 0.0

            delta_detached = delta.detach().float()
            entry = {
                "step": step,
                "loss": float(loss.detach()),
                "l_anchor": float(l_anchor.detach()),
                "l_cf": float(l_cf.detach()),
                "kl": float(kl.detach()),
                "grad_norm": grad_norm,
                "lr": float(resolved["lr"]),
                "delta_cf_mean": float(delta_detached.mean()),
                "delta_cf_std": float(delta_detached.std(unbiased=False)),
                "delta_cf_p10": float(torch.quantile(delta_detached, 0.1)),
                "delta_cf_p50": float(torch.quantile(delta_detached, 0.5)),
                "delta_cf_p90": float(torch.quantile(delta_detached, 0.9)),
                "mean_lp_plus": float(lp_plus.detach().mean()),
                "mean_lp_minus": float(lp_minus.detach().mean()),
                "n_pairs": n_pairs,
                "pairing_ok": _pairing_ok(batch),
                "peak_gpu_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                **probe_result,
            }
            log_handle.write(json.dumps(entry) + "\n")
            log_handle.flush()
            if step in checkpoint_steps:
                tag = f"step_{step:04d}"
                save_checkpoint(policy, processor, optimizer, step, output_dir, tag)
                checkpoints.append(tag)
    finally:
        log_handle.close()

    save_checkpoint(policy, processor, optimizer, step, output_dir, "step_final")
    checkpoints.append("step_final")
    base_digest_after = _adapter_digest(Path(resolved["base_checkpoint"]))
    last_entry = json.loads((output_dir / "train_log.jsonl").read_text(encoding="utf-8").strip().splitlines()[-1])

    summary = {
        "run_name": resolved.get("run_name", "cer"),
        "steps": step,
        "max_steps": max_steps,
        "epochs": epochs,
        "seed": seed,
        "n_eligible_train_images": len(records),
        "checkpoints": checkpoints,
        "parameters": parameter_report,
        "base_adapter_sha256_before": base_digest_before,
        "base_adapter_sha256_after": base_digest_after,
        "base_checkpoint_unchanged": base_digest_before == base_digest_after,
        "wall_seconds": time.perf_counter() - started,
        "peak_gpu_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None,
        "final": {key: last_entry[key] for key in ("loss", "l_anchor", "l_cf", "kl", "grad_norm", "delta_cf_mean")},
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _probe_grads(trainable: Sequence[Any], l_anchor: Any, l_cf: Any, kl: Any, resolved: dict[str, Any]) -> dict[str, Any]:
    """Gradient-presence probe with no optimizer step (graph retained)."""
    def zero():
        for parameter in trainable:
            parameter.grad = None

    zero()
    l_cf.backward(retain_graph=True)
    cf_grad = _grad_norm(trainable)

    zero()
    if float(resolved["lambda_kl"]) > 0:
        kl.backward(retain_graph=True)
    kl_grad = _grad_norm(trainable)

    zero()
    l_anchor.backward(retain_graph=True)
    anchor_grad = _grad_norm(trainable)

    zero()
    return {
        "probe_cf_grad_norm": cf_grad,
        "probe_kl_grad_norm": kl_grad,
        "probe_anchor_grad_norm": anchor_grad,
        "probe_cf_nonzero": cf_grad > 0.0,
        "probe_kl_nonzero": kl_grad > 0.0,
        "probe_l_cf_value": float(l_cf.detach()),
    }
