"""Online UVFC + violation-constraint GRPO (Phase 4B research pilot).

Fresh on-policy rollouts every optimizer step.  Reward = frozen UVFC with a
violation-first lexicographic group ranking; the objective is the Phase-3A
clipped group-relative surrogate against a frozen Final-SFT reference.  The
undercoverage component ``u_j`` is frozen (Variant A / Static) or refreshed from
the current-policy feature distribution (Variant B / Dynamic); the visual
sensitivity ``v_j``, the SAE and the feature extraction never change.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from vvqg.generation.generator import DEFAULT_INSTRUCTION
from vvqg.generation.inference import generate_questions, model_prompt
from vvqg.rewards.support.online_flat import OnlineFlatVerifier
from vvqg.rewards.visual_information import uvfc as uvfc_mod
from vvqg.training.fixed_group_relative import (
    VIOLATION,
    build_trainable_model,
    canonicalize_group,
    clipped_surrogate,
    count_parameters,
    norm_question,
)
from vvqg.training.policy_logprobs import encode_group, gather_token_logprobs, to_device

ABORT_KL = 0.2
ABORT_CLIP = 0.5
ABORT_MALFORMED = 0.02
ABORT_LENGTH_FACTOR = 1.5
ABORT_DUP_DELTA = 0.2
GUARD_STREAK = 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rng_state() -> dict[str, Any]:
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict[str, Any]) -> None:
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def build_frozen_model(base_model: str, checkpoint: str, attn_implementation: str = "sdpa"):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(base_model, local_files_only=True, use_fast=False)
    processor.tokenizer.padding_side = "right"
    base = AutoModelForImageTextToText.from_pretrained(
        base_model, dtype=torch.bfloat16, local_files_only=True, attn_implementation=attn_implementation
    )
    model = PeftModel.from_pretrained(base, checkpoint, is_trainable=False).eval().requires_grad_(False)
    return model, processor


def save_checkpoint(model: Any, processor: Any, optimizer: Any, scheduler: Any, step: int, output_dir: Path, tag: str) -> dict[str, Any]:
    import torch

    target = output_dir / "checkpoints" / tag
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(target))
    processor.save_pretrained(str(target))
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": step,
            "rng": _rng_state(),
        },
        target / "training_state.pt",
    )
    files = {
        path.name: _sha256(path) for path in sorted(target.iterdir()) if path.is_file() and path.name != "CHECKPOINT.json"
    }
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    manifest = {"tag": tag, "global_step": step, "files": files, "checkpoint_sha256": digest}
    (target / "CHECKPOINT.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _load_train_records(path: str, step_seed: int) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty training manifest {path}")
    random.Random(step_seed).shuffle(rows)
    return rows


def _forward_logprobs(model: Any, inputs: dict[str, Any], encoded: dict[str, Any], with_grad: bool) -> tuple[list[Any], Any]:
    import torch

    counts = [len(ids) for ids in encoded["completion_token_ids"]]
    if with_grad:
        logits = model(**inputs, use_cache=False).logits
    else:
        with torch.no_grad():
            logits = model(**inputs, use_cache=False).logits
    per_token = gather_token_logprobs(logits, inputs["input_ids"], encoded["completion_start"], counts)
    if with_grad:
        return per_token, logits
    return per_token, None


def _group_terms(policy: Any, encoded: dict[str, Any], inputs: dict[str, Any], old_lp: Sequence[Any], ref_lp: Sequence[Any], advantages: Sequence[float], cfg: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """One group's loss plus detached statistics. Assumes ``old_lp``/``ref_lp`` are CPU tensors."""
    import torch

    current, _ = _forward_logprobs(policy, inputs, encoded, with_grad=True)
    response_surrogate, response_kl = [], []
    ratios, kl_values, advantages_out, lengths = [], [], [], []
    for index, current_tokens in enumerate(current):
        old_tokens = old_lp[index].to(current_tokens.device, dtype=torch.float32)
        ref_tokens = ref_lp[index].to(current_tokens.device, dtype=torch.float32)
        advantage = torch.tensor(float(advantages[index]), device=current_tokens.device)
        log_ratio = torch.clamp(current_tokens - old_tokens, -cfg["log_ratio_clamp"], cfg["log_ratio_clamp"])
        ratio = torch.exp(log_ratio)
        response_surrogate.append(clipped_surrogate(ratio, advantage, cfg["clip_epsilon"]).mean())
        ref_ratio = ref_tokens - current_tokens
        response_kl.append((torch.exp(ref_ratio) - ref_ratio - 1.0).mean())
        ratios.append(ratio.detach())
        kl_values.append(response_kl[-1].detach())
        advantages_out.append(float(advantages[index]))
        lengths.append(int(current_tokens.numel()))
    policy_loss = -torch.stack(response_surrogate).mean()
    mean_kl = torch.stack(response_kl).mean()
    loss = policy_loss + cfg["beta_kl"] * mean_kl
    stats = {
        "policy_loss": float(policy_loss.detach()),
        "reference_kl": float(mean_kl.detach()),
        "ratios": torch.cat(ratios),
        "kl_values": torch.stack(kl_values),
        "advantages": advantages_out,
        "lengths": lengths,
        "loss": float(loss.detach()),
    }
    return loss, stats


def _percentile(values: Any, fraction: float) -> float:
    import torch

    return float(torch.quantile(values, fraction))


def _question_stats(processor: Any, questions: Sequence[str]) -> dict[str, float]:
    lengths = [len(processor.tokenizer(question, add_special_tokens=False)["input_ids"]) for question in questions]
    normalized = [norm_question(question) for question in questions]
    total = len(questions)
    counts = Counter(normalized)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    return {
        "n": total,
        "mean_token_length": sum(lengths) / total,
        "max_token_length": max(lengths),
        "malformed_rate": sum(len(question.strip()) < 2 for question in questions) / total,
        "duplicate_rate": duplicates / total,
        "unique_rate": len(set(normalized)) / total,
    }


def _generate(model: Any, processor: Any, images: Sequence[str], prompt: str, cfg: dict[str, Any], seed: int, num_samples: int, temperature: float):
    return generate_questions(
        model,
        processor,
        images,
        prompt,
        num_samples=num_samples,
        temperature=temperature,
        top_p=cfg["top_p"] if temperature > 0 else 1.0,
        max_new_tokens=cfg["max_new_tokens"],
        seed=seed,
        batch_size=int(cfg["generation_batch_size"]),
    )


def _build_groups(rollout_rows: Sequence[dict[str, Any]], group_size: int) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rollout_rows:
        groups[row["image_id"]].append(row)
    return groups


def run_online(cfg: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch

    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rollouts").mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    config_record = {**cfg, "variant": cfg["variant"], "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (output_dir / "config.json").write_text(json.dumps(config_record, indent=2, default=str) + "\n", encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = _load_train_records(cfg["train_manifest"], seed)
    group_size = int(cfg["group_size"])
    prompt = None

    policy, processor, _ = build_trainable_model(cfg["base_model"], cfg["sft_checkpoint"], cfg.get("attn_implementation", "sdpa"))
    reference, _ = build_frozen_model(cfg["base_model"], cfg["sft_checkpoint"], cfg.get("attn_implementation", "sdpa"))
    reference.to(device)
    prompt = model_prompt(processor, cfg["instruction"])

    frozen = uvfc_mod.load_uvfc(cfg["uvfc_dir"], device)
    flat = OnlineFlatVerifier(cfg["flat_model"], device, batch_size=int(cfg.get("flat_batch_size", 32)))

    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=float(cfg["lr"]), betas=tuple(cfg.get("betas", (0.9, 0.999))), weight_decay=float(cfg.get("weight_decay", 0.0))
    )
    max_steps = int(cfg["max_steps"])
    inner_epochs = int(cfg.get("inner_epochs", 2))
    total_updates = max_steps * inner_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_updates, eta_min=0.0)

    trainable_names = {name for name, parameter in policy.named_parameters() if parameter.requires_grad}
    base_trainable = [name for name in trainable_names if ".lora_" not in name]
    if base_trainable:
        raise RuntimeError(f"non-LoRA trainable parameters: {base_trainable[:5]}")
    parameter_report = count_parameters(policy)

    checkpoint_steps = set(int(s) for s in cfg.get("checkpoint_steps", [0, 100, 200, 500, 750]))
    snapshot_every = int(cfg.get("rollout_snapshot_every", 50))
    refresh_every = int(cfg.get("refresh_every", 0)) if cfg["variant"] == "dynamic" else 0

    save_checkpoint(policy, processor, optimizer, scheduler, 0, output_dir, "step_0000")
    checkpoint_manifests = ["step_0000"]

    log_handle = (output_dir / "train_log.jsonl").open("w", encoding="utf-8")
    refresh_dir = output_dir / "uvfc_refresh"
    if refresh_every:
        refresh_dir.mkdir(parents=True, exist_ok=True)

    p_anchor = frozen["p_anchor"]
    u_current = frozen["u"].clone()
    window_active = torch.zeros(uvfc_mod.DICT, device=device)
    window_total = 0

    initial_length = None
    initial_dup = None
    stop_reason = None
    recent: list[dict[str, Any]] = []
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    cursor = 0
    step = 0
    from tqdm.auto import tqdm

    progress = tqdm(total=max_steps, desc=cfg.get("progress_label", f"{cfg['run_name']} GRPO"), unit="step", dynamic_ncols=True)

    def snapshot(step_index: int, rows: list[dict[str, Any]]) -> None:
        path = output_dir / "rollouts" / f"step_{step_index:04d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    while step < max_steps and stop_reason is None:
        if cursor + int(cfg["groups_per_step"]) > len(records):
            random.Random(seed + step).shuffle(records)
            cursor = 0
        batch_records = records[cursor : cursor + int(cfg["groups_per_step"])]
        cursor += int(cfg["groups_per_step"])
        step += 1

        images = [row["image"] for row in batch_records]
        generation_seed = seed * 1_000_003 + step
        sampled = _generate(policy, processor, images, prompt, cfg, generation_seed, group_size, float(cfg["temperature"]))
        rollout_rows: list[dict[str, Any]] = []
        malformed_questions = 0
        for image, questions in zip(images, sampled):
            for generation_id, question in enumerate(questions):
                if len(question.strip()) < 2:
                    malformed_questions += 1
                rollout_rows.append(
                    {"image_id": image, "image": image, "question": question, "generation_id": generation_id}
                )

        uvfc_scores = []
        for start in range(0, len(rollout_rows), group_size):
            uvfc_scores.extend(
                uvfc_mod.score_uvfc(
                    reference, processor, rollout_rows[start : start + group_size], frozen, device, u_override=u_current
                )
            )
        for row, score in zip(rollout_rows, uvfc_scores):
            row["UVFC"] = score["UVFC"]
            row["VFC"] = score["VFC"]
            row["question_token_count"] = score["question_token_count"]
            row["active_features"] = score["active_features"]
            row["_active"] = score["_active"]
        flat_states = flat.verify(rollout_rows)
        for row, state in zip(rollout_rows, flat_states):
            row["flat_state"] = state

        if refresh_every:
            for row in rollout_rows:
                window_active.add_(row["_active"].to(device).float())
            window_total += len(rollout_rows)

        groups = _build_groups(rollout_rows, group_size)
        usable: list[dict[str, Any]] = []
        skipped_groups = 0
        for image_id, members in groups.items():
            if len(members) != group_size or any(len(str(m["question"]).strip()) < 2 for m in members):
                skipped_groups += 1
                continue
            canonical = canonicalize_group(members, "p2")
            if canonical[0]["zero_variance_group"]:
                skipped_groups += 1
                continue
            canonical[0]["group_id"] = image_id
            usable.append({"image_id": image_id, "members": canonical})
        if not usable:
            progress.update(1)
            continue

        step_ratios, step_advantages, step_lengths, step_losses, step_kls, step_policy = [], [], [], [], [], []
        for inner in range(inner_epochs):
            optimizer.zero_grad(set_to_none=True)
            for group in usable:
                members = group["members"]
                encoded = encode_group(processor, members[0]["image"], [m["question"] for m in members], cfg["instruction"])
                inputs = to_device(encoded["inputs"], device)
                with torch.no_grad():
                    old_tokens, _ = _forward_logprobs(policy, inputs, encoded, with_grad=False)
                    ref_tokens, _ = _forward_logprobs(reference, inputs, encoded, with_grad=False)
                old_lp = [value.detach().float().cpu() for value in old_tokens]
                ref_lp = [value.detach().float().cpu() for value in ref_tokens]
                if inner == 0:
                    for member, old_value, ref_value in zip(members, old_lp, ref_lp):
                        member["policy_logprob_sum"] = float(old_value.sum())
                        member["reference_logprob_sum"] = float(ref_value.sum())
                advantages = [float(m["advantage"]) for m in members]
                loss, stats = _group_terms(policy, encoded, inputs, old_lp, ref_lp, advantages, cfg)
                (loss / len(usable)).backward()
                if inner == inner_epochs - 1:
                    step_losses.append(stats["loss"])
                    step_kls.append(stats["reference_kl"])
                    step_policy.append(stats["policy_loss"])
                    step_ratios.append(stats["ratios"])
                    step_advantages.extend(stats["advantages"])
                    step_lengths.extend(stats["lengths"])
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
            optimizer.step()
            scheduler.step()

        ratio_tensor = torch.cat(step_ratios)
        violation_rate = sum(row["flat_state"] == VIOLATION for row in rollout_rows) / len(rollout_rows)
        question_stats = _question_stats(processor, [row["question"] for row in rollout_rows])
        if initial_length is None:
            initial_length = question_stats["mean_token_length"]
            initial_dup = question_stats["duplicate_rate"]
        entry = {
            "step": step,
            "global_step": step,
            "loss": sum(step_losses) / len(step_losses),
            "policy_loss": sum(step_policy) / len(step_policy),
            "reference_kl": sum(step_kls) / len(step_kls),
            "grad_norm": grad_norm,
            "lr": scheduler.get_last_lr()[0],
            "clip_fraction": float(((ratio_tensor < 1 - cfg["clip_epsilon"]) | (ratio_tensor > 1 + cfg["clip_epsilon"])).float().mean()),
            "ratio_mean": float(ratio_tensor.mean()),
            "ratio_p95": _percentile(ratio_tensor, 0.95),
            "ratio_max": float(ratio_tensor.max()),
            "mean_uvfc": sum(row["UVFC"] for row in rollout_rows) / len(rollout_rows),
            "std_uvfc": float(np.std([row["UVFC"] for row in rollout_rows])),
            "violation_rate": violation_rate,
            "mean_advantage": sum(step_advantages) / len(step_advantages),
            "mean_question_tokens": question_stats["mean_token_length"],
            "max_question_tokens": question_stats["max_token_length"],
            "malformed_rate": question_stats["malformed_rate"],
            "duplicate_rate": question_stats["duplicate_rate"],
            "groups_used": len(usable),
            "groups_skipped": skipped_groups,
            "responses": len(rollout_rows),
            "u_released_features": int((u_current > 0).sum()),
            "peak_gpu_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        log_handle.write(json.dumps(entry) + "\n")
        log_handle.flush()
        progress.update(1)
        progress.set_postfix(loss=f"{entry['loss']:.4g}", KL=f"{entry['reference_kl']:.4g}", grad=f"{entry['grad_norm']:.3g}",
                             reward=f"{entry['mean_uvfc']:.3g}", lr=f"{entry['lr']:.2g}")
        recent.append(entry)
        recent[:] = recent[-GUARD_STREAK:]

        if step % snapshot_every == 0:
            snapshot(step, [
                {
                    "image_id": row["image_id"],
                    "question": row["question"],
                    "uvfc_score": row["UVFC"],
                    "flat_state": row["flat_state"],
                    "rank_reward": row.get("canonical_reward"),
                    "advantage": row.get("advantage"),
                    "policy_logprob": row.get("policy_logprob_sum"),
                    "reference_logprob": row.get("reference_logprob_sum"),
                }
                for group in usable
                for row in group["members"]
            ])

        if refresh_every and step % refresh_every == 0 and window_total:
            before_u = u_current.detach()
            new_u = uvfc_mod.dynamic_u(p_anchor, window_active, window_total)
            changed = (new_u - before_u).abs()
            top = torch.topk(changed, 20)
            record = {
                "step": step,
                "window_questions": window_total,
                "before": {"active": int((before_u > 0).sum()), "sum": float(before_u.sum())},
                "after": {"active": int((new_u > 0).sum()), "sum": float(new_u.sum())},
                "top_changed_features": [
                    {"feature": int(index), "before": float(before_u[index]), "after": float(new_u[index])}
                    for index in top.indices.tolist()
                ],
                "p_policy_min": float((window_active / window_total).min()),
                "p_policy_max": float((window_active / window_total).max()),
            }
            (refresh_dir / f"step_{step:04d}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            u_current = new_u
            window_active = torch.zeros(uvfc_mod.DICT, device=device)
            window_total = 0

        if step in checkpoint_steps:
            tag = f"step_{step:04d}"
            save_checkpoint(policy, processor, optimizer, scheduler, step, output_dir, tag)
            checkpoint_manifests.append(tag)

        finite = all(math.isfinite(entry[key]) for key in ("loss", "reference_kl", "grad_norm"))
        if not finite:
            stop_reason = "NUMERICAL_FAILURE"
        elif len(recent) == GUARD_STREAK and all(row["reference_kl"] > ABORT_KL for row in recent):
            stop_reason = "KL_DRIFT"
        elif len(recent) == GUARD_STREAK and all(row["clip_fraction"] > ABORT_CLIP for row in recent):
            stop_reason = "POLICY_DRIFT"
        elif (
            entry["malformed_rate"] > ABORT_MALFORMED
            or entry["mean_question_tokens"] > ABORT_LENGTH_FACTOR * initial_length
            or entry["duplicate_rate"] - initial_dup > ABORT_DUP_DELTA
        ):
            stop_reason = "DEGENERATION"

    progress.close()
    save_checkpoint(policy, processor, optimizer, scheduler, step, output_dir, "step_final")
    checkpoint_manifests.append("step_final")
    log_handle.close()

    dev_manifest = cfg.get("dev_manifest")
    if dev_manifest:
        dev = [json.loads(line) for line in Path(dev_manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
        dev_images = [row.get("image_path") or row["image"] for row in dev]
        greedy = _generate(policy, processor, dev_images, prompt, cfg, int(cfg["seed"]), 1, 0.0)
        with Path(cfg["greedy_out"]).open("w", encoding="utf-8") as handle:
            for row, image, questions in zip(dev, dev_images, greedy):
                handle.write(json.dumps({"image_id": str(row["image_id"]), "image": image, "question": questions[0], "source": cfg["run_name"]}, ensure_ascii=False) + "\n")
        sampled = _generate(policy, processor, dev_images, prompt, cfg, int(cfg["seed"]), group_size, float(cfg["temperature"]))
        with Path(cfg["sampled_out"]).open("w", encoding="utf-8") as handle:
            for row, image, questions in zip(dev, dev_images, sampled):
                for generation_id, question in enumerate(questions):
                    handle.write(json.dumps({"image_id": str(row["image_id"]), "image": image, "question": question, "generation_id": generation_id, "source": cfg["run_name"]}, ensure_ascii=False) + "\n")

    summary = {
        "variant": cfg["variant"],
        "steps": step,
        "max_steps": max_steps,
        "inner_epochs": inner_epochs,
        "stop_reason": stop_reason,
        "checkpoints": checkpoint_manifests,
        "parameters": parameter_report,
        "wall_seconds": time.perf_counter() - started,
        "peak_gpu_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None,
        "peak_gpu_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3) if torch.cuda.is_available() else None,
        "final_u_released_features": int((u_current > 0).sum()),
        "final_u_sum": float(u_current.sum()),
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
