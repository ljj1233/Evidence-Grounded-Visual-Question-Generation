"""Support-Delta GRPO trainer (corrected group-relative policy optimization).

old_logprob is computed once per rollout group *before* any optimizer update and
reused across inner epochs; KL reference is the frozen pi_0.  Reward is the
Support-Delta objective (no delta_cf, no lp_plus, no external labels).
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Sequence

from vvqg.generation.inference import generate_questions, model_prompt
from vvqg.rewards.support.structured_flat import StructuredFlatVerifier, support_summary
from vvqg.rewards.visual_information import uvfc as uvfc_mod
from vvqg.support_delta.reward import (
    VIOLATION,
    duplicate_flags,
    group_advantages,
    support_delta_reward,
    support_score,
)
from vvqg.training.fixed_group_relative import build_trainable_model, count_parameters, norm_question
from vvqg.training.g3_reward import generic_flags
from vvqg.training.online_uvfc_relative_grpo import (
    _forward_logprobs,
    _group_terms,
    _question_stats,
    build_frozen_model,
    save_checkpoint,
)
from vvqg.training.policy_logprobs import encode_group, gather_token_logprobs, to_device

DEFAULTS = {
    "instruction": "Generate a question about this image.",
    "group_size": 8,
    "groups_per_step": 1,
    "max_steps": 300,
    "inner_epochs": 2,
    "temperature": 0.8,
    "top_p": 0.9,
    "max_new_tokens": 32,
    "generation_batch_size": 8,
    "flat_batch_size": 32,
    "lr": 5e-7,
    "betas": (0.9, 0.999),
    "weight_decay": 0.0,
    "clip_epsilon": 0.2,
    "beta_kl": 0.02,
    "log_ratio_clamp": 20.0,
    "max_grad_norm": 1.0,
    "seed": 42,
    "attn_implementation": "sdpa",
    "checkpoint_steps": [0, 50, 100, 150, 200, 250, 300],
}

ABORT_KL_HARD = 0.02
ABORT_MALFORMED = 0.05
ABORT_GENERIC_DELTA = 0.05
ABORT_DUP_DELTA = 0.2
DEGEN_STREAK = 3


def _calibrate_duplicate(model: Any, processor: Any, images: Sequence[str], prompt: str, cfg: dict) -> float:
    sampled = generate_questions(
        model, processor, images, prompt, num_samples=int(cfg["group_size"]), temperature=float(cfg["temperature"]),
        top_p=float(cfg["top_p"]), max_new_tokens=int(cfg["max_new_tokens"]), seed=int(cfg["seed"]),
        batch_size=int(cfg["generation_batch_size"]),
    )
    rates = []
    for group in sampled:
        normalized = [norm_question(q) for q in group]
        rates.append((len(group) - len(set(normalized))) / len(group))
    return sum(rates) / len(rates) if rates else 0.0


def _compute_q0_batch(model: Any, processor: Any, prompt: str, flat: Any, frozen: Any, device: Any,
                      images: Sequence[str], image_ids: Sequence[str], cfg: dict) -> list[dict]:
    generated = generate_questions(
        model, processor, images, prompt, num_samples=1, temperature=0.0, top_p=1.0,
        max_new_tokens=int(cfg["max_new_tokens"]), seed=int(cfg["seed"]), batch_size=int(cfg["generation_batch_size"]),
    )
    records = []
    for image_id, image, group in zip(image_ids, images, generated):
        question = group[0].strip()
        record = {"image_id": image_id, "image": image, "question": question, "flat_state": None,
                  "support_floor": None, "support_mean": None, "requirement_labels": None,
                  "parser_status": None, "UVFC": None, "support_score": None}
        records.append(record)
    parsed = flat.verify_structured(records)
    for record, parse in zip(records, parsed):
        summary = support_summary(parse)
        record.update(summary)
        record["parser_status"] = parse.get("status")
    for start in range(0, len(records), int(cfg["flat_batch_size"])):
        chunk = records[start : start + int(cfg["flat_batch_size"])]
        for record, score in zip(chunk, uvfc_mod.score_uvfc(model, processor, chunk, frozen, device)):
            record["UVFC"] = float(score["UVFC"])
    for record in records:
        malformed = len(str(record["question"]).strip()) < 2
        record["support_score"] = support_score(record.get("support_floor"), record.get("flat_state"),
                                                malformed or record.get("parser_status") != "ok")
    return records


def _reward_groups(anchors: dict, by_image: dict, processor: Any) -> tuple[list[dict], dict[str, Any]]:
    usable, stats = [], {"guard": 0, "dup": 0, "malformed": 0, "rollouts": 0, "zero_variance": 0,
                         "generic": 0, "lengths": [], "questions": [], "violations": 0}
    for image_id, members in by_image.items():
        q0 = anchors[image_id]
        questions = [member["question"] for member in members]
        duplicates = duplicate_flags(questions)
        uvfc_group = [float(q0["UVFC"])] + [float(member["UVFC"]) for member in members]
        rewards = []
        for index, member in enumerate(members):
            malformed = len(member["question"].strip()) < 2
            parser_failed = member.get("parser_status") != "ok"
            result = support_delta_reward(
                support_floor=member.get("support_floor"), flat_state=member.get("flat_state"), malformed=malformed,
                q0_score=float(q0["support_score"]), uvfc_group=uvfc_group, q0_index=0, rollout_index=index + 1,
                duplicate=duplicates[index], parser_failed=parser_failed,
            )
            member.update(result)
            rewards.append(result["reward"])
            stats["guard"] += int(result["coverage_guard"] > 0)
            stats["dup"] += int(result["duplicate_penalty"] > 0)
            stats["malformed"] += int(result["malformed_penalty"] > 0)
            stats["rollouts"] += 1
        advantages, std, zero = group_advantages(rewards)
        for member, advantage in zip(members, advantages):
            member["advantage"] = advantage
        stats["zero_variance"] += int(zero)
        for member in members:
            stats["questions"].append(member["question"])
            stats["lengths"].append(len(processor.tokenizer(member["question"], add_special_tokens=False)["input_ids"]))
            stats["generic"] += int(generic_flags([member["question"]])[0])
            stats["violations"] += int(member.get("flat_state") == VIOLATION)
        if not zero:
            usable.append({"image_id": image_id, "members": members})
    return usable, stats


def run_pilot(cfg: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch

    resolved = {**DEFAULTS, **cfg}
    seed = int(resolved["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    output_dir = Path(resolved["output_dir"])
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "rollouts").mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.json").write_text(json.dumps({k: v for k, v in resolved.items()}, indent=2, default=str) + "\n", encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = [json.loads(line) for line in Path(resolved["train_manifest"]).read_text().splitlines() if line.strip()]
    random.Random(seed).shuffle(records)
    for row in records:
        row["image_id"] = str(row["image_id"]); row["image"] = str(row.get("image") or row.get("image_path"))

    policy, processor, _ = build_trainable_model(resolved["base_model"], resolved["base_checkpoint"], resolved["attn_implementation"])
    reference, _ = build_frozen_model(resolved["base_model"], resolved["base_checkpoint"], resolved["attn_implementation"])
    reference.to(device).eval(); reference.requires_grad_(False)
    prompt = model_prompt(processor, resolved["instruction"])
    flat = StructuredFlatVerifier(resolved["flat_model"], device, batch_size=int(resolved["flat_batch_size"]))
    frozen = uvfc_mod.load_uvfc(resolved["uvfc_dir"], device)

    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(resolved["lr"]), betas=tuple(resolved["betas"]), weight_decay=float(resolved["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)

    cal_images = [row["image"] for row in records[:128]]
    cal_dup = _calibrate_duplicate(reference, processor, cal_images, prompt, resolved)
    t_dup = cal_dup + ABORT_DUP_DELTA
    calibration = {"calibration_images": 128, "duplicate_rate_pi0": cal_dup, "t_dup": t_dup}

    checkpoint_steps = set(int(s) for s in resolved["checkpoint_steps"])
    save_checkpoint(policy, processor, optimizer, scheduler, 0, output_dir, "step_0000")
    checkpoints = ["step_0000"]

    anchors: dict[str, dict] = {}
    anchor_handle = (output_dir / "BASELINE_ANCHORS.jsonl").open("a", encoding="utf-8")
    log_handle = (output_dir / "train_log.jsonl").open("w", encoding="utf-8")

    K = int(resolved["group_size"])
    G = int(resolved["groups_per_step"])
    max_steps = int(resolved["max_steps"])
    inner_epochs = int(resolved["inner_epochs"])
    cursor = 0
    step = 0
    stop_reason = None
    recent: list[dict] = []
    checkpoint_dup: list[float] = []
    checkpoint_generic: list[float] = []
    initial_generic = None
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    from tqdm.auto import tqdm
    progress = tqdm(total=max_steps, desc="support-delta grpo", unit="step", dynamic_ncols=True)

    try:
        while step < max_steps and stop_reason is None:
            if cursor + G > len(records):
                random.Random(seed + step).shuffle(records); cursor = 0
            batch = records[cursor : cursor + G]; cursor += G
            step += 1
            images = [row["image"] for row in batch]
            image_ids = [row["image_id"] for row in batch]

            new = [index for index, image_id in enumerate(image_ids) if image_id not in anchors]
            if new:
                new_records = _compute_q0_batch(reference, processor, prompt, flat, frozen, device,
                                                [images[i] for i in new], [image_ids[i] for i in new], resolved)
                for record in new_records:
                    anchors[record["image_id"]] = record
                    anchor_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                anchor_handle.flush()

            generation_seed = seed * 1_000_003 + step
            sampled = generate_questions(policy, processor, images, prompt, num_samples=K, temperature=float(resolved["temperature"]),
                                         top_p=float(resolved["top_p"]), max_new_tokens=int(resolved["max_new_tokens"]),
                                         seed=generation_seed, batch_size=int(resolved["generation_batch_size"]))
            rollout_rows = []
            for image, questions in zip(images, sampled):
                for generation_id, question in enumerate(questions):
                    rollout_rows.append({"image_id": image, "image": image, "question": question.strip(), "generation_id": generation_id})
            parsed = flat.verify_structured(rollout_rows)
            for row, parse in zip(rollout_rows, parsed):
                summary = support_summary(parse); row.update(summary); row["parser_status"] = parse.get("status")
            for start in range(0, len(rollout_rows), int(resolved["flat_batch_size"])):
                chunk = rollout_rows[start : start + int(resolved["flat_batch_size"])]
                for row, score in zip(chunk, uvfc_mod.score_uvfc(reference, processor, chunk, frozen, device)):
                    row["UVFC"] = float(score["UVFC"])

            by_image: dict[str, list[dict]] = {}
            for row in rollout_rows:
                by_image.setdefault(row["image_id"], []).append(row)
            usable, stats = _reward_groups(anchors, by_image, processor)
            if not usable:
                for row in rollout_rows:
                    row["advantage"] = 0.0

            # Precompute old/ref logprobs ONCE per group (before any update).
            prepared = []
            for group in usable:
                members = group["members"]
                encoded = encode_group(processor, members[0]["image"], [m["question"] for m in members], resolved["instruction"])
                inputs = to_device(encoded["inputs"], device)
                with torch.no_grad():
                    old_tokens, _ = _forward_logprobs(policy, inputs, encoded, with_grad=False)
                    ref_tokens, _ = _forward_logprobs(reference, inputs, encoded, with_grad=False)
                prepared.append({
                    "encoded": encoded, "inputs": inputs,
                    "old_lp": [value.detach().float().cpu() for value in old_tokens],
                    "ref_lp": [value.detach().float().cpu() for value in ref_tokens],
                    "advantages": [float(m["advantage"]) for m in members],
                })
                for member, old_value in zip(members, prepared[-1]["old_lp"]):
                    member["policy_logprob_sum"] = float(old_value.sum())

            step_losses, step_kls, step_policy, step_ratios = [], [], [], []
            for inner in range(inner_epochs):
                optimizer.zero_grad(set_to_none=True)
                for item in prepared:
                    loss, terms = _group_terms(policy, item["encoded"], item["inputs"], item["old_lp"], item["ref_lp"], item["advantages"], resolved)
                    (loss / len(prepared)).backward()
                    if inner == inner_epochs - 1:
                        step_losses.append(terms["loss"]); step_kls.append(terms["reference_kl"])
                        step_policy.append(terms["policy_loss"]); step_ratios.append(terms["ratios"])
                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, float(resolved["max_grad_norm"])))
                optimizer.step()

            ratio_tensor = torch.cat(step_ratios) if step_ratios else torch.ones(1)
            qstats = _question_stats(processor, stats["questions"])
            generic_rate = stats["generic"] / stats["rollouts"]
            if initial_generic is None:
                initial_generic = generic_rate
            entry = {
                "step": step, "loss": sum(step_losses) / len(step_losses) if step_losses else 0.0,
                "policy_loss": sum(step_policy) / len(step_policy) if step_policy else 0.0,
                "reference_kl": sum(step_kls) / len(step_kls) if step_kls else 0.0,
                "no_op_step": not bool(prepared),
                "grad_norm": grad_norm, "lr": float(resolved["lr"]),
                "ratio_mean": float(ratio_tensor.mean()), "ratio_p01": float(torch.quantile(ratio_tensor, 0.01)),
                "ratio_p50": float(torch.quantile(ratio_tensor, 0.5)), "ratio_p99": float(torch.quantile(ratio_tensor, 0.99)),
                "clip_fraction": float(((ratio_tensor < 1 - resolved["clip_epsilon"]) | (ratio_tensor > 1 + resolved["clip_epsilon"])).float().mean()),
                "reward_mean": float(np.mean([m["reward"] for group in usable for m in group["members"]])) if usable else 0.0,
                "reward_std": float(np.std([m["reward"] for group in usable for m in group["members"]])) if usable else 0.0,
                "advantage_mean": float(np.mean([m["advantage"] for group in usable for m in group["members"]])) if usable else 0.0,
                "advantage_std": float(np.std([m["advantage"] for group in usable for m in group["members"]])) if usable else 0.0,
                "zero_variance_group_ratio": stats["zero_variance"] / max(len(by_image), 1),
                "coverage_guard_rate": stats["guard"] / max(stats["rollouts"], 1),
                "duplicate_penalty_rate": stats["dup"] / max(stats["rollouts"], 1),
                "duplicate_rate": qstats["duplicate_rate"],
                "malformed_rate": qstats["malformed_rate"],
                "generic_rate": generic_rate,
                "mean_question_tokens": qstats["mean_token_length"],
                "confirmed_violation_rate": stats["violations"] / max(stats["rollouts"], 1),
                "groups_used": len(usable), "responses": stats["rollouts"],
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "peak_gpu_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None,
            }
            log_handle.write(json.dumps(entry) + "\n"); log_handle.flush()
            progress.update(1); progress.set_postfix(loss=f"{entry['loss']:.3g}", KL=f"{entry['reference_kl']:.3g}", ratio=f"{entry['ratio_mean']:.4f}")
            recent.append(entry); recent[:] = recent[-DEGEN_STREAK:]

            if step in checkpoint_steps:
                tag = f"step_{step:04d}"; save_checkpoint(policy, processor, optimizer, scheduler, step, output_dir, tag); checkpoints.append(tag)
                checkpoint_dup.append(entry["duplicate_rate"])
                checkpoint_generic.append(generic_rate)
                if len(checkpoint_dup) >= DEGEN_STREAK and all(value > t_dup for value in checkpoint_dup[-DEGEN_STREAK:]):
                    stop_reason = "DEGENERATION_DUPLICATE"
                if len(checkpoint_generic) >= DEGEN_STREAK and all(value - initial_generic > ABORT_GENERIC_DELTA for value in checkpoint_generic[-DEGEN_STREAK:]):
                    stop_reason = "DEGENERATION_GENERIC"

            finite = all(math.isfinite(entry[key]) for key in ("loss", "reference_kl", "grad_norm"))
            if not finite:
                stop_reason = "NUMERICAL_FAILURE"
            elif len(recent) == DEGEN_STREAK and all(row["reference_kl"] > ABORT_KL_HARD for row in recent):
                stop_reason = "KL_DRIFT"
            elif entry["malformed_rate"] > ABORT_MALFORMED:
                stop_reason = "DEGENERATION_MALFORMED"
    finally:
        log_handle.close(); anchor_handle.close()
        save_checkpoint(policy, processor, optimizer, scheduler, step, output_dir, "step_final")
        checkpoints.append("step_final")

    log = [json.loads(line) for line in (output_dir / "train_log.jsonl").read_text().splitlines() if line.strip()]
    ratios = [row["ratio_mean"] for row in log]
    clips = [row["clip_fraction"] for row in log]
    kls = [row["reference_kl"] for row in log]
    summary = {
        "steps": step, "max_steps": max_steps, "stop_reason": stop_reason, "checkpoints": checkpoints,
        "calibration": calibration, "n_anchor_images": len(anchors), "parameters": count_parameters(policy),
        "ratio_mean_range": [min(ratios), max(ratios)], "clip_fraction_range": [min(clips), max(clips)],
        "kl_range": [min(kls), max(kls)],
        "wall_seconds": time.perf_counter() - started,
        "peak_gpu_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
