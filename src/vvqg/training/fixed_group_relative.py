"""Fixed-rollout group-relative policy optimization (Phase 3A pilot).

This is NOT on-policy GRPO: the four completions per image are frozen rollouts
generated once by the Final SFT behavior policy, and the behavior/reference
token log-probabilities are precomputed from that same frozen policy.  Only the
group-relative canonicalization and the PPO-style clipped surrogate live here.

Pure functions (no torch model, no GPU) are separated from the training loop so
they can be unit-tested directly.
"""

from __future__ import annotations

import json
import math
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from vvqg.generation.inference import generate_questions, model_prompt
from vvqg.training.policy_logprobs import encode_group, encode_items, gather_token_logprobs, to_device

RANK_REWARDS: tuple[float, float, float, float] = (1.5, 0.5, -0.5, -1.5)
ADMISSIBLE = "NO_CONFIRMED_VIOLATION"
VIOLATION = "CONFIRMED_VIOLATION"
UNCERTAIN = "UNCERTAIN"
FAILED = "FAILED"
SUCCESS_STATES = (ADMISSIBLE, VIOLATION, UNCERTAIN)
DUPLICATE_UVFC_TOL = 1e-6
ZERO_VARIANCE_TOL = 1e-8
CLIP_EPSILON = 0.2
LOG_RATIO_CLAMP = 20.0


def norm_question(question: str) -> str:
    """Canonical duplicate key body: NFKC, strip, collapse whitespace, casefold."""
    return " ".join(unicodedata.normalize("NFKC", question).split()).casefold()


def group_key(image_id: str) -> str:
    return str(image_id)


def validate_rollouts(rows: Sequence[dict[str, Any]], group_size: int = 4) -> dict[str, Any]:
    """Assert the frozen rollout structure; raise on any violation."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row["image_id"])].append(row)
    if len(groups) == 0:
        raise ValueError("empty rollout file")
    sizes = Counter(len(v) for v in groups.values())
    if set(sizes) != {group_size}:
        raise ValueError(f"group sizes not uniform {group_size}: {dict(sizes)}")
    for image_id, members in groups.items():
        if len({str(m["image"]) for m in members}) != 1:
            raise ValueError(f"group {image_id} spans multiple source images")
        if len({int(m["generation_id"]) for m in members}) != group_size:
            raise ValueError(f"group {image_id} has duplicate generation ids")
        for member in members:
            if not str(member["question"]).strip():
                raise ValueError(f"group {image_id} has an empty question")
    candidate_ids = [f"{group_key(r['image_id'])}#{int(r['generation_id'])}" for r in rows]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate ids are not globally unique")
    return {
        "rows": len(rows),
        "groups": len(groups),
        "candidates_per_group": group_size,
        "unique_images": len({str(r["image"]) for r in rows}),
    }


def canonicalize_group(members: Sequence[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    """Return training rows with canonical rewards/advantages for one group.

    ``mode`` is ``p1`` (UVFC ordering) or ``p2`` (violation-first lexicographic,
    then UVFC).  Inputs are not mutated.  Raises on inconsistent duplicates.
    """
    if mode not in ("p1", "p2"):
        raise ValueError(f"unknown mode {mode!r}")
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for member in members:
        by_question[norm_question(member["question"])].append(member)

    rows: list[dict[str, Any]] = []
    for key, copies in by_question.items():
        # The UVFC consistency STOP applies to *exact* duplicate questions: only
        # those are truly identical responses that must not receive contradictory
        # rewards.  Case/whitespace variants share the canonical key but tokenize
        # differently and may legitimately score differently.
        exact: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for copy in copies:
            exact[str(copy["question"])].append(copy)
        for variant in exact.values():
            values = [float(c["UVFC"]) for c in variant]
            if max(values) - min(values) > DUPLICATE_UVFC_TOL:
                raise ValueError(f"UVFC_DUPLICATE_INCONSISTENCY for {key!r}: {values}")
            canonical_uvfc = values[0]
            for copy in variant:
                copy["UVFC"] = canonical_uvfc

        training_state = _canonical_flat_state([str(c["flat_state"]) for c in copies], key)
        for copy in copies:
            rows.append(
                {
                    **copy,
                    "original_flat_state": str(copy["flat_state"]),
                    "training_flat_state": training_state,
                    "is_duplicate": len(copies) > 1,
                    "duplicate_key": key,
                }
            )

    if mode == "p1":
        rows.sort(key=lambda r: -float(r["UVFC"]))
    else:
        rows.sort(key=lambda r: (0 if r["training_flat_state"] != VIOLATION else 1, -float(r["UVFC"])))

    rewards = assign_tie_rewards(rows, mode)
    mean, std = _mean_std(rewards)
    advantage = [(r - mean) / (std + 1e-8) for r in rewards] if std > ZERO_VARIANCE_TOL else [0.0] * len(rewards)
    for row, reward, adv, raw in zip(rows, rewards, advantage, (r["UVFC"] for r in rows)):
        row["original_reward"] = float(row.get("rank_reward", 0.0))
        row["canonical_reward"] = float(reward)
        row["advantage"] = float(adv)
        row["zero_variance_group"] = std <= ZERO_VARIANCE_TOL
        row["candidate_id"] = f"{group_key(row['image_id'])}#{int(row['generation_id'])}"
        row["group_id"] = group_key(row["image_id"])
    _annotate_tie_blocks(rows, mode)
    return rows


def _canonical_flat_state(states: Sequence[str], key: str) -> str:
    successful = {s for s in states if s in SUCCESS_STATES}
    if len(successful) > 1:
        raise ValueError(f"FLAT_DUPLICATE_INCONSISTENCY for {key!r}: {sorted(successful)}")
    if successful:
        return successful.pop()
    if set(states) == {FAILED}:
        return FAILED
    raise ValueError(f"unexpected flat states for {key!r}: {states}")


def _tie_key(row: dict[str, Any], mode: str) -> tuple:
    if mode == "p1":
        return (float(row["UVFC"]),)
    return (0 if row["training_flat_state"] != VIOLATION else 1, float(row["UVFC"]))


def assign_tie_rewards(rows: Sequence[dict[str, Any]], mode: str) -> list[float]:
    """Mean of the rank-slot rewards occupied by each equal-sort-key tie block."""
    rewards: list[float] = [0.0] * len(rows)
    start = 0
    while start < len(rows):
        key = _tie_key(rows[start], mode)
        stop = start + 1
        while stop < len(rows) and _tie_key(rows[stop], mode) == key:
            stop += 1
        block_mean = sum(RANK_REWARDS[start:stop]) / (stop - start)
        for index in range(start, stop):
            rewards[index] = block_mean
        start = stop
    return rewards


def _annotate_tie_blocks(rows: list[dict[str, Any]], mode: str) -> None:
    start = 0
    while start < len(rows):
        key = _tie_key(rows[start], mode)
        stop = start + 1
        while stop < len(rows) and _tie_key(rows[stop], mode) == key:
            stop += 1
        for index in range(start, stop):
            rows[index]["tie_block_size"] = stop - start
        start = stop


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(variance)


def canonicalize(rows: Sequence[dict[str, Any]], mode: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Canonicalize all groups and return (training rows, summary)."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row["image_id"])].append(row)
    training: list[dict[str, Any]] = []
    stats = Counter()
    for _, members in sorted(groups.items()):
        group_rows = canonicalize_group(members, mode)
        training.extend(group_rows)
        stats["groups"] += 1
        stats["failed_repairs"] += sum(
            r["training_flat_state"] != r["original_flat_state"] for r in group_rows
        )
        clusters: dict[str, set[float]] = defaultdict(set)
        for r in group_rows:
            clusters[r["duplicate_key"]].add(float(r["UVFC"]))
        stats["norm_dup_uvfc_inconsistent"] += sum(len(v) > 1 for v in clusters.values())
        if any(r["is_duplicate"] for r in group_rows):
            stats["duplicate_groups"] += 1
        if any(r["tie_block_size"] > 1 for r in group_rows):
            stats["tie_groups"] += 1
        if group_rows[0]["zero_variance_group"]:
            stats["zero_variance_groups"] += 1
        stats["usable_groups"] += 0 if group_rows[0]["zero_variance_group"] else 1
        if mode == "p2":
            states = {r["training_flat_state"] for r in group_rows}
            if states == {VIOLATION}:
                stats["all_violation_groups"] += 1
            elif VIOLATION not in states:
                stats["all_admissible_groups"] += 1
            else:
                stats["mixed_violation_groups"] += 1
            for r in group_rows:
                if r["training_flat_state"] == FAILED:
                    stats["failed_responses"] += 1
                elif r["training_flat_state"] == VIOLATION:
                    stats["confirmed_violations"] += 1
                else:
                    stats["admissible_responses"] += 1
    summary = {
        "total_groups": stats["groups"],
        "tie_groups": stats["tie_groups"],
        "duplicate_groups": stats["duplicate_groups"],
        "zero_variance_groups": stats["zero_variance_groups"],
        "usable_groups": stats["usable_groups"],
        "total_responses": len(training),
        "failed_flat_state_repairs": stats["failed_repairs"],
        "normalized_duplicate_uvfc_inconsistent_clusters": stats["norm_dup_uvfc_inconsistent"],
    }
    if mode == "p2":
        summary.update(
            mixed_violation_groups=stats["mixed_violation_groups"],
            all_admissible_groups=stats["all_admissible_groups"],
            all_violation_groups=stats["all_violation_groups"],
            confirmed_violations=stats["confirmed_violations"],
            admissible_responses=stats["admissible_responses"],
            failed_responses=stats["failed_responses"],
        )
    return training, summary


# --------------------------------------------------------------------------- #
# Objective pieces (tensor in -> tensor out)                                   #
# --------------------------------------------------------------------------- #
def importance_ratio(logp_current: Any, logp_old: Any) -> Any:
    """exp(clamp(logp_current - logp_old)) for numerical stability only."""
    import torch

    return torch.exp(torch.clamp(logp_current - logp_old, -LOG_RATIO_CLAMP, LOG_RATIO_CLAMP))


def clipped_surrogate(ratio: Any, advantage: Any, clip_epsilon: float = CLIP_EPSILON) -> Any:
    """min(ratio*A, clip(ratio, 1-eps, 1+eps)*A)."""
    import torch

    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
    return torch.minimum(unclipped, clipped)


def kl_token_estimator(logp_ref: Any, logp_current: Any) -> Any:
    """Non-negative estimator exp(ref-current) - (ref-current) - 1."""
    import torch

    ref_ratio = logp_ref - logp_current
    return torch.exp(ref_ratio) - ref_ratio - 1.0


def masked_mean(values: Any, mask: Any) -> Any:
    """Mean over mask==1 positions, safe when the mask is empty."""
    import torch

    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


# --------------------------------------------------------------------------- #
# Runtime trainer (one implementation for P1 and P2)                           #
# --------------------------------------------------------------------------- #
def load_training_groups(path: Path) -> dict[str, list[dict[str, Any]]]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["zero_variance_group"]:
            continue
        groups[row["group_id"]].append(row)
    for members in groups.values():
        members.sort(key=lambda row: int(row["generation_id"]))
    return dict(groups)


def load_old_logprobs(cache_dir: Path) -> dict[str, dict[str, Any]]:
    path = Path(cache_dir) / "logprobs.jsonl"
    return {row["candidate_id"]: row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}


def count_parameters(model: Any) -> dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "trainable_pct": 100.0 * trainable / total}


def _percentile(values: Any, fraction: float) -> float:
    import torch

    return float(torch.quantile(values, fraction))


def measure_current_logprobs(
    model: Any, processor: Any, rows: Sequence[dict[str, Any]], device: Any, instruction: str, batch_size: int = 16
) -> dict[str, list[float]]:
    """Current-policy per-completion-token log-probabilities, keyed by candidate_id."""
    import torch

    values: dict[str, list[float]] = {}
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        encoded = encode_items(processor, [(row["image"], row["question"]) for row in chunk], instruction)
        inputs = to_device(encoded["inputs"], device)
        counts = [len(ids) for ids in encoded["completion_token_ids"]]
        with torch.no_grad():
            logits = model(**inputs, use_cache=False).logits
        per_token = gather_token_logprobs(logits, inputs["input_ids"], encoded["completion_start"], counts)
        for row, lp in zip(chunk, per_token):
            values[row["candidate_id"]] = lp.float().cpu().tolist()
    return values


def _masked_token_lengths(processor: Any, questions: Sequence[str]) -> list[int]:
    return [len(processor.tokenizer(question, add_special_tokens=False)["input_ids"]) for question in questions]


def _question_stats(processor: Any, questions: Sequence[str]) -> dict[str, float]:
    lengths = _masked_token_lengths(processor, questions)
    normalized = [norm_question(question) for question in questions]
    total = len(questions)
    counts = Counter(normalized)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    return {
        "n": total,
        "mean_token_length": sum(lengths) / total,
        "p95_token_length": sorted(lengths)[min(total - 1, int(0.95 * total))],
        "question_mark_rate": sum(question.strip().endswith("?") for question in questions) / total,
        "malformed_rate": sum(len(question.strip()) < 2 for question in questions) / total,
        "unique_rate": len(set(normalized)) / total,
        "normalized_duplicate_rate": duplicates / total,
    }


def build_trainable_model(base_model: str, checkpoint: str, attn_implementation: str = "sdpa"):
    """Fresh trainable copy of the frozen Final SFT state (base + LoRA adapter)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(base_model, local_files_only=True, use_fast=False)
    processor.tokenizer.padding_side = "right"
    base = AutoModelForImageTextToText.from_pretrained(
        base_model, dtype=torch.bfloat16, local_files_only=True, attn_implementation=attn_implementation
    )
    model = PeftModel.from_pretrained(base, checkpoint, is_trainable=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, processor, device


def save_checkpoint(model: Any, processor: Any, output_dir: Path, tag: str) -> str:
    target = Path(output_dir) / "checkpoints" / tag
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(target))
    processor.save_pretrained(str(target))
    return str(target)


def _save_final(model: Any, processor: Any, output_dir: Path) -> str:
    target = Path(output_dir) / "checkpoint_final"
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(target))
    processor.save_pretrained(str(target))
    return str(target)


def _group_step(model: Any, processor: Any, members: Sequence[dict[str, Any]], old: dict[str, Any], device: Any, cfg: dict[str, Any]):
    """Return (loss, per-response tensors, ratio tensor, advantage/len lists)."""
    import torch

    encoded = encode_group(processor, members[0]["image"], [row["question"] for row in members], cfg["instruction"])
    inputs = to_device(encoded["inputs"], device)
    counts = [len(ids) for ids in encoded["completion_token_ids"]]
    logits = model(**inputs, use_cache=False).logits
    current = gather_token_logprobs(logits, inputs["input_ids"], encoded["completion_start"], counts)
    response_surrogate, response_kl, ratios, advantages, lengths = [], [], [], [], []
    for row, current_tokens in zip(members, current):
        cached = old[row["candidate_id"]]
        old_tokens = torch.tensor(cached["old_logprobs"], device=device, dtype=torch.float32)
        advantage = torch.tensor(float(row["advantage"]), device=device)
        log_ratio = torch.clamp(current_tokens - old_tokens, -cfg["log_ratio_clamp"], cfg["log_ratio_clamp"])
        ratio = torch.exp(log_ratio)
        surrogate = clipped_surrogate(ratio, advantage, cfg["clip_epsilon"])
        response_surrogate.append(surrogate.mean())
        ref_ratio = old_tokens - current_tokens
        response_kl.append((torch.exp(ref_ratio) - ref_ratio - 1.0).mean())
        ratios.append(ratio.detach())
        advantages.append(float(row["advantage"]))
        lengths.append(int(current_tokens.numel()))
    policy_loss = -torch.stack(response_surrogate).mean()
    mean_kl = torch.stack(response_kl).mean()
    loss = policy_loss + cfg["beta_kl"] * mean_kl
    return loss, policy_loss.detach(), mean_kl.detach(), ratios, advantages, lengths


def _parity(model: Any, processor: Any, old: dict[str, Any], groups: dict[str, list[dict[str, Any]]], device: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    import torch

    sample = random.Random(42).sample(sorted(groups), min(max(1, cfg["audit_responses"] // 4), len(groups)))
    error = 0.0
    responses = 0
    for group_id in sample:
        members = groups[group_id]
        encoded = encode_group(processor, members[0]["image"], [row["question"] for row in members], cfg["instruction"])
        inputs = to_device(encoded["inputs"], device)
        counts = [len(ids) for ids in encoded["completion_token_ids"]]
        with torch.no_grad():
            logits = model(**inputs, use_cache=False).logits
        per_token = gather_token_logprobs(logits, inputs["input_ids"], encoded["completion_start"], counts)
        for row, current in zip(members, per_token):
            cached = old[row["candidate_id"]]["old_logprobs"]
            values = current.float().cpu().tolist()
            error = max(error, *(abs(a - b) for a, b in zip(cached, values)))
            responses += 1
    return {
        "responses": responses,
        "seed": 42,
        "max_abs_token_logprob_error": error,
        "tolerance_1e-4": error <= 1e-4,
        "pass": error <= 1e-4,
        "batching": "one forward per group (matches frozen precompute)",
    }


def run_training(cfg: dict[str, Any]) -> dict[str, Any]:
    import torch
    from transformers import get_cosine_schedule_with_warmup

    seed = int(cfg["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np

    np.random.seed(seed)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str) + "\n", encoding="utf-8")

    raw_groups = {
        json.loads(line)["group_id"]
        for line in Path(cfg["training_rollouts"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    groups = load_training_groups(Path(cfg["training_rollouts"]))
    skipped_groups = len(raw_groups) - len(groups)
    keys = sorted(groups)
    if cfg.get("limit_groups"):
        keys = keys[: cfg["limit_groups"]]
    random.Random(cfg["seed"]).shuffle(keys)
    order_hash = __import__("hashlib").sha256(",".join(keys).encode()).hexdigest()
    old = load_old_logprobs(Path(cfg["old_logprobs"]))
    missing = [row["candidate_id"] for key in keys for row in groups[key] if row["candidate_id"] not in old]
    if missing:
        raise RuntimeError(f"old logprobs missing for {len(missing)} responses")

    model, processor, device = build_trainable_model(cfg["base_model"], cfg["sft_checkpoint"], cfg.get("attn_implementation", "sdpa"))
    parameter_report = count_parameters(model)
    if cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    start = __import__("time").perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    smoke_dir = Path(cfg["smoke_dir"]) if cfg.get("smoke_dir") else None
    if smoke_dir:
        smoke_dir.mkdir(parents=True, exist_ok=True)
    parity = _parity(model, processor, old, groups, device, cfg) if cfg.get("run_parity", True) else None
    if parity is not None and smoke_dir:
        (smoke_dir / "parity.json").write_text(json.dumps(parity, indent=2) + "\n", encoding="utf-8")
    if parity is not None and not parity["pass"]:
        raise RuntimeError(f"step-0 parity failed: {parity}")

    smoke_rows = [row for key in keys for row in groups[key]] if cfg.get("directional") else []
    before = measure_current_logprobs(model, processor, smoke_rows, device, cfg["instruction"]) if smoke_rows else None

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg["lr"], betas=tuple(cfg.get("betas", (0.9, 0.999))), weight_decay=cfg.get("weight_decay", 0.0))
    accumulation = int(cfg["grad_accum_groups"])
    total_steps = math.ceil(len(keys) / accumulation) * int(cfg["epochs"])
    if cfg.get("max_steps"):
        total_steps = int(cfg["max_steps"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * cfg.get("warmup_ratio", 0.05)), total_steps)

    monitor_rows = [json.loads(line) for line in Path(cfg["monitor_file"]).read_text().splitlines() if line.strip()] if cfg.get("monitor_file") else []
    monitor_images = [row.get("image_path") or row["image"] for row in monitor_rows]
    monitor_steps = {0, total_steps, round(0.25 * total_steps), round(0.5 * total_steps), round(0.75 * total_steps)}
    monitor_labels = {0: "step0", round(0.25 * total_steps): "quarter", round(0.5 * total_steps): "half", round(0.75 * total_steps): "three_quarter", total_steps: "final"}
    save_steps = {step for step in (50, total_steps // 2) if 0 < step < total_steps}
    monitor_baseline = None
    degeneration_streak = 0

    prompt = model_prompt(processor, cfg["instruction"])
    log_path = output_dir / "train_log.jsonl"
    log_handle = log_path.open("w", encoding="utf-8")
    recent: list[dict[str, Any]] = []
    stop_reason = None
    step = 0
    pending = 0
    optimizer.zero_grad(set_to_none=True)

    def run_monitor(current_step: int, label: str) -> None:
        nonlocal monitor_baseline, degeneration_streak, stop_reason
        (output_dir / "monitor").mkdir(parents=True, exist_ok=True)
        questions = [q for group in generate_questions(model, processor, monitor_images, prompt, num_samples=1, temperature=0.0, max_new_tokens=32, seed=4242, batch_size=16) for q in group]
        stats = _question_stats(processor, questions)
        with (output_dir / "monitor" / f"{label}.jsonl").open("w", encoding="utf-8") as handle:
            for row, question in zip(monitor_rows, questions):
                handle.write(json.dumps({"image_id": row.get("image_id"), "question": question}, ensure_ascii=False) + "\n")
        if monitor_baseline is None:
            monitor_baseline = stats
            stats["degeneration"] = False
        else:
            triggered = (
                stats["malformed_rate"] > 0.02
                or stats["question_mark_rate"] < 0.95
                or stats["mean_token_length"] > 1.50 * monitor_baseline["mean_token_length"]
                or stats["normalized_duplicate_rate"] - monitor_baseline["normalized_duplicate_rate"] > 0.20
            )
            stats["degeneration"] = bool(triggered)
            degeneration_streak = degeneration_streak + 1 if triggered else 0
            if degeneration_streak >= 2:
                stop_reason = "PILOT_REWARD_HACKING_OR_DEGENERATION"
        (output_dir / "monitor" / f"{label}_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    if monitor_images:
        run_monitor(0, "step0")

    def finish(reason: str | None) -> dict[str, Any]:
        checkpoint = _save_final(model, processor, output_dir)
        summary = {
            "checkpoint": checkpoint,
            "steps": step,
            "total_steps": total_steps,
            "skipped_zero_variance_groups": skipped_groups,
            "usable_groups": len(keys) * int(cfg["epochs"]),
            "parameters": parameter_report,
            "order_hash": order_hash,
            "parity": parity,
            "early_stop": reason is not None,
            "stop_reason": reason,
            "wall_seconds": __import__("time").perf_counter() - start,
            "peak_gpu_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None,
            "peak_gpu_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3) if torch.cuda.is_available() else None,
        }
        (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        runtime = {"wall_seconds": summary["wall_seconds"], "steps": step, "peak_gpu_allocated_gb": summary["peak_gpu_allocated_gb"], "peak_gpu_reserved_gb": summary["peak_gpu_reserved_gb"]}
        (output_dir / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
        return summary

    step_ratio: list[Any] = []
    step_advantage: list[float] = []
    step_lengths: list[int] = []
    step_policy = 0.0
    step_kl = 0.0
    step_loss = 0.0
    groups_in_step = 0
    stop_reason_local = None

    def do_step() -> bool:
        nonlocal step, pending, step_ratio, step_advantage, step_lengths, step_policy, step_kl, step_loss, groups_in_step, stop_reason_local
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        ratio_tensor = torch.cat(step_ratio)
        entry = {
            "step": step,
            "total_loss": step_loss / groups_in_step,
            "policy_loss": step_policy / groups_in_step,
            "reference_kl": step_kl / groups_in_step,
            "grad_norm": grad_norm,
            "lr": scheduler.get_last_lr()[0],
            "ratio_mean": float(ratio_tensor.mean()),
            "ratio_p05": _percentile(ratio_tensor, 0.05),
            "ratio_p50": _percentile(ratio_tensor, 0.50),
            "ratio_p95": _percentile(ratio_tensor, 0.95),
            "ratio_max": float(ratio_tensor.max()),
            "clip_fraction": float(((ratio_tensor < 1 - cfg["clip_epsilon"]) | (ratio_tensor > 1 + cfg["clip_epsilon"])).float().mean()),
            "advantage_mean": sum(step_advantage) / len(step_advantage),
            "advantage_std": float(torch.tensor(step_advantage).std(unbiased=False)),
            "advantage_min": min(step_advantage),
            "advantage_max": max(step_advantage),
            "mean_response_tokens": sum(step_lengths) / len(step_lengths),
            "max_response_tokens": max(step_lengths),
            "peak_gpu_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None,
        }
        log_handle.write(json.dumps(entry) + "\n")
        log_handle.flush()
        recent.append(entry)
        recent[:] = recent[-10:]
        step_ratio, step_advantage, step_lengths = [], [], []
        step_loss = step_policy = step_kl = 0.0
        groups_in_step = 0
        pending = 0

        if step in save_steps:
            save_checkpoint(model, processor, output_dir, f"step_{step}")
        if monitor_images and step in monitor_steps:
            run_monitor(step, monitor_labels[step])

        if not math.isfinite(entry["total_loss"]) or not math.isfinite(entry["grad_norm"]):
            stop_reason_local = "NUMERICAL_FAILURE"
        elif len(recent) == 10 and all(row["clip_fraction"] > 0.50 and row["ratio_p95"] > 2.5 for row in recent):
            stop_reason_local = "PILOT_POLICY_DRIFT"
        elif len(recent) == 10 and all(row["reference_kl"] > 0.20 for row in recent):
            stop_reason_local = "PILOT_POLICY_DRIFT"
        elif stop_reason == "PILOT_REWARD_HACKING_OR_DEGENERATION":
            stop_reason_local = stop_reason
        return bool(stop_reason_local) or (bool(cfg.get("max_steps")) and step >= int(cfg["max_steps"]))

    for _ in range(int(cfg["epochs"])):
        for key in keys:
            loss, policy_loss, mean_kl, ratios, advantages, lengths = _group_step(model, processor, groups[key], old, device, cfg)
            (loss / accumulation).backward()
            pending += 1
            groups_in_step += 1
            step_loss += float(loss.detach())
            step_policy += float(policy_loss)
            step_kl += float(mean_kl)
            step_ratio.extend(ratios)
            step_advantage.extend(advantages)
            step_lengths.extend(lengths)
            if pending == accumulation and do_step():
                break
        if pending > 0 and not stop_reason_local:
            do_step()
        if stop_reason_local or (cfg.get("max_steps") and step >= int(cfg["max_steps"])):
            break

    if before is not None:
        after = measure_current_logprobs(model, processor, smoke_rows, device, cfg["instruction"])
        positive = [after[row["candidate_id"]][-1] - before[row["candidate_id"]][-1] for row in smoke_rows if row["advantage"] > 0]
        negative = [after[row["candidate_id"]][-1] - before[row["candidate_id"]][-1] for row in smoke_rows if row["advantage"] < 0]
        deltas = {
            "delta_logp_sum_positive_mean": sum(positive) / len(positive),
            "delta_logp_sum_negative_mean": sum(negative) / len(negative),
            "positive_minus_negative": sum(positive) / len(positive) - sum(negative) / len(negative),
        }
        deltas["direction_ok"] = deltas["positive_minus_negative"] > 0
        (smoke_dir / "directional_gradient.json").write_text(json.dumps(deltas, indent=2) + "\n", encoding="utf-8")
        if not deltas["direction_ok"]:
            stop_reason_local = "DIRECTIONAL_GRADIENT_REVERSED"

    log_handle.close()
    summary = finish(stop_reason_local)
    if before is not None:
        (smoke_dir / "summary.json").write_text(json.dumps({"steps": step, "stop_reason": stop_reason_local}, indent=2) + "\n", encoding="utf-8")

    if cfg.get("dev_manifest") and not stop_reason_local == "NUMERICAL_FAILURE":
        dev = [json.loads(line) for line in Path(cfg["dev_manifest"]).read_text().splitlines() if line.strip()]
        images = [row.get("image_path") or row["image"] for row in dev]
        greedy = generate_questions(model, processor, images, prompt, num_samples=1, temperature=0.0, max_new_tokens=32, seed=42, batch_size=16)
        with Path(cfg["greedy_out"]).open("w", encoding="utf-8") as handle:
            for row, image, group in zip(dev, images, greedy):
                handle.write(json.dumps({"image_id": str(row["image_id"]), "image": image, "question": group[0], "source": cfg["run_name"]}, ensure_ascii=False) + "\n")
        sampled = generate_questions(model, processor, images, prompt, num_samples=4, temperature=0.8, top_p=0.9, max_new_tokens=32, seed=42, batch_size=8)
        with Path(cfg["sampled_out"]).open("w", encoding="utf-8") as handle:
            for row, questions in zip(dev, sampled):
                image = row.get("image_path") or row["image"]
                for generation_id, question in enumerate(questions):
                    handle.write(json.dumps({"image_id": str(row["image_id"]), "image": image, "question": question, "generation_id": generation_id, "source": cfg["run_name"]}, ensure_ascii=False) + "\n")
    return summary
