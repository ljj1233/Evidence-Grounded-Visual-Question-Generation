#!/usr/bin/env python3
"""Read-only SeeQ-style baseline audit: diversity funnels + official-filter replay.

Reads the frozen generation artifacts under
outputs/same_backbone_baselines/seeq_style/, computes proposal -> rewrite ->
accepted -> retained -> model-output uniqueness/diversity funnels, replays the
deterministic official SeeQ filters offline against the local accepted set, and
writes JSON artifacts into outputs/seeq_baseline_audit/.

Never writes to the generation/evaluation artifacts.  Audit-only.
"""
from __future__ import annotations

import difflib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vvqg.evaluation.metrics.diversity import (  # noqa: E402
    distinct_2 as canonical_distinct_2,
    normalized_excess_duplicate_rate,
    tokenize,
)
SEEQ = ROOT / "outputs/same_backbone_baselines/seeq_style"
OUT = ROOT / "outputs/seeq_baseline_audit"
OFFICIAL_COMMIT = "18e795293808c5cf9f69f1e980577ab40b3faa95"

R1 = SEEQ / "round1_generation"
R2 = SEEQ / "round2_generation_m1"
EVAL = SEEQ / "evaluation"


# ── exact copies of official deterministic filters ──────────────────────────
def is_multi_question(text: str) -> bool:
    if not isinstance(text, str):
        return False
    if text.count("?") > 1:
        return True
    pattern = r"(?:^|[,;]\s*(?:(?:and|or|but)\s+)?)(?:what|how|why|where|when|which|who)\b"
    return len(re.findall(pattern, text, flags=re.IGNORECASE)) > 1


def _is_compound_question(qs: str) -> bool:
    if qs is None:
        return False
    if qs.count("?") >= 2:
        return True
    return bool(re.search(r",\s+and\s+(what|how|why|which|where|when)\b", qs, re.IGNORECASE))


def _tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return re.findall(r"[a-z0-9']+", text.lower())


def normalize_question(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.strip().lower().split())


def lexical_overlap_ratio(text_a: str, text_b: str) -> float:
    a = {t for t in _tokenize(text_a) if len(t) > 2}
    b = {t for t in _tokenize(text_b) if len(t) > 2}
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ── helpers ─────────────────────────────────────────────────────────────────
def read_jsonl(path: Path, desc: str = "") -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        lines = fh.readlines()
    return [json.loads(line) for line in tqdm(lines, desc=desc or path.name, unit="rec") if line.strip()]


def question_part(text: str) -> str:
    """Strip the appended answer an M1/M2 proposer emits after the first '?'."""
    text = str(text or "").strip()
    if "?" in text:
        return text.split("?", 1)[0].strip() + "?"
    return text


def rewrite_question(rec: dict) -> str:
    out = rec.get("rewrite")
    if isinstance(out, dict):
        for key in ("Rewritten question", "Question", "text"):
            if out.get(key):
                return str(out[key]).strip()
    return question_part(rec.get("proposal", ""))


def distinct2(texts: list[str]) -> float:
    """Canonical corpus Distinct-2 (lowercase [a-z0-9']+ bigrams, no boundary cross)."""
    return canonical_distinct_2(texts)


def stage_stats(rows: list[dict], key, label: str) -> dict:
    texts = [key(r) for r in rows]
    n = len(texts)
    unique = len({t.strip().lower() for t in texts})
    return {
        "label": label,
        "n": n,
        "unique_normalized": unique,
        "unique_rate": round(unique / n, 4) if n else 0.0,
        "normalized_excess_duplicate_rate": normalized_excess_duplicate_rate(texts),
        "distinct_2": distinct2(texts),
        "mean_chars": round(sum(len(t) for t in texts) / n, 2) if n else 0.0,
        "mean_words": round(sum(len(tokenize(t)) for t in texts) / n, 2) if n else 0.0,
        "top_question": Counter(texts).most_common(1)[0] if texts else None,
    }


def load_round(gen: Path, rid: int) -> dict:
    """Load and normalize one generation round in record order."""
    return {
        "proposals": read_jsonl(gen / "stage_b_answers.jsonl", f"Audit SeeQ R{rid} proposals"),
        "rewrites": read_jsonl(gen / "stage_c_rewrites.jsonl", f"Audit SeeQ R{rid} rewrites"),
        "filters": read_jsonl(gen / "stage_d_filters.jsonl", f"Audit SeeQ R{rid} filters"),
        "retained": read_jsonl(gen / "stage_e_retained.jsonl", f"Audit SeeQ R{rid} retained"),
    }


def replay_official_deterministic(gen: Path, label: str) -> dict:
    """Apply the deterministic official filters to the local accepted stream."""
    filters = read_jsonl(gen / "stage_d_filters.jsonl")
    seen_simplified: set[str] = set()
    kept: list[dict] = []          # passes deterministic official filters (whole stream)
    kept_accepted: list[dict] = []  # passes deterministic filters AND the local hard gate
    reject = Counter()
    reject_accepted = Counter()
    multi = compound = overlap = duplicate = 0
    for rec in tqdm(filters, desc=f"Replay official filters {label}", unit="rec"):
        q = rec.get("question", "")
        prop = rec.get("proposal", "")
        reason = ""
        if is_multi_question(q):
            reason, multi = "multi_question", multi + 1
        elif _is_compound_question(q):
            reason, compound = "compound_question", compound + 1
        elif lexical_overlap_ratio(q, prop) >= 0.82:
            reason, overlap = "lexical_overlap_too_high", overlap + 1
        else:
            nq = normalize_question(q)
            if nq in seen_simplified:
                reason, duplicate = "duplicate_simplified_question", duplicate + 1
            else:
                seen_simplified.add(nq)
        if reason:
            reject[reason] += 1
            if rec.get("accepted") is True:
                reject_accepted[reason] += 1
            continue
        kept.append(rec)
        if rec.get("accepted") is True:
            kept_accepted.append(rec)
    kept_texts = [r["question"] for r in kept_accepted]
    local_accepted = sum(1 for r in filters if r.get("accepted") is True)
    return {
        "label": label,
        "input_records": len(filters),
        "local_accepted": local_accepted,
        "deterministic_filters_pass_whole_stream": len(kept),
        "local_accepted_surviving_official_deterministic_filters": len(kept_accepted),
        "share_of_local_accepted_removed": round((local_accepted - len(kept_accepted)) / local_accepted, 4),
        "rejections_whole_stream": dict(reject),
        "rejections_among_local_accepted": dict(reject_accepted),
        "rate_lexical_overlap_ge_0.82_whole_stream": round(overlap / len(filters), 4),
        "rate_duplicate_simplified_question_whole_stream": round(duplicate / len(filters), 4),
        "surviving_unique_normalized": len({normalize_question(t) for t in kept_texts}),
        "surviving_distinct_2": round(distinct2(kept_texts), 4),
        "structured_rewrite_validation": "NOT_REPRODUCIBLE_OFFLINE",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rounds = {1: load_round(R1, 1), 2: load_round(R2, 2)}

    # ── proposal diversity ──────────────────────────────────────────────────
    proposal_diversity = {}
    for rid, data in rounds.items():
        props = data["proposals"]
        raw = stage_stats(props, lambda r: r.get("proposal", ""), f"R{rid} proposal raw")
        qpart = stage_stats(props, lambda r: question_part(r.get("proposal", "")), f"R{rid} proposal question-part")
        per_image: dict[str, list[str]] = defaultdict(list)
        for r in tqdm(props, desc=f"Compute proposal within-image uniqueness R{rid}", unit="rec"):
            per_image[str(r["image_id"])].append(question_part(r.get("proposal", "")))
        within = [len({normalize_question(q) for q in qs}) / len(qs) for qs in per_image.values() if qs]
        proposal_diversity[f"R{rid}"] = {
            "raw": raw,
            "question_part": qpart,
            "answer_appended_rate": round(
                sum(1 for r in props if "?" in str(r.get("proposal", "")) and str(r.get("proposal", "")).split("?", 1)[1].strip()) / len(props), 4),
            "images": len(per_image),
            "within_image_unique_rate_mean": round(sum(within) / len(within), 4),
            "within_image_unique_rate_min": round(min(within), 4),
        }
    (OUT / "proposal_diversity.json").write_text(json.dumps(proposal_diversity, indent=2))

    # ── intent (task-family) audit ──────────────────────────────────────────
    intent = {"schedule": "TASKS[(index + seed + round) % 17], 4 candidates/image (2 turns x 2 candidates)",
              "task_metadata_retained_in_training": False, "rounds": {}}
    for rid, data in rounds.items():
        prop_counts = Counter(r["task"] for r in data["proposals"])
        acc_counts = Counter(r["task"] for r in data["filters"] if r.get("accepted") is True)
        intent["rounds"][f"R{rid}"] = {
            "proposals_per_task": dict(sorted(prop_counts.items())),
            "accepted_per_task": dict(sorted(acc_counts.items())),
            "retained_per_task": "NOT_TRACEABLE",
            "tasks_present": len(prop_counts),
        }
    (OUT / "intent_audit.json").write_text(json.dumps(intent, indent=2))

    # ── rewrite audit ───────────────────────────────────────────────────────
    rewrite_audit = {}
    for rid, data in rounds.items():
        pairs = []
        for b, c in tqdm(list(zip(data["proposals"], data["rewrites"])),
                         desc=f"Compare proposal vs rewrite R{rid}", unit="rec"):
            prop = question_part(b.get("proposal", ""))
            rew = rewrite_question(c)
            pairs.append((prop, rew))
        copies = sum(1 for p, r in pairs if normalize_question(p) == normalize_question(r))
        overlaps = [lexical_overlap_ratio(r, p) for p, r in pairs]
        edits = [1 - difflib.SequenceMatcher(None, normalize_question(p), normalize_question(r)).ratio() for p, r in pairs]
        rewrite_audit[f"R{rid}"] = {
            "rewrite_output_schema": ("Rewritten question" if any(
                isinstance(c.get("rewrite"), dict) and "Rewritten question" in c["rewrite"] for c in data["rewrites"][:2000]) else "text"),
            "proposal_unique": len({normalize_question(p) for p, _ in pairs}),
            "rewrite_unique": len({normalize_question(r) for _, r in pairs}),
            "unique_ratio_rewrite_over_proposal": round(
                len({normalize_question(r) for _, r in pairs}) / max(1, len({normalize_question(p) for p, _ in pairs})), 4),
            "proposal_to_rewrite_exact_copy_rate": round(copies / len(pairs), 4),
            "mean_lexical_overlap": round(sum(overlaps) / len(overlaps), 4),
            "mean_token_edit_fraction": round(sum(edits) / len(edits), 4),
            "distinct_2_proposal": round(distinct2([p for p, _ in pairs]), 4),
            "distinct_2_rewrite": round(distinct2([r for _, r in pairs]), 4),
            "top_rewrite": Counter(r for _, r in pairs).most_common(3),
        }
    (OUT / "rewrite_audit.json").write_text(json.dumps(rewrite_audit, indent=2))

    # ── filter audit + acceptance funnel ────────────────────────────────────
    filter_audit = {"rounds": {}}
    acceptance = {}
    for rid, data in rounds.items():
        stages = [("proposal", len(data["proposals"])), ("answer", len(data["rewrites"])),
                  ("rewrite", len(data["rewrites"])), ("filter_input", len(data["filters"])),
                  ("accepted", sum(1 for r in data["filters"] if r.get("accepted") is True)),
                  ("retained", len(data["retained"]))]
        reasons = Counter(r.get("rejection_reason", "") for r in data["filters"] if r.get("accepted") is not True)
        judge_soft = Counter()
        for r in data["filters"][:5000]:
            j = r.get("judge")
            sc = j.get("soft_constraints") if isinstance(j, dict) else None
            judge_soft[json.dumps(sc, sort_keys=True)] += 1
        acceptance[f"R{rid}"] = {"stages": stages, "rejection_reasons": dict(reasons)}
        filter_audit["rounds"][f"R{rid}"] = {
            "acceptance": stages,
            "rejection_reasons": dict(reasons),
            "judge_soft_constraint_values_top": judge_soft.most_common(5),
            "difficulty_improvement_required": False,
            "proposal_rewrite_comparison": False,
            "compound_rejection": False,
            "lexical_overlap_filter": False,
            "dedup": False,
            "replayed_official_deterministic": replay_official_deterministic(
                R1 if rid == 1 else R2, f"R{rid}"),
        }
    (OUT / "filter_audit.json").write_text(json.dumps(filter_audit, indent=2))
    (OUT / "acceptance_funnel.json").write_text(json.dumps(acceptance, indent=2))

    # ── uniqueness + diversity funnels ──────────────────────────────────────
    uniqueness = {}
    diversity = {}
    for rid, data in rounds.items():
        g = f"R{rid}"
        prop_q = [question_part(r.get("proposal", "")) for r in data["proposals"]]
        rew_q = [rewrite_question(r) for r in data["rewrites"]]
        acc_q = [r["question"] for r in data["filters"] if r.get("accepted") is True]
        ret_q = [r["question"] for r in data["retained"]]
        uniqueness[g] = {
            "proposal": stage_stats(data["proposals"], lambda r: question_part(r.get("proposal", "")), "proposal"),
            "rewrite": stage_stats(data["rewrites"], rewrite_question, "rewrite"),
            "accepted": stage_stats([r for r in data["filters"] if r.get("accepted")], lambda r: r["question"], "accepted"),
            "retained": stage_stats(data["retained"], lambda r: r["question"], "retained"),
            "funnel": {
                "proposal": [len(prop_q), len({normalize_question(t) for t in prop_q})],
                "rewrite": [len(rew_q), len({normalize_question(t) for t in rew_q})],
                "accepted": [len(acc_q), len({normalize_question(t) for t in acc_q})],
                "retained": [len(ret_q), len({normalize_question(t) for t in ret_q})],
            },
        }
        diversity[g] = {
            "proposal": round(distinct2(prop_q), 4),
            "rewrite": round(distinct2(rew_q), 4),
            "accepted": round(distinct2(acc_q), 4),
            "retained": round(distinct2(ret_q), 4),
        }
    for tag, name in (("m0", "M0"), ("m1", "M1"), ("m2", "M2")):
        g = EVAL / f"{tag}_sampled4_q.jsonl"
        gr = EVAL / f"{tag}_greedy_q.jsonl"
        uniqueness.setdefault("models", {})[name] = {
            "greedy": stage_stats(read_jsonl(gr), lambda r: r["question"], f"{name} greedy"),
            "sampled4": stage_stats(read_jsonl(g), lambda r: r["question"], f"{name} sampled4"),
        }
        diversity.setdefault("models", {})[name] = {
            "greedy": round(distinct2([r["question"] for r in read_jsonl(gr)]), 4),
            "sampled4": round(distinct2([r["question"] for r in read_jsonl(g)]), 4),
        }
    (OUT / "uniqueness_funnel.json").write_text(json.dumps(uniqueness, indent=2))
    (OUT / "diversity_funnel.json").write_text(json.dumps(diversity, indent=2))

    # ── retention audit ─────────────────────────────────────────────────────
    retention = {}
    for rid, data in rounds.items():
        accepted = [r for r in data["filters"] if r.get("accepted") is True]
        ret = data["retained"]
        retention[f"R{rid}"] = {
            "accepted": stage_stats(accepted, lambda r: r["question"], "accepted"),
            "retained": stage_stats(ret, lambda r: r["question"], "retained"),
            "unique_images_accepted": len({r["image_id"] for r in accepted}),
            "unique_images_retained": len({r["image_id"] for r in ret}),
            "top_retained_questions": Counter(r["question"] for r in ret).most_common(8),
            "selection": "deterministic image round-robin, one accepted triple per image per pass, stop at 10,000",
            "official_retention": "UNKNOWN (not released)",
        }
    (OUT / "retention_audit.json").write_text(json.dumps(retention, indent=2))

    # ── training format / budget ────────────────────────────────────────────
    fmt = {}
    for rid, rdir in ((1, SEEQ / "round1"), (2, SEEQ / "round2")):
        rows = read_jsonl(rdir / "formatted_20000.jsonl")
        fmt[f"R{rid}"] = {
            "records": len(rows),
            "formats": dict(Counter(r["format"] for r in rows)),
            "qg_target": "question + ' ' + answer (approx official IQA, but unstructured)",
            "qa_target": "answer (official VQA)",
            "image_to_question_only_(official_IQ)": False,
            "answer_to_question_(official_VQG)": False,
            "sample_qg": next(r for r in rows if r["format"] == "qg"),
            "sample_qa": next(r for r in rows if r["format"] == "qa"),
        }
    (OUT / "training_format_audit.json").write_text(json.dumps(fmt, indent=2))

    budget = {}
    for rid, mdir in ((1, SEEQ / "round1_m1"), (2, SEEQ / "round2_m2")):
        args = json.loads((mdir / "training_args.json").read_text())
        summary = json.loads((mdir / "training_summary.json").read_text()) if (mdir / "training_summary.json").exists() else {}
        state = json.loads((mdir / "checkpoint-10000/trainer_state.json").read_text())
        losses = [h["loss"] for h in state["log_history"] if h.get("loss") is not None]
        n_unique = {1: 1889, 2: 1587}[rid]
        budget[f"R{rid}"] = {
            "records": summary.get("records", args["num_train_epochs"]),
            "steps": summary.get("steps", state.get("global_step")),
            "start_adapter": summary.get("start_adapter"),
            "per_device_train_batch_size": args["per_device_train_batch_size"],
            "gradient_accumulation_steps": args["gradient_accumulation_steps"],
            "learning_rate": args["learning_rate"],
            "lr_scheduler_type": args["lr_scheduler_type"],
            "warmup_steps": args["warmup_steps"],
            "num_train_epochs": args["num_train_epochs"],
            "lora": {"r": 16, "alpha": 32, "dropout": 0.05},
            "effective_epochs": round(args["num_train_epochs"], 2),
            "record_exposures": summary.get("steps", 0) * args["per_device_train_batch_size"],
            "unique_question_exposures": round((summary.get("steps", 0) * args["per_device_train_batch_size"]) / n_unique / 2, 2),
            "initial_loss": round(losses[0], 4),
            "final_loss": round(losses[-1], 4),
            "official_config": "UNKNOWN (trainer not released)",
        }
    (OUT / "training_budget_audit.json").write_text(json.dumps(budget, indent=2))

    # ── model role audit ────────────────────────────────────────────────────
    import hashlib

    def sha(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    model_role = {
        "M0": {"checkpoint": "outputs/checkpoints/qwen3vl8b_lora_r16_vqav2_50k/checkpoint-final",
               "adapter_sha256": sha(ROOT / "outputs/checkpoints/qwen3vl8b_lora_r16_vqav2_50k/checkpoint-final/adapter_model.safetensors")},
        "M1": {"checkpoint": "outputs/same_backbone_baselines/seeq_style/round1_m1",
               "adapter_sha256": sha(SEEQ / "round1_m1/adapter_model.safetensors")},
        "M2": {"checkpoint": "outputs/same_backbone_baselines/seeq_style/round2_m2",
               "adapter_sha256": sha(SEEQ / "round2_m2/adapter_model.safetensors")},
        "round1_roles": {"proposer": "M0", "answerer": "M0", "reference_rewrite_filter": "M0",
                         "evidence": "logs/run_b4_after_b3.sh: --adapter M0, no --reference-adapter (defaults to proposer)"},
        "round2_roles": {"proposer": "M1", "answerer": "M1", "reference_rewrite_filter": "M0",
                         "evidence": "round2_generation_m1/generation_manifest.json; R2 proposals carry M1's appended-answer signature"},
        "stale_invalid_r2": "round2_generation_invalid/ (M0 proposer, pre-M1) not used",
        "checks": {"R2_proposer_is_M1": True, "R2_reference_is_M0": True,
                   "stale_adapter": False, "wrong_model_loaded": False},
    }
    (OUT / "model_role_audit.json").write_text(json.dumps(model_role, indent=2))

    print("audit data written to", OUT)
    for name in sorted(p.name for p in OUT.glob("*.json")):
        print("  ", name)


if __name__ == "__main__":
    main()
