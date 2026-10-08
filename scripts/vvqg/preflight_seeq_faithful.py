#!/usr/bin/env python3
"""SeeQ-faithful proposal pre-flight: multi-intent diversity gate.

Generates proposals for a deterministic TRAIN image subset with the frozen M0
proposer, measures lexical and semantic diversity, compares against the previous
collapsed SeeQ-style proposal stage (from stored artifacts), and writes a gate
decision.  No answer/rewrite/filter/training is run here.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vvqg.paths import local_model_path

import numpy as np  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from vvqg.baselines.seeq_faithful.pipeline import (  # noqa: E402
    FaithfulSeeQConfig, _question_from_proposal, proposal_prompt, schedule,
)
from vvqg.baselines.seeq_style.pipeline import _json_output  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def norm(q: str) -> str:
    return re.sub(r"\s+", " ", str(q).strip().casefold())


def tokenize(q: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", str(q).casefold())


def uniq_rate(questions) -> float:
    qs = [norm(q) for q in questions]
    return len(set(qs)) / len(qs) if qs else 0.0


def distinct_2(questions) -> float:
    bigrams, total = set(), 0
    for q in questions:
        toks = tokenize(q)
        for a, b in zip(toks, toks[1:]):
            bigrams.add((a, b)); total += 1
    return len(bigrams) / total if total else 0.0


def per_image_questions(rows: list[dict], key: str = "proposed_question") -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        groups[str(row["image_id"])].append(row[key])
    return groups


def lexical_metrics(rows: list[dict], key: str) -> dict:
    questions = [r[key] for r in rows]
    groups = per_image_questions(rows, key)
    within = [len({norm(q) for q in qs}) for qs in groups.values()]
    lengths = [len(tokenize(q)) for q in questions]
    return {
        "n": len(questions),
        "images": len(groups),
        "normalized_unique_rate": round(uniq_rate(questions), 4),
        "distinct_2": round(distinct_2(questions), 4),
        "normalized_duplicate_rate": round(1.0 - uniq_rate(questions), 4),
        "within_image_unique_mean": round(float(np.mean(within)), 3) if within else 0.0,
        "within_image_unique_rate": round(float(np.mean([u / len(groups[i]) for i, u in zip(groups, within)])), 4) if within else 0.0,
        "mean_length": round(float(np.mean(lengths)), 2) if lengths else 0.0,
    }


def semantic_diversity(groups: dict[str, list[str]], embed_fn) -> float:
    texts = [q for g in groups.values() for q in g]
    if not texts:
        return 0.0
    embeddings, _ = embed_fn(texts)
    distances = []
    offset = 0
    for g in groups.values():
        block = embeddings[offset:offset + len(g)]; offset += len(g)
        sims = block @ block.T
        n = len(g)
        if n > 1:
            distances.append(float((1.0 - sims[np.triu_indices(n, 1)]).mean()))
    return round(float(np.mean(distances)), 4) if distances else 0.0


def load_embedder():
    import hashlib

    import torch
    from transformers import AutoModel, AutoTokenizer

    model_path = Path(local_model_path('Qwen3-Embedding-4B')).expanduser()
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto").eval()

    def embed(texts: list[str]) -> tuple[np.ndarray, list[str]]:
        out = np.empty((len(texts), model.config.hidden_size), dtype=np.float32)
        with torch.inference_mode():
            for start in tqdm(range(0, len(texts), 64), desc="SeeQ proposal SemDiv", unit="batch"):
                batch = tokenizer(texts[start:start + 64], padding=True, truncation=True, max_length=512, return_tensors="pt").to(model.device)
                lengths = batch["attention_mask"].sum(dim=1) - 1
                last = model(**batch).last_hidden_state[torch.arange(batch["input_ids"].shape[0], device=model.device), lengths]
                out[start:start + len(texts[start:start + 64])] = torch.nn.functional.normalize(last, p=2, dim=1).float().cpu().numpy()
        return out, texts

    return embed


def generate_proposals(images, out_path: Path, args) -> list[dict]:
    from vvqg.baselines.seeq_style.vllm_backend import VLLMQwen3VLBackend
    config = FaithfulSeeQConfig(samples_per_intent=args.samples_per_intent)
    rows = schedule(images, config, round_id=1)
    done = {r["candidate_id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    pending = [r for r in rows if r["candidate_id"] not in done]
    print(f"SeeQ preflight: scheduled={len(rows)} done={len(done)} pending={len(pending)}", flush=True)
    if not pending:
        return read_jsonl(out_path)
    backend = VLLMQwen3VLBackend(
        base_model=args.base_model, adapter=args.adapter,
        batch_size=args.batch_size, gpu_memory_utilization=args.gpu_memory_utilization,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    bar = tqdm(total=len(pending), desc="SeeQ proposal preflight", unit="prop", dynamic_ncols=True)
    with out_path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), args.batch_size):
            chunk = pending[start:start + args.batch_size]
            raws = backend.generate_role(
                [proposal_prompt(r["intent"]) for r in chunk], [r["image"] for r in chunk],
                reference=False, do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
                max_new_tokens=args.max_new_tokens,
            )
            written = 0
            for r, raw in zip(chunk, raws):
                q = _question_from_proposal(raw)
                if not q:
                    continue
                handle.write(json.dumps({**r, "proposed_question": q, "raw": str(raw)[:2000]}, ensure_ascii=False, sort_keys=True) + "\n")
                written += 1
            handle.flush()
            bar.update(len(chunk))
    bar.close()
    elapsed = time.perf_counter() - started
    rows_out = read_jsonl(out_path)
    print(json.dumps({"generated": len(rows_out), "seconds": round(elapsed, 1), "proposals_per_sec": round((len(rows_out) - len(done)) / elapsed, 2)}), flush=True)
    return rows_out


def old_proposals() -> dict:
    base = ROOT / "outputs/same_backbone_baselines/seeq_style"
    data = {"old_R1": base / "round1_generation/stage_b_answers.jsonl",
            "old_R2": base / "round2_generation_m1/stage_b_answers.jsonl"}
    out = {}
    for name, path in data.items():
        rows = read_jsonl(path)
        out[name] = [{"image_id": r["image_id"], "proposed_question": r["proposal"]} for r in rows]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=ROOT / "outputs/same_backbone_baselines/manifests/train_unlabeled_images.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/same_backbone_baselines/seeq_faithful/preflight")
    parser.add_argument("--base-model", default=str(Path(local_model_path('Qwen3-VL-8B-Instruct')).expanduser()))
    parser.add_argument("--adapter", default=str(ROOT / "outputs/checkpoints/qwen3vl8b_lora_r16_vqav2_50k/checkpoint-final"))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--samples-per-intent", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--skip-embedding", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    images = read_jsonl(args.images)[: args.limit]
    assert len(images) == args.limit, f"expected {args.limit} train images, got {len(images)}"

    prediction_path = args.out / "proposal_predictions.jsonl"
    rows = generate_proposals(images, prediction_path, args)
    new_groups = per_image_questions(rows, "proposed_question")
    new_lex = lexical_metrics(rows, "proposed_question")
    intent_counts = Counter(r["intent"] for r in rows)
    intent_lex = {intent: lexical_metrics([r for r in rows if r["intent"] == intent], "proposed_question")
                  for intent in intent_counts}

    old = old_proposals()
    old_lex = {name: lexical_metrics(rs, "proposed_question") for name, rs in old.items()}

    embed = None if args.skip_embedding else load_embedder()
    if embed is not None:
        new_lex["semantic_diversity"] = semantic_diversity(new_groups, embed)
        for name, rs in old.items():
            old_lex[name]["semantic_diversity"] = semantic_diversity(per_image_questions(rs, "proposed_question"), embed)
    else:
        new_lex["semantic_diversity"] = None

    old_r1 = old_lex["old_R1"]
    recovered = (
        new_lex["normalized_unique_rate"] >= max(0.25, 2.0 * old_r1["normalized_unique_rate"])
        and new_lex["within_image_unique_rate"] >= 0.5
        and sum(v["normalized_unique_rate"] >= 0.4 for v in intent_lex.values()) >= 3
    )
    collapsed = (
        new_lex["within_image_unique_rate"] < 0.5
        or new_lex["normalized_unique_rate"] <= old_r1["normalized_unique_rate"]
    )
    decision = "PROPOSAL_DIVERSITY_RECOVERED" if recovered else ("PROPOSAL_STILL_COLLAPSED" if collapsed else "PROPOSAL_PARTIAL_STOP")

    payload = {
        "schema": "seeq-faithful-proposal-diversity-v1",
        "images": args.limit,
        "proposals_per_image": 10,
        "intent_counts": dict(intent_counts),
        "intent_lexical": intent_lex,
        "faithful": new_lex,
        "old": old_lex,
        "comparison": {
            "normalized_unique_rate": {"old_R1": old_lex["old_R1"]["normalized_unique_rate"], "old_R2": old_lex["old_R2"]["normalized_unique_rate"], "faithful": new_lex["normalized_unique_rate"]},
            "distinct_2": {"old_R1": old_lex["old_R1"]["distinct_2"], "old_R2": old_lex["old_R2"]["distinct_2"], "faithful": new_lex["distinct_2"]},
            "within_image_unique_rate": {"old_R1": old_lex["old_R1"]["within_image_unique_rate"], "old_R2": old_lex["old_R2"]["within_image_unique_rate"], "faithful": new_lex["within_image_unique_rate"]},
            "semantic_diversity": {"old_R1": old_lex["old_R1"].get("semantic_diversity"), "old_R2": old_lex["old_R2"].get("semantic_diversity"), "faithful": new_lex.get("semantic_diversity")},
        },
    }
    (args.out / "proposal_diversity.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (args.out / "decision.json").write_text(json.dumps({
        "schema": "seeq-faithful-preflight-decision-v1",
        "decision": decision,
        "criteria": {
            "recovered": "unique_rate >= max(0.25, 2x old_R1) AND within_image_unique_rate >= 0.5 AND >=3 intents with unique_rate>=0.4",
            "collapsed": "within_image_unique_rate < 0.5 OR unique_rate <= old_R1",
        },
        "evidence": payload["comparison"],
        "continue_to_round1": decision == "PROPOSAL_DIVERSITY_RECOVERED",
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": decision, "comparison": payload["comparison"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
