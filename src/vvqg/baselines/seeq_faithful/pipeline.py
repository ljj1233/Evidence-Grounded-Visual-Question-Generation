"""Faithful SeeQ multi-intent proposal / rewrite / filtering pipeline.

Core mechanisms (PAPER 3.1-3.4, OFFICIAL_CODE generation_pipeline.py):
  * multi-intent proposal with the current model M_t,
  * answer generation by M_t,
  * frozen-M0 rewrite with an evolution instruction,
  * rewritten-answer generation by M0,
  * model-based visual-validity + perception/reasoning-improvement filter,
  * image round-robin retention and dual-format QG+QA construction.

Stages are resumable by stable candidate key (image_id, round, intent,
sample_index).  The generation backend only needs ``generate_role``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import re

from tqdm.auto import tqdm

from ..common import SEED, assert_exact_count, read_jsonl
from ..seeq_style.pipeline import _json_output, _retain_round_robin
from . import prompts as P
from .filtering import decide, is_compound_question, lexical_overlap_ratio, perception_score, reasoning_score


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().casefold())


@dataclass(frozen=True)
class FaithfulSeeQConfig:
    retained_per_round: int = 10_000
    samples_per_intent: int = 2
    intents: tuple[str, ...] = P.INTENT_NAMES
    seed: int = SEED
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    max_new_tokens_proposal: int = 512
    max_new_tokens_answer: int = 256
    max_new_tokens_rewrite: int = 512
    max_new_tokens_judge: int = 512


def candidate_key(image_id: str, round_id: int, intent: str, sample_index: int) -> str:
    return f"r{round_id}-{image_id}-{intent}-{sample_index}"


def resolve_round_models(round_id: int, m0_adapter: str, m1_adapter: str | None = None) -> tuple[str, str]:
    """PAPER 3.1/3.2/3.4.2: R1 proposer=M0; R2 proposer=M1; rewrite/filter=M0 always."""
    if round_id == 1:
        return m0_adapter, m0_adapter
    if round_id == 2:
        if not m1_adapter:
            raise ValueError("round 2 requires the M1 proposer adapter")
        return m1_adapter, m0_adapter
    raise ValueError(f"unsupported round: {round_id}")


def schedule(images: Sequence[dict], config: FaithfulSeeQConfig, round_id: int) -> list[dict]:
    rows: list[dict] = []
    for image in images:
        image_id = str(image["image_id"])
        image_path = image.get("image") or image.get("image_path")
        for intent in config.intents:
            for sample_index in range(config.samples_per_intent):
                rows.append({
                    "candidate_id": candidate_key(image_id, round_id, intent, sample_index),
                    "image_id": image_id,
                    "image": image_path,
                    "round": round_id,
                    "intent": intent,
                    "sample_index": sample_index,
                })
    return rows


def proposal_prompt(intent: str) -> str:
    return f"Visual intent: {P.INTENTS[intent]}\n" + P.BASE_PROPOSAL_PROMPT


def _question_from_proposal(raw: Any) -> str | None:
    out = _json_output(raw)
    if isinstance(out, dict):
        q = out.get("Question") or out.get("question")
        return str(q).strip() if q else None
    return None


def _rewritten_question(raw: Any) -> str | None:
    out = _json_output(raw)
    if isinstance(out, dict):
        q = out.get("Rewritten question") or out.get("Question") or out.get("question")
        return str(q).strip() if q else None
    return None


def _append(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


class FaithfulSeeQPipeline:
    def __init__(self, backend, config: FaithfulSeeQConfig = FaithfulSeeQConfig(), *,
                 proposer_checkpoint: str = "", reference_checkpoint: str = ""):
        self.backend = backend
        self.config = config
        self.proposer_checkpoint = proposer_checkpoint
        self.reference_checkpoint = reference_checkpoint

    # -- generation helpers -------------------------------------------------
    @property
    def batch_size(self) -> int:
        return max(1, int(getattr(self.backend, "batch_size", 64)))

    def _gen(self, prompts, images, *, reference=False, do_sample=False,
             max_new_tokens=256, temperature=None, top_p=None, top_k=None):
        c = self.config
        return self.backend.generate_role(
            list(prompts), images, reference=reference, do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            temperature=(c.temperature if temperature is None else temperature),
            top_p=(c.top_p if top_p is None else top_p),
            top_k=(c.top_k if top_k is None else top_k),
        )

    def score_questions(self, questions: Sequence[str], images: Sequence[Any]) -> list[dict]:
        """OFFICIAL_CODE self_judge: validity + perception + reasoning per question."""
        n = len(questions)
        entities = self._gen([P.TEXT_ENTITY_PROMPT + f"Question: {q}" for q in questions],
                             None, max_new_tokens=256)
        # Greedy judge generations are independent per sequence, so the three
        # image-conditioned prompt families are dispatched as one vLLM call
        # (fewer round-trips, better batching, one image-preprocess pass).
        prompts = (
            [P.SOFT_CONSTRAINTS_EXTRACTED_PROMPT.replace("[INSERT_SHARED_ENTITIES]", str(e or "")) for e in entities]
            + [P.REASONING_STEPS_PROMPT + f"\nQuestion: {q}" for q in questions]
            + [P.HARD_CONSTRAINTS_PROMPT + f"\nQuestion: {q}" for q in questions]
        )
        raw = self._gen(prompts, list(images) * 3, max_new_tokens=512)
        perception_raw, reasoning_raw, hard_raw = raw[:n], raw[n:2 * n], raw[2 * n:]
        assert len(perception_raw) == len(reasoning_raw) == len(hard_raw) == n
        return [{
            "perception": perception_score(_json_output(perception_raw[i])),
            "reasoning": reasoning_score(_json_output(reasoning_raw[i])),
            "hard": _json_output(hard_raw[i]),
        } for i in range(n)]

    # -- pipeline -----------------------------------------------------------
    def run(self, images: Sequence[dict], out_dir: Path, *, round_id: int = 1, limit: int | None = None) -> dict:
        out_dir = Path(out_dir)
        rows = schedule(list(images), self.config, round_id)
        if limit is not None:
            rows = rows[:limit]
        dirs = {name: out_dir / name for name in ("proposals", "answers", "rewrites", "filters", "accepted", "retained")}
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        paths = {
            "proposals": dirs["proposals"] / "proposals.jsonl",
            "answers": dirs["answers"] / "answers.jsonl",
            "rewrites": dirs["rewrites"] / "rewrites.jsonl",
            "filters": dirs["filters"] / "filters.jsonl",
        }
        by_id = {r["candidate_id"]: r for r in rows}
        self._round = round_id

        self._propose(rows, paths["proposals"])
        proposals = self._load(paths["proposals"])
        self._answer([proposals[r["candidate_id"]] for r in rows if r["candidate_id"] in proposals], paths["answers"])
        answers = self._load(paths["answers"])
        self._rewrite([answers[r["candidate_id"]] for r in rows if r["candidate_id"] in answers], paths["rewrites"])
        rewrites = self._load(paths["rewrites"])
        self._filter([rewrites[r["candidate_id"]] for r in rows if r["candidate_id"] in rewrites], paths["filters"])
        filtered = list(self._load(paths["filters"]).values())

        accepted = [r for r in filtered if r.get("accepted") is True]
        retained = _retain_round_robin(accepted, self.config.retained_per_round)
        formatted_dir = dirs["retained"]
        from .training import dual_format_records
        from ..common import write_jsonl
        triples = [{"image_id": r["image_id"], "image": r["image"], "question": r["rewritten_question"],
                    "answer": r["rewritten_answer"], "intent": r["intent"], "candidate_id": r["candidate_id"]}
                   for r in retained]
        write_jsonl(dirs["accepted"] / "accepted.jsonl", accepted)
        write_jsonl(formatted_dir / "retained.jsonl", triples)
        if triples:
            write_jsonl(formatted_dir / "formatted.jsonl", dual_format_records(triples))

        stats = {
            "round": round_id,
            "scheduled": len(rows),
            "proposals": len(proposals),
            "answers": len(answers),
            "rewrites": len(rewrites),
            "filtered": len(filtered),
            "accepted": len(accepted),
            "retained": len(triples),
            "acceptance_rate": (len(accepted) / len(filtered)) if filtered else 0.0,
            "images_visited": len({r["image_id"] for r in proposals.values()}),
            "images_retained": len({r["image_id"] for r in triples}),
            "proposer_checkpoint": self.proposer_checkpoint,
            "reference_checkpoint": self.reference_checkpoint,
            "config": self.config.__dict__,
            "seeq_commit": P.SEEQ_COMMIT,
        }
        (out_dir / "funnel.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
        (out_dir / "manifest.json").write_text(json.dumps({
            "round": round_id, "proposer_checkpoint": self.proposer_checkpoint,
            "reference_checkpoint": self.reference_checkpoint,
            "official_seeq_commit": P.SEEQ_COMMIT, "seed": self.config.seed,
        }, indent=2, sort_keys=True) + "\n")
        return stats

    @staticmethod
    def _load(path: Path) -> dict[str, dict]:
        if not path.exists():
            return {}
        return {row["candidate_id"]: row for row in read_jsonl(path)}

    def _bar(self, pending, stage: str):
        return tqdm(total=len(pending), desc=f"R{self._round} {stage}", unit="cand", dynamic_ncols=True)

    def _propose(self, rows: list[dict], path: Path) -> None:
        done = self._load(path)
        pending = [r for r in rows if r["candidate_id"] not in done]
        step = self.batch_size
        bar = self._bar(pending, "proposal")
        for start in range(0, len(pending), step):
            chunk = pending[start:start + step]
            raws = self._gen([proposal_prompt(r["intent"]) for r in chunk],
                             [r["image"] for r in chunk], do_sample=True,
                             max_new_tokens=self.config.max_new_tokens_proposal)
            out = []
            for r, raw in zip(chunk, raws):
                q = _question_from_proposal(raw)
                if not q:
                    continue
                out.append({**r, "proposer_checkpoint": self.proposer_checkpoint,
                            "question": q, "proposal_raw": str(raw)[:2000]})
            _append(path, out)
            bar.update(len(chunk))
        bar.close()

    def _answer(self, rows: list[dict], path: Path) -> None:
        done = self._load(path)
        pending = [r for r in rows if r["candidate_id"] not in done]
        step = self.batch_size
        bar = self._bar(pending, "answer")
        for start in range(0, len(pending), step):
            chunk = pending[start:start + step]
            answers = self._gen([P.ANSWER_PROMPT + " Question: " + r["question"] for r in chunk],
                                [r["image"] for r in chunk], do_sample=True,
                                max_new_tokens=self.config.max_new_tokens_answer)
            _append(path, [{**r, "answer": str(a).strip()} for r, a in zip(chunk, answers)])
            bar.update(len(chunk))
        bar.close()

    def _score_cached(self, questions, images, cache: dict) -> list[dict]:
        keys = [(normalize_question(q), str(img)) for q, img in zip(questions, images)]
        unique: dict[tuple, int] = {}
        for i, key in enumerate(keys):
            if key not in cache and key not in unique:
                unique[key] = i
        if unique:
            idx = list(unique.values())
            scored = self.score_questions([questions[i] for i in idx], [images[i] for i in idx])
            for key, value in zip(unique, scored):
                cache[key] = value
        return [cache[key] for key in keys]

    @staticmethod
    def _reject(row: dict, reason: str) -> dict:
        return {**row, "valid_visual": None, "perception_improved": None, "reasoning_improved": None,
                "improved": False, "perception_delta": None, "reasoning_delta": None, "seed_overlap": None,
                "seed_perception": None, "seed_reasoning": None, "rew_perception": None, "rew_reasoning": None,
                "accepted": False, "rejection_reason": reason}

    def _rewrite(self, rows: list[dict], path: Path) -> None:
        done = self._load(path)
        pending = [r for r in rows if r["candidate_id"] not in done]
        step = self.batch_size
        bar = self._bar(pending, "rewrite")
        for start in range(0, len(pending), step):
            chunk = pending[start:start + step]
            rewrite_prompts = [P.INTENT_REWRITE[r["intent"]] + f"\nQuestion: {r['question']}\t Answer: {r['answer']}"
                               for r in chunk]
            raws = self._gen(rewrite_prompts, [r["image"] for r in chunk], reference=True,
                             do_sample=True, max_new_tokens=self.config.max_new_tokens_rewrite)
            rew_qs = [_rewritten_question(raw) for raw in raws]
            keep = [(r, raw, q) for r, raw, q in zip(chunk, raws, rew_qs) if q]
            answers = self._gen([P.ANSWER_PROMPT + " Question: " + q for _, _, q in keep],
                                [r["image"] for r, _, _ in keep], reference=True,
                                do_sample=True, max_new_tokens=self.config.max_new_tokens_answer)
            out = [{**r, "rewritten_question": q, "rewritten_answer": str(a).strip(),
                    "rewrite_raw": str(raw)[:2000], "reference_checkpoint": self.reference_checkpoint}
                   for (r, raw, q), a in zip(keep, answers)]
            _append(path, out)
            bar.update(len(chunk))
        bar.close()

    def _filter(self, rows: list[dict], path: Path) -> None:
        done = self._load(path)
        pending = [r for r in rows if r["candidate_id"] not in done]
        cache: dict = {}
        seen = {normalize_question(r["rewritten_question"]) for r in done.values() if r.get("accepted")}
        step = self.batch_size
        bar = self._bar(pending, "filter")
        for start in range(0, len(pending), step):
            chunk = pending[start:start + step]
            survivors, out = [], []
            for r in chunk:
                q = r["rewritten_question"]
                if is_compound_question(q):
                    out.append(self._reject(r, "multi_question"))
                elif lexical_overlap_ratio(q, r["question"]) >= 0.82:
                    out.append(self._reject(r, "lexical_overlap_too_high"))
                elif normalize_question(q) in seen:
                    out.append(self._reject(r, "duplicate_simplified_question"))
                else:
                    survivors.append(r)
            if survivors:
                n = len(survivors)
                scores = self._score_cached(
                    [r["question"] for r in survivors] + [r["rewritten_question"] for r in survivors],
                    [r["image"] for r in survivors] * 2, cache)
                seed_scores, rew_scores = scores[:n], scores[n:]
                for r, ss, rs in zip(survivors, seed_scores, rew_scores):
                    d = decide(hard_judge=rs["hard"], perception_seed=ss["perception"],
                               perception_rew=rs["perception"], reasoning_seed=ss["reasoning"],
                               reasoning_rew=rs["reasoning"], seed_question=r["question"],
                               rewritten_question=r["rewritten_question"], seen_questions=seen)
                    if d["accepted"]:
                        seen.add(normalize_question(r["rewritten_question"]))
                    out.append({**r, "seed_perception": ss["perception"], "seed_reasoning": ss["reasoning"],
                                "rew_perception": rs["perception"], "rew_reasoning": rs["reasoning"],
                                "rew_hard": rs["hard"], **d})
            _append(path, out)
            bar.update(len(chunk))
        bar.close()


__all__ = ["FaithfulSeeQConfig", "FaithfulSeeQPipeline", "schedule", "proposal_prompt", "candidate_key"]
