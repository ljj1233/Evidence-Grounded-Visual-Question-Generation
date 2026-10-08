"""Canonical SeeQ proposal/evolution/filter pipeline and local backend."""
from __future__ import annotations
import ast, json, re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
from ..common import SEED, assert_exact_count, read_jsonl, write_jsonl

SEEQ_COMMIT = "18e795293808c5cf9f69f1e980577ab40b3faa95"
SIMPLIFY_PROMPT = """You are a careful question normalizer.
Rewrite the question into a shorter, cleaner form while preserving the same reasoning structure.
Rules:
- Keep it as ONE syntactic question ending with '?'.
- Preserve all compared entities, visible evidence anchors, spatial relations, and counterfactual conditions.
- Preserve whether the question is asking for comparison, causal explanation, multi-hop inference, or conditional reasoning.
- Remove only incidental wording, not the core reasoning requirements.
- Do NOT turn the question into a generic template like "What happens", "What effect", "What impact", "Which is more important", "How does X affect Y", "How does X impact Y", "How does X influence Y".
- Do NOT broaden the question into mood/usefulness/aesthetic/comfort commentary.
Only output the rewritten question. Do not add any explanation or formatting.
"""
ANSWER_PROMPT = "You are a helpful assistant that helps generating the answer of the question based on the given image. Directly output the final answer. "
PROPOSAL_PROMPT = "You will be given an image. Based on the image, generate a question that focus on the objects, actions, and spatial relationships shown in this image. The question must rely on visual information, be deterministic and answerable, and be one focused question rather than multiple sub-questions. Return result in a json format: {'Question': 'xxx', 'Thinkings': 'Your selection process'}"
REWRITE_PROMPT = """You are a Question generator. Your objective is to rewrite a given Q&A into a genuinely harder version that requires more visual evidence or reasoning steps. Make it harder by increasing the DEPTH of reasoning required. Do NOT make it harder by simply appending more sub-questions or extra clauses. Do NOT generate a question that asks the same thing as the original or is just a paraphrase. Do NOT start your rewritten question with the same first 5 words as the seed question. The question should not become broader, more subjective, or more generic. The added constraint must change the type or depth of reasoning required, not just expand the question's length.
Generate only ONE focused question grounded in the image and return JSON with key 'Rewritten question'."""
FILTER_PROMPT = """You are a strict JSON evaluation engine. Return a SINGLE JSON object with hard_constraints.Factuality.count, hard_constraints.Ambiguity.count, hard_constraints.Validity.answerable, soft_constraints.perception_difficulty and soft_constraints.reasoning_difficulty. A question is invalid when required information cannot be inferred from the image in principle. Output JSON only."""
TASKS = ("Object Detection", "Object Classification", "Attribute Recognition", "Pattern Recognition", "Spatial Reasoning", "Event Understanding", "Action Prediction", "Fact Recall", "Implicit Relation Inference", "Commonsense Reasoning", "Logical Reasoning", "Causal Reasoning", "Multi-Step Reasoning", "Abductive Reasoning", "Analogical Reasoning", "Constraint Satisfaction", "Social Reasoning")

@dataclass(frozen=True)
class SeeQConfig:
    retained_per_round: int = 10_000
    seed: int = SEED
    rounds: int = 2
    turns: int = 2
    candidates_per_image: int = 2

def _json_output(text: Any) -> Any:
    if isinstance(text, (dict, list)): return text
    match = re.search(r"\{.*\}", str(text), re.S)
    if not match: return {"text": str(text).strip()}
    payload = match.group(0)
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(payload)
        except (SyntaxError, ValueError):
            for key in ("Question", "Rewritten question"):
                hit = re.search(rf"[\"']{re.escape(key)}[\"']\s*:\s*(['\"])(.*?)\1\s*(?:,\s*[\"']|\}})", payload, re.S)
                if hit:
                    return {key: hit.group(2).strip()}
            return {"text": str(text).strip()}

def _rewrite_output(text: Any) -> Any:
    """Normalize the rewrite role: a bare question is the rewritten question."""
    out = text if isinstance(text, dict) else _json_output(text)
    if isinstance(out, dict) and "text" in out and out.keys() == {"text"}:
        return {"Rewritten question": out["text"]}
    return out

def _retain_round_robin(accepted: list[dict], retained: int) -> list[dict]:
    """Take one accepted triple per image per pass so 10k retained span many images."""
    groups: dict[str, list[dict]] = {}
    for row in accepted:
        groups.setdefault(str(row["image_id"]), []).append(row)
    out: list[dict] = []
    depth = 0
    while len(out) < retained:
        added = 0
        for rows in groups.values():
            if depth < len(rows):
                out.append(rows[depth]); added += 1
                if len(out) == retained: break
        if added == 0: break
        depth += 1
    return out

def accepted_triples(records: Iterable[dict], config: SeeQConfig = SeeQConfig()) -> list[dict]:
    accepted = [r for r in records if r.get("accepted") is True]
    retained = _retain_round_robin(accepted, config.retained_per_round)
    assert_exact_count(retained, config.retained_per_round, "retained SeeQ triples")
    return [{"image_id": str(r["image_id"]), "image": r["image"], "question": r["question"], "answer": r["answer"], "rejection_reason": None} for r in retained]

def dual_format_records(triples: Iterable[dict]) -> list[dict]:
    triples = list(triples)
    output = []
    for row in triples:
        base = {k: row[k] for k in ("image_id", "image", "question", "answer")}
        output.extend(({**base, "format": "qg", "input": "image", "target": row["question"] + " " + row["answer"]}, {**base, "format": "qa", "input": "image+question", "target": row["answer"]}))
    assert_exact_count(output, 2 * len(triples), "SeeQ dual-format records")
    return output

def check_stable_reference(m0_hash: str, current_m0_hash: str) -> None:
    if m0_hash != current_m0_hash: raise AssertionError("M0 rewrite/filter reference changed")

def _accepted(judge: Any) -> tuple[bool, str]:
    """Apply the released hard-constraint gate to one self-judge result."""
    if not isinstance(judge, dict):
        return False, "invalid_judge"
    # A deterministic adapter/test backend may return the already-decided form.
    if "accepted" in judge:
        if not isinstance(judge["accepted"], bool):
            return False, "invalid_acceptance_flag"
        return judge["accepted"], str(judge.get("rejection_reason", ""))
    hard = judge.get("hard_constraints")
    if not isinstance(hard, dict):
        return False, "missing_hard_constraints"
    factuality, ambiguity, validity = (hard.get(k) for k in ("Factuality", "Ambiguity", "Validity"))
    if isinstance(factuality, dict):
        factuality = factuality.get("count")
    if isinstance(ambiguity, dict):
        ambiguity = ambiguity.get("count")
    if isinstance(validity, dict):
        validity = validity.get("answerable")
    if not isinstance(factuality, (int, float)) or not isinstance(ambiguity, (int, float)):
        return False, "invalid_hard_constraint_counts"
    answerable = validity
    if isinstance(answerable, (int, float)) and answerable in (0, 1):
        answerable = bool(answerable)
    if not isinstance(answerable, bool):
        return False, "invalid_answerability"
    if factuality or ambiguity or not answerable:
        return False, "hard_constraint_failed"
    return True, ""

class Qwen3VLBackend:
    """Batched local adapter; reference roles default to the proposer model."""
    def __init__(self, model=None, processor=None, *, reference_model=None, reference_processor=None, base_model=None, reference_base_model=None, adapter=None, reference_adapter=None, device="cuda", batch_size=8, dtype="bfloat16", attn_implementation="flash_attention_2"):
        self.batch_size, self.device = batch_size, device
        if model is None:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
            processor = AutoProcessor.from_pretrained(base_model, local_files_only=True, use_fast=False)
            model = AutoModelForImageTextToText.from_pretrained(base_model, dtype=getattr(torch, dtype), local_files_only=True, attn_implementation=attn_implementation)
            if adapter:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, adapter)
            if device.startswith("cuda") and torch.cuda.is_available(): model.to(device)
            model.eval()
        self.model, self.processor = model, processor
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"
        if reference_model is None:
            if reference_base_model is None and reference_adapter is None:
                reference_model, reference_processor = model, processor
            else:
                import torch
                from transformers import AutoModelForImageTextToText, AutoProcessor
                reference_processor = AutoProcessor.from_pretrained(reference_base_model or base_model, local_files_only=True, use_fast=False)
                reference_model = AutoModelForImageTextToText.from_pretrained(reference_base_model or base_model, dtype=getattr(torch, dtype), local_files_only=True, attn_implementation=attn_implementation)
                if reference_adapter:
                    from peft import PeftModel
                    reference_model = PeftModel.from_pretrained(reference_model, reference_adapter)
                if device.startswith("cuda") and torch.cuda.is_available(): reference_model.to(device)
                reference_model.eval()
        self.reference_model, self.reference_processor = reference_model, reference_processor or processor
        if hasattr(self.reference_processor, "tokenizer"):
            self.reference_processor.tokenizer.padding_side = "left"

    @staticmethod
    def _image(value):
        if not isinstance(value, (str, Path)): return value
        path = Path(value)
        if not path.exists(): return value
        from PIL import Image
        with Image.open(path) as image: return image.convert("RGB")

    def _generate(self, model, processor, prompts: Sequence[str], images: Sequence[Any] | None, *, max_new_tokens=256, do_sample=False, temperature=0.0, top_p=1.0, top_k=None):
        prompts = list(prompts)
        if hasattr(model, "generate_text"): return [str(x) for x in model.generate_text(prompts, images=images)]
        import torch
        outputs = []
        for start in range(0, len(prompts), self.batch_size):
            p = prompts[start:start+self.batch_size]
            if images is None:
                im = None
            else:
                # SeeQ schedules four prompts per image.  Load each image once per
                # batch so disk decode is not repeated for the four candidates.
                image_cache = {}
                im = []
                for value in images[start:start+self.batch_size]:
                    key = str(value) if isinstance(value, (str, Path)) else id(value)
                    if key not in image_cache:
                        image_cache[key] = self._image(value)
                    im.append(image_cache[key])
            if im is not None:
                p = [processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}], tokenize=False, add_generation_prompt=True) for prompt in p]
                kwargs = {"text": p, "images": im, "padding": True, "return_tensors": "pt"}
            else:
                kwargs = {"text": p, "padding": True, "return_tensors": "pt"}
            batch = processor(**kwargs)
            target_device = getattr(model, "device", None)
            if target_device is not None and hasattr(batch, "to"): batch = batch.to(target_device)
            generation = {"max_new_tokens": max_new_tokens, "do_sample": do_sample, "temperature": (temperature or 1.0), "top_p": top_p}
            if top_k is not None:
                generation["top_k"] = int(top_k)
            with torch.inference_mode(): ids = model.generate(**batch, **generation)
            prompt_width = batch["input_ids"].shape[1]
            outputs.extend(processor.tokenizer.decode(row[prompt_width:], skip_special_tokens=True) for row in ids)
        return [x.strip() for x in outputs]

    def generate(self, prompts, images=None, **kwargs): return self._generate(self.model, self.processor, prompts, images, **kwargs)

    def generate_role(self, prompts, images=None, *, reference=False, do_sample=False, temperature=0.0, top_p=1.0, top_k=None, max_new_tokens=256):
        model, processor = (self.reference_model, self.reference_processor) if reference else (self.model, self.processor)
        return self._generate(model, processor, prompts, images, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k)
    def _role(self, prompt, images, values, *, reference=False, **kwargs):
        model, processor = (self.reference_model, self.reference_processor) if reference else (self.model, self.processor)
        return [_json_output(x) for x in self._generate(model, processor, [prompt + v for v in values], images, **kwargs)]
    def propose(self, images, tasks): return self._role(PROPOSAL_PROMPT + " Task family: ", images, tasks, max_new_tokens=256)
    def answer(self, images, questions): return self.generate([ANSWER_PROMPT + " Question: " + q for q in questions], images, max_new_tokens=128)
    def rewrite(self, images, questions, answers): return [_rewrite_output(x) for x in self._role(REWRITE_PROMPT, images, [f"\nQuestion: {q}\nAnswer: {a}" for q,a in zip(questions, answers)], reference=True, max_new_tokens=256)]
    def judge(self, images, questions, answers): return self._role(FILTER_PROMPT, images, [f"\nQuestion: {q}\nAnswer: {a}" for q,a in zip(questions, answers)], reference=True, max_new_tokens=512)
    filter = judge

class SeeQPipeline:
    def __init__(self, backend, config=SeeQConfig(), *, fact_cache=None, sample_bank=None): self.backend, self.config, self.fact_cache, self.sample_bank = backend, config, fact_cache or {}, sample_bank or {}
    @staticmethod
    def schedule(images, config=SeeQConfig(), round_id=1):
        rows = []
        for i, image in enumerate(images):
            image_path = image.get("image", image.get("image_path"))
            for turn in range(config.turns):
                for candidate in range(config.candidates_per_image):
                    index = i * config.turns * config.candidates_per_image + turn * config.candidates_per_image + candidate
                    rows.append({"proposal_id": f"r{round_id}-{i:06d}-{turn}-{candidate}", "image_id": str(image["image_id"]), "image": image_path, "task": TASKS[(index + config.seed + round_id) % len(TASKS)], "round": round_id, "turn": turn, "candidate": candidate})
        return rows
    @staticmethod
    def _stage(path, expected):
        if not path.exists(): return None
        rows = read_jsonl(path)
        ids = [str(r["proposal_id"]) for r in rows]
        if ids != expected[:len(ids)]: raise AssertionError(f"inconsistent SeeQ stage cache: {path}")
        return rows
    def run(self, images, stage_dir: Path, *, round_id=1, limit=None):
        stage_dir.mkdir(parents=True, exist_ok=True)
        proposals = self.schedule(list(images), self.config, round_id)
        if limit is not None:
            proposals = proposals[:limit]
        paths = [stage_dir / f"stage_{n}.jsonl" for n in ("a_proposals", "b_answers", "c_rewrites", "d_filters", "e_retained")]
        ids = [r["proposal_id"] for r in proposals]
        cached = self._stage(paths[0], ids)
        if cached is None:
            write_jsonl(paths[0], proposals)
        elif len(cached) == len(proposals):
            proposals = cached
        else:
            write_jsonl(paths[0], proposals)

        step = max(1, int(getattr(self.backend, "batch_size", 64)))
        answers = self._stage(paths[1], ids) or []
        if len(answers) < len(proposals):
            pending = proposals[len(answers):]
            for start in range(0, len(pending), step):
                chunk = pending[start:start + step]
                qs = self.backend.propose([r["image"] for r in chunk], [r["task"] for r in chunk])
                answers += [{**r, "proposal": str(x.get("Question", x.get("text", x))) if isinstance(x, dict) else str(x)} for r, x in zip(chunk, qs)]
                write_jsonl(paths[1], answers)
            if len(answers) != len(proposals):
                raise AssertionError("SeeQ proposer returned wrong record count")

        rewrites = self._stage(paths[2], ids) or []
        if len(rewrites) < len(answers):
            pending = answers[len(rewrites):]
            for start in range(0, len(pending), step):
                chunk = pending[start:start + step]
                aa = self.backend.answer([r["image"] for r in chunk], [r["proposal"] for r in chunk])
                rr = self.backend.rewrite([r["image"] for r in chunk], [r["proposal"] for r in chunk], aa)
                rewrites += [{**r, "answer": str(a), "rewrite": x} for r, a, x in zip(chunk, aa, rr)]
                write_jsonl(paths[2], rewrites)
            if len(rewrites) != len(answers):
                raise AssertionError("SeeQ rewrite returned wrong record count")

        filtered = self._stage(paths[3], ids) or []
        if len(filtered) < len(rewrites):
            pending = rewrites[len(filtered):]
            for start in range(0, len(pending), step):
                chunk = pending[start:start + step]
                questions = [str(r["rewrite"].get("Rewritten question", r["rewrite"].get("Question", r["rewrite"].get("text", r["proposal"])))) if isinstance(r["rewrite"], dict) else str(r["rewrite"]) for r in chunk]
                judges = self.backend.filter([r["image"] for r in chunk], questions, [r["answer"] for r in chunk])
                for r, q, j in zip(chunk, questions, judges):
                    ok, reason = _accepted(j)
                    filtered.append({**r, "question": q, "judge": j, "accepted": ok, "rejection_reason": reason, "source": "seeq_self_evolution", "fact_cache_used": str(r["image_id"]) in self.fact_cache, "sample_bank_used": str(r["image_id"]) in self.sample_bank, "rag_used": False})
                write_jsonl(paths[3], filtered)
            if len(filtered) != len(rewrites):
                raise AssertionError("SeeQ filter returned wrong record count")

        accepted = [r for r in filtered if r.get("accepted") is True]
        retained = accepted_triples(filtered, self.config)
        write_jsonl(paths[4], [{k: r[k] for k in ("image_id", "image", "question", "answer")} for r in retained])
        return {"proposals": len(proposals), "answers": len(answers), "rewrites": len(rewrites), "rejected": len(filtered) - len(accepted), "accepted": len(accepted), "retained": len(retained), "images_retained": len({r["image_id"] for r in retained}), "images_visited": len({r["image_id"] for r in proposals}), "stage_dir": str(stage_dir), "commit": SEEQ_COMMIT}

__all__ = ["Qwen3VLBackend", "SeeQPipeline", "SeeQConfig", "accepted_triples", "dual_format_records", "check_stable_reference", "SEEQ_COMMIT", "TASKS"]
