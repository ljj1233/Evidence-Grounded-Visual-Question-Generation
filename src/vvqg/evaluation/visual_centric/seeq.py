"""Failure-safe, cacheable runner for the audited SeeQ single-question rubric."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import httpx
from tqdm.auto import tqdm

from vvqg.evaluation.common.cache import needs_request

ROOT = Path(__file__).resolve().parents[4]
AUDIT = ROOT / "outputs/phase23/evaluation/metric_audit.json"
VERSION = "seeq-single-question-v1"
DIMENSIONS = (
    "visual_search_difficulty",
    "visual_evidence_coverage",
    "visual_context_reasoning",
    "visual_spatial_reasoning",
)


def _audit() -> dict[str, Any]:
    return json.loads(AUDIT.read_text(encoding="utf-8"))


def load_env() -> dict[str, str]:
    values: dict[str, str] = {}
    env_path = ROOT / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                values[key] = value.strip().strip('"').strip("'")
    values.update({key: value for key, value in os.environ.items() if value})
    return values


def image_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_key(record: dict[str, Any], model: str, prompt_hash: str, image_hash: str) -> str:
    payload = {
        "version": VERSION,
        "model": model,
        "prompt_hash": prompt_hash,
        "image_id": str(record["image_id"]),
        "image_hash": image_hash,
        "question": record["question"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _record_prompt(record: dict[str, Any], user_template: str) -> str:
    payload = {
        "question_id": f"{record.get('system_id', 'system')}:{record['image_id']}",
        "question": record["question"],
        "reference_answer": record.get("reference_answer", ""),
        "prompt": record.get("prompt", ""),
    }
    return user_template.replace("{record_json}", json.dumps(payload, indent=2, ensure_ascii=False))


def parse_scores(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        raise ValueError("judge response is not an object")
    if any(key not in raw for key in DIMENSIONS):
        raise ValueError("judge response is missing a required dimension")
    scores = {key: raw[key] for key in DIMENSIONS}
    if any(type(value) is not int or not 0 <= value <= 5 for value in scores.values()):
        raise ValueError("judge scores must be integer values in [0,5]")
    return scores


def _image_part(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


async def evaluate(
    records: list[dict[str, Any]],
    output: Path,
    *,
    model: str = "gpt-5.4",
    concurrency: int = 16,
    allow_variant: bool = False,
    label: str = "SeeQ evaluation",
    retry_failures: bool = False,
) -> dict[str, Any]:
    audit = _audit()
    dimension = audit["seeq_protocol"]["dimensions"][0]
    if model != "gpt-5.4" and not allow_variant:
        raise ValueError(f"model {model!r} is not official SeeQ GPT-5.4; pass --allow-variant to label a variant")
    env = load_env()
    url = env.get("LLM_JUDGE_BASE_URL") or env.get("AUDIT_JUDGE_BASE_URL")
    key = env.get("LLM_JUDGE_API_KEY") or env.get("AUDIT_JUDGE_API_KEY")
    if not url or not key:
        raise RuntimeError("LLM_JUDGE_BASE_URL and LLM_JUDGE_API_KEY are required")
    system_prompt = dimension["prompt_system"]
    user_template = dimension["prompt_user"]
    prompt_hash = hashlib.sha256((system_prompt + user_template).encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    cached = {}
    if output.is_file():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                cached[row["cache_key"]] = row
    pending = []
    for record in records:
        image = ROOT / record["image"] if not Path(record["image"]).is_absolute() else Path(record["image"])
        image_hash = image_digest(image)
        key_id = cache_key(record, model, prompt_hash, image_hash)
        if needs_request(key_id, cached, retry_failures):
            pending.append((record, image, image_hash, key_id))
    sem = asyncio.Semaphore(concurrency)
    started = time.perf_counter()

    async def one(item: tuple[dict[str, Any], Path, str, str], client: httpx.AsyncClient) -> dict[str, Any]:
        record, image, image_hash, key_id = item
        request = {
            "model": model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": _record_prompt(record, user_template)}, _image_part(image)]},
            ],
        }
        base = {"cache_key": key_id, "image_id": str(record["image_id"]), "question": record["question"], "model": model, "image_hash": image_hash, "status": "error"}
        try:
            async with sem:
                response = await client.post(url, json=request)
            response.raise_for_status()
            body = response.json()
            raw = body["choices"][0]["message"]["content"]
            parsed = json.loads(raw)
            scores = parse_scores(parsed)
            return {**base, "status": "ok", "scores": scores}
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            return {**base, "error_type": type(error).__name__, "error": str(error)[:500]}

    if pending:
        async with httpx.AsyncClient(trust_env=False, timeout=float(env.get("LLM_JUDGE_TIMEOUT", "180")), headers={"Authorization": f"Bearer {key}"}) as client:
            bar = tqdm(total=len(records), initial=len(records) - len(pending), desc=label, unit="q", dynamic_ncols=True)
            for task in asyncio.as_completed([one(item, client) for item in pending]):
                row = await task
                cached[row["cache_key"]] = row
                bar.update(1)
                output.write_text("\n".join(json.dumps(cached[k], ensure_ascii=False) for k in sorted(cached)) + "\n", encoding="utf-8")
            bar.close()
    rows = [cached[cache_key(r, model, prompt_hash, image_digest(ROOT / r["image"] if not Path(r["image"]).is_absolute() else Path(r["image"]))) ] for r in records]
    valid = [r for r in rows if r["status"] == "ok"]
    summary = {key: sum(r["scores"][key] for r in valid) / (5 * len(valid)) if valid else None for key in DIMENSIONS}
    runtime = {"schema": VERSION, "model": model, "status": "VARIANT_DOCUMENTED" if model != "gpt-5.4" else "VERIFIED_OFFICIAL", "requested": len(records), "cached": len(records) - len(pending), "new_requests": len(pending), "ok": len(valid), "failed": len(rows) - len(valid), "wall_seconds": round(time.perf_counter() - started, 3), "concurrency": concurrency, "prompt_hash": prompt_hash, "failed_error_types": {name: sum(r.get("error_type") == name for r in rows) for name in sorted({r.get("error_type") for r in rows if r.get("error_type")})}, "summary": summary}
    output.with_name(output.stem + "_runtime.json").write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    return runtime
