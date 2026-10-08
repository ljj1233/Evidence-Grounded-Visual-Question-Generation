"""Independent GPT-4o support labels with content-addressed resume."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import resource
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

import httpx
from tqdm.auto import tqdm

from vvqg.evaluation.common.cache import needs_request

from vvqg.rewards.support.calibration import LABELS, overall_support, parse_labels

VERSION = "visual-support-phase23-v1.0"
SYSTEM = '''You are a strict image-grounded visual-support annotator. Return JSON only with keys premise_consistency, reference_clarity, visual_answerability, failure_type, reason. Labels: premise_consistency=supported|unsupported|uncertain; reference_clarity=clear|ambiguous|not_applicable|uncertain; visual_answerability=answerable|unanswerable|uncertain; failure_type=none|missing_entity|wrong_attribute|wrong_relation|ambiguous_reference|unobservable_target|external_knowledge_required|other|uncertain. A yes/no question may be valid when the answer is No. A count question may be valid when the answer is zero. A queried property is not automatically a premise. External or non-visible information is visually unanswerable. When evidence is uncertain, use uncertain rather than guessing. Judge only visible image evidence.'''
FAILURE_TYPES = ("missing_entity", "wrong_attribute", "wrong_relation", "ambiguous_reference", "unobservable_target", "external_knowledge_required")


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


@lru_cache(maxsize=None)
def image_hash(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=128)
def image_url(path: str) -> str:
    mime = "image/jpeg" if Path(path).suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return f"data:{mime};base64,{base64.b64encode(Path(path).read_bytes()).decode()}"


def cache_identity(row: dict, model: str, endpoint: str) -> str:
    identity = [VERSION, model, endpoint, SYSTEM, str(row["image_id"]), image_hash(row["image"]), row["question"]]
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()


def summarize(records: list[dict]) -> dict:
    success = [r for r in records if r["status"] == "ok"]
    n = len(success)
    result = {"requested": len(records), "valid": n, "api_failures": sum(r["status"] == "api_failure" for r in records),
              "parser_failures": sum(r["status"] == "parser_failure" for r in records), "rate_denominator": "valid parsed labels"}
    for dimension, labels in LABELS.items():
        counts = Counter(r[dimension] for r in success)
        result[dimension] = {"counts": {v: counts[v] for v in labels}, "rates": {v: counts[v] / n if n else None for v in labels}}
    result["six_failure_type_rates"] = {v: result["failure_type"]["rates"][v] for v in FAILURE_TYPES}
    return result


async def evaluate(inputs: dict[str, Path], out: Path, endpoint: str, key: str, model: str = "gpt-4o", concurrency: int = 16, limit: int | None = None, retry_failures: bool = False, label: str = "Visual support") -> dict:
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "cache.jsonl"
    cached = {r["cache_key"]: r for r in rows(cache_path)} if cache_path.exists() else {}
    records = []
    for system_id, path in inputs.items():
        for i, row in enumerate(rows(path)):
            if limit is not None and i >= limit:
                break
            row = {**row, "image": row.get("image") or row["image_path"], "system_id": system_id}
            row["cache_key"] = cache_identity(row, model, endpoint)
            records.append(row)
    unique = {r["cache_key"]: r for r in records}
    pending = [r for ck, r in unique.items() if needs_request(ck, cached, retry_failures)]
    sem = asyncio.Semaphore(concurrency)
    started = time.monotonic()
    async with httpx.AsyncClient(timeout=120, headers={"Authorization": f"Bearer {key}"}, trust_env=False) as client:
        async def one(row: dict) -> dict:
            result = {"cache_key": row["cache_key"], "image_id": row["image_id"], "image_sha256": image_hash(row["image"]), "question": row["question"], "model": model, "evaluator_version": VERSION}
            async with sem:
                content = [{"type": "text", "text": f"{SYSTEM}\n\nQuestion: {row['question']}"}, {"type": "image_url", "image_url": {"url": image_url(row["image"])}}]
                try:
                    response = await client.post(endpoint, json={"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0, "response_format": {"type": "json_object"}})
                    response.raise_for_status()
                except httpx.HTTPError as error:
                    return {**result, "status": "api_failure", "error_type": type(error).__name__, "http_status": error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None}
            try:
                payload = response.json()
                raw = payload["choices"][0]["message"]["content"]
                labels = parse_labels(raw)
                labels["overall_visual_support"] = overall_support(labels["premise_consistency"], labels["reference_clarity"], labels["visual_answerability"])
            except (ValueError, KeyError, TypeError, IndexError) as error:
                return {**result, "status": "parser_failure", "error_type": type(error).__name__, "raw_output": response.text}
            return {**result, "status": "ok", **labels, "raw_output": raw, "returned_model": payload.get("model"), "usage": payload.get("usage", {})}
        with cache_path.open("a", encoding="utf-8") as handle:
            bar = tqdm(total=len(records), initial=len(records) - len(pending), desc=label, unit="q", dynamic_ncols=True)
            for task in asyncio.as_completed([one(row) for row in pending]):
                result = await task
                cached[result["cache_key"]] = result
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                bar.update(1)
            bar.close()
    summaries = {}
    for system_id in inputs:
        labeled = [{**r, **cached[r["cache_key"]]} for r in records if r["system_id"] == system_id]
        with (out / f"{system_id}.jsonl").open("w", encoding="utf-8") as handle:
            for row in labeled:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summaries[system_id] = summarize(labeled)
    seconds = time.monotonic() - started
    report = {"model": model, "prompt": SYSTEM, "evaluator_version": VERSION, "temperature": 0, "concurrency": concurrency,
              "runtime": {"api_requests": len(pending), "cache_hits": len(records) - len(pending), "wall_seconds": seconds, "requests_per_second": len(pending) / seconds, "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024},
              "inputs": {k: {"path": str(v), "sha256": hashlib.sha256(v.read_bytes()).hexdigest()} for k, v in inputs.items()}, "systems": summaries}
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
