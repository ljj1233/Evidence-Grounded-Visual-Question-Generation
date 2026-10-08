"""Image-only visual concept proposal: prompt, strict schema, parsing.

The proposer sees the image only and returns explicit concepts. This module has
no model code; scripts own loading and generation.
"""

from __future__ import annotations

import json
import re

PROMPT = (
    "You are an image annotation system. Look at the image only. "
    "Extract visual concepts that a question could target. "
    "Return ONLY a compact JSON object on one line, no prose, with this exact schema:\n"
    '{"objects":[{"text":"<short phrase>","confidence":<0-1>}],'
    '"attributes":[{"text":"<short phrase>","confidence":<0-1>}],'
    '"relations":[{"text":"<short phrase>","confidence":<0-1>}],'
    '"scene":[{"text":"<short phrase>","confidence":<0-1>}]}\n'
    "Every element MUST be an object with a \"text\" string and a numeric "
    "\"confidence\". Rules: objects are visible things (max 4); attributes are "
    "visible properties such as color, size or material (max 4); relations are "
    "visible spatial or action relations between things (max 4); scene is the "
    "overall scene or location (max 2); each text has at most 8 English words; do "
    "not output questions, answers or explanations."
)

CAPS = {"objects": 4, "attributes": 4, "relations": 4, "scene": 2}
TYPES = tuple(CAPS)
SINGULAR = {"objects": "object", "attributes": "attribute", "relations": "relation", "scene": "scene"}
_ANSWER_PREFIX = re.compile(
    r"^(how many|what|where|who|which|why|when|is |are |does |do |can |could |did )",
    re.IGNORECASE,
)


def extract_json(raw: str):
    """Return the parsed dict or None. Records parse failure via the caller."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.partition("\n")[2].rsplit("```", 1)[0].strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _clean(text) -> str | None:
    if not isinstance(text, str):
        return None
    text = " ".join(text.split())
    if not text or len(text.split()) > 8:
        return None
    if text.endswith("?") or _ANSWER_PREFIX.match(text):
        return None
    return text


def parse_concepts(raw: str):
    """Parse one raw response into normalised concepts + status.

    Returns (concepts, status, parsed). status is 'ok' | 'parser_failure' |
    'empty'. A parser failure is never silently dropped: it is returned so the
    caller can flag fallback.
    """
    data = extract_json(raw)
    if data is None:
        return {}, "parser_failure", None
    concepts: dict[str, list[dict]] = {}
    for kind in TYPES:
        seen, kept = set(), []
        values = data.get(kind) or []
        for item in values if isinstance(values, list) else []:
            text = _clean(item.get("text") if isinstance(item, dict) else item)
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            raw_conf = item.get("confidence", 0.0) if isinstance(item, dict) else 0.5
            try:
                conf = min(1.0, max(0.0, float(raw_conf)))
            except (TypeError, ValueError):
                conf = 0.0
            kept.append({"text": text, "confidence": conf})
            if len(kept) >= CAPS[kind]:
                break
        concepts[kind] = kept
    status = "ok" if any(concepts.values()) else "empty"
    return concepts, status, data


def flatten(concepts: dict) -> list[dict]:
    """Concept list with type attached, in fixed type order."""
    return [
        {"concept_type": SINGULAR[kind], "concept_text": item["text"], "concept_confidence": item["confidence"]}
        for kind in TYPES
        for item in concepts.get(kind, [])
    ]
