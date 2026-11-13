"""Three generation workflows sharing one provider interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .prompts import PromptSet
from .providers import CallRecorder, CompletionRequest, OutputParseError, parse_json_object
from .records import QuestionStatus

METHODS = ("plain", "strong", "evidence_anchored")


class GenerationFailure(RuntimeError):
    """The model call completed but the output cannot be turned into a question."""

    def __init__(self, reason: str, raw_output: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.raw_output = raw_output


@dataclass
class GenerationOutcome:
    status: str
    question: str | None
    raw_output: str | None
    failure_reason: str | None


def _parse_items(raw: str, stage: str) -> list[dict[str, str]]:
    try:
        data = parse_json_object(raw)
    except OutputParseError as exc:
        raise GenerationFailure(f"{stage}_output_parse_error", raw) from exc
    items = data.get("items")
    if not isinstance(items, list):
        raise GenerationFailure(f"{stage}_items_missing", raw)
    parsed: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        location = item.get("location", "unspecified")
        if isinstance(content, str) and content.strip():
            parsed.append({"location": str(location), "content": content.strip()})
    if not parsed:
        raise GenerationFailure(f"{stage}_empty_evidence", raw)
    return parsed


def _parse_pair(raw: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    try:
        data = parse_json_object(raw)
    except OutputParseError as exc:
        raise GenerationFailure("pair_evidence_output_parse_error", raw) from exc
    pair = data.get("pair")
    if not isinstance(pair, dict):
        raise GenerationFailure("pair_evidence_missing_pair", raw)
    text_items = pair.get("text", [])
    image_items = pair.get("image", [])
    if not isinstance(text_items, list) or not isinstance(image_items, list):
        raise GenerationFailure("pair_evidence_invalid_lists", raw)
    clean_text = [
        {"location": str(item.get("location", "unspecified")), "content": str(item.get("content", "")).strip()}
        for item in text_items
        if isinstance(item, dict) and str(item.get("content", "")).strip()
    ]
    clean_image = [
        {"location": str(item.get("location", "unspecified")), "content": str(item.get("content", "")).strip()}
        for item in image_items
        if isinstance(item, dict) and str(item.get("content", "")).strip()
    ]
    if not clean_text or not clean_image:
        raise GenerationFailure("pair_evidence_incomplete", raw)
    return clean_text, clean_image


def _format_evidence(text_items: list[dict[str, str]], image_items: list[dict[str, str]]) -> str:
    lines = ["文本证据："]
    lines.extend(f"- [{item['location']}] {item['content']}" for item in text_items)
    lines.append("图像证据：")
    lines.extend(f"- [{item['location']}] {item['content']}" for item in image_items)
    return "\n".join(lines)


def _format_extracted(items: list[dict[str, str]]) -> str:
    return "\n".join(f"- [{item['location']}] {item['content']}" for item in items)


def _image_note(image_path: Path | None) -> str:
    if image_path is None:
        return "（本次没有提供图像。）"
    return "图像随请求一并提供，请查看附图。"


def run_generation(
    provider,
    prompts: PromptSet,
    recorder: CallRecorder,
    material: Mapping[str, Any],
    method: str,
    *,
    material_image_path: Path | None,
    run_id: str,
    protocol_version: str,
    question_id: str,
    generator_model: str,
) -> GenerationOutcome:
    if method not in METHODS:
        raise ValueError(f"unknown generation method: {method}")

    material_id = material["material_id"]
    image_note = _image_note(material_image_path)
    evidence_block = ""
    if method == "evidence_anchored":
        text_completion = recorder.complete(
            provider,
            CompletionRequest(
                task="extract_text_evidence",
                prompt=prompts.render("extract_text_evidence", text=material["text"]),
                text=material["text"],
                metadata={"material_id": material_id},
            ),
            role="generator",
            material_id=material_id,
            question_id=question_id,
            model=generator_model,
        )
        text_items = _parse_items(text_completion.text, "text_evidence")

        image_completion = recorder.complete(
            provider,
            CompletionRequest(
                task="extract_image_evidence",
                prompt=prompts.render("extract_image_evidence", image_note=image_note),
                image_path=str(material_image_path) if material_image_path else None,
                metadata={"material_id": material_id},
            ),
            role="generator",
            material_id=material_id,
            question_id=question_id,
            model=generator_model,
        )
        image_items = _parse_items(image_completion.text, "image_evidence")

        pair_completion = recorder.complete(
            provider,
            CompletionRequest(
                task="pair_evidence",
                prompt=prompts.render(
                    "pair_evidence",
                    text_evidence=_format_extracted(text_items),
                    image_evidence=_format_extracted(image_items),
                ),
                metadata={"material_id": material_id},
            ),
            role="generator",
            material_id=material_id,
            question_id=question_id,
            model=generator_model,
        )
        text_pair, image_pair = _parse_pair(pair_completion.text)
        evidence_block = _format_evidence(text_pair, image_pair)
        prompt_name = "qg_evidence"
    elif method == "strong":
        prompt_name = "qg_strong"
    else:
        prompt_name = "qg_plain"

    prompt = prompts.render(
        prompt_name,
        text=material["text"],
        image_note=image_note,
        evidence=evidence_block,
    )
    completion = recorder.complete(
        provider,
        CompletionRequest(
            task="generate_question",
            prompt=prompt,
            text=material["text"],
            image_path=str(material_image_path) if material_image_path else None,
            metadata={"material_id": material_id, "method": method, "protocol_version": protocol_version},
        ),
        role="generator",
        material_id=material_id,
        question_id=question_id,
        model=generator_model,
    )

    if completion.finish_reason == "refusal":
        raise GenerationFailure("model_refusal", completion.text)
    if completion.finish_reason != "stop":
        raise GenerationFailure("incomplete_completion", completion.text)
    try:
        data = parse_json_object(completion.text)
    except OutputParseError as exc:
        raise GenerationFailure("output_parse_error", completion.text) from exc
    if set(data) != {"question"}:
        raise GenerationFailure("generation_must_only_output_question", completion.text)
    question = data.get("question")
    if not isinstance(question, str) or not question.strip():
        raise GenerationFailure("empty_question", completion.text)
    return GenerationOutcome(
        status=QuestionStatus.OK,
        question=question.strip(),
        raw_output=completion.text,
        failure_reason=None,
    )
