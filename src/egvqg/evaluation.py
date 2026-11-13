"""Independent answer annotation, validity checks, four-condition answering and scoring."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any, Mapping

from .prompts import PromptSet
from .providers import CallRecorder, CompletionRequest, OutputParseError, ProviderError, parse_json_object
from .records import (
    CONDITIONS,
    AnswerRecord,
    AnswerStatus,
    BDRState,
    QuestionStatus,
    ResponseRecord,
    ResponseStatus,
    ScoreRecord,
    ScoreStatus,
    ValidityRecord,
    ValidityStatus,
    sha256_text,
    stable_id,
    utc_now,
)


def _image_note(image_path: Path | None) -> str:
    if image_path is None:
        return "（本次没有提供图像。）"
    return "图像随请求一并提供，请查看附图。"


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def validate_question(
    provider,
    prompts: PromptSet,
    recorder: CallRecorder,
    *,
    material: Mapping[str, Any],
    method: str,
    question: str,
    material_image_path: Path | None,
    run_id: str,
    protocol_version: str,
    question_id: str,
    validator_model: str,
    source: str,
) -> ValidityRecord:
    prompt = prompts.render(
        "validate_question",
        text=material["text"],
        image_note=_image_note(material_image_path),
        question=question,
    )
    reasons: list[str] = []
    status = ValidityStatus.UNRESOLVED
    try:
        completion = recorder.complete(
            provider,
            CompletionRequest(
                task="validate_question",
                prompt=prompt,
                question=question,
                text=material["text"],
                image_path=str(material_image_path) if material_image_path else None,
            ),
            role="validator",
            material_id=material["material_id"],
            question_id=question_id,
            model=validator_model,
        )
    except ProviderError as exc:
        reasons = [f"provider_technical_failure: {exc}"]
    else:
        try:
            data = parse_json_object(completion.text)
        except OutputParseError as exc:
            reasons = [f"validity_output_parse_error: {exc}"]
        else:
            candidate_status = data.get("status")
            if candidate_status in {
                ValidityStatus.VALID,
                ValidityStatus.INVALID,
                ValidityStatus.UNRESOLVED,
            }:
                status = candidate_status
            reasons = _string_list(data.get("reasons")) or reasons
    return ValidityRecord(
        validity_id=stable_id("validity", run_id, question_id),
        run_id=run_id,
        protocol_version=protocol_version,
        question_id=question_id,
        material_id=material["material_id"],
        method=method,
        status=status,
        reasons=reasons,
        source=source,
        validator_model=validator_model,
        created_at=utc_now(),
    )


def annotate_and_verify_answer(
    provider,
    prompts: PromptSet,
    recorder: CallRecorder,
    *,
    material: Mapping[str, Any],
    method: str,
    question: str,
    material_image_path: Path | None,
    run_id: str,
    protocol_version: str,
    question_id: str,
    annotation_model: str,
    verifier_model: str,
    source: str,
) -> AnswerRecord:
    prompt = prompts.render(
        "annotate_answer",
        text=material["text"],
        image_note=_image_note(material_image_path),
        question=question,
    )
    candidate_status = AnswerStatus.UNRESOLVED
    candidate_answer: str | None = None
    candidate_acceptable: list[str] = []
    candidate_rubric: str | None = None
    evidence_text: list[str] = []
    evidence_image: list[str] = []
    notes: list[str] = []

    try:
        completion = recorder.complete(
            provider,
            CompletionRequest(
                task="annotate_answer",
                prompt=prompt,
                question=question,
                text=material["text"],
                image_path=str(material_image_path) if material_image_path else None,
            ),
            role="annotator",
            material_id=material["material_id"],
            question_id=question_id,
            model=annotation_model,
        )
    except ProviderError as exc:
        notes.append(f"annotation_provider_technical_failure: {exc}")
    else:
        try:
            data = parse_json_object(completion.text)
        except OutputParseError as exc:
            notes.append(f"annotation_output_parse_error: {exc}")
        else:
            if data.get("status") == AnswerStatus.CANDIDATE:
                candidate_status = AnswerStatus.CANDIDATE
            candidate_answer = data.get("answer") if isinstance(data.get("answer"), str) else None
            candidate_acceptable = _string_list(data.get("acceptable_answers"))
            candidate_rubric = data.get("rubric") if isinstance(data.get("rubric"), str) else None
            evidence_text = _string_list(data.get("evidence_text"))
            evidence_image = _string_list(data.get("evidence_image"))
            notes.extend(_string_list(data.get("notes")))
            if candidate_status == AnswerStatus.CANDIDATE and not candidate_answer:
                candidate_status = AnswerStatus.UNRESOLVED
                notes.append("candidate_answer_missing")

    final_status = AnswerStatus.UNRESOLVED
    final_answer: str | None = None
    final_acceptable: list[str] = []
    final_rubric: str | None = None
    verification_notes: str | None = "; ".join(notes) if notes else None

    if candidate_status == AnswerStatus.CANDIDATE and candidate_answer:
        verify_prompt = prompts.render(
            "verify_answer",
            text=material["text"],
            image_note=_image_note(material_image_path),
            question=question,
            candidate_answer=candidate_answer,
            candidate_acceptable="\n".join(candidate_acceptable) if candidate_acceptable else "（无）",
            candidate_rubric=candidate_rubric or "（无）",
        )
        try:
            verify_completion = recorder.complete(
                provider,
                CompletionRequest(
                    task="verify_answer",
                    prompt=verify_prompt,
                    question=question,
                    text=material["text"],
                    image_path=str(material_image_path) if material_image_path else None,
                ),
                role="verifier",
                material_id=material["material_id"],
                question_id=question_id,
                model=verifier_model,
            )
        except ProviderError as exc:
            verification_notes = "; ".join([*notes, f"verification_provider_technical_failure: {exc}"])
        else:
            try:
                data = parse_json_object(verify_completion.text)
            except OutputParseError as exc:
                verification_notes = "; ".join([*notes, f"verification_output_parse_error: {exc}"])
            else:
                status = data.get("status")
                verified_answer = data.get("answer") if isinstance(data.get("answer"), str) else None
                verified_acceptable = _string_list(data.get("acceptable_answers"))
                verified_rubric = data.get("rubric") if isinstance(data.get("rubric"), str) else None
                if status == AnswerStatus.VERIFIED and (verified_answer or candidate_answer):
                    final_status = AnswerStatus.VERIFIED
                    final_answer = verified_answer or candidate_answer
                    final_acceptable = verified_acceptable or candidate_acceptable
                    final_rubric = verified_rubric or candidate_rubric
                    if final_answer not in final_acceptable:
                        final_acceptable = [final_answer, *final_acceptable]
                    final_acceptable = _dedupe(final_acceptable)
                verification_notes = "; ".join([*notes, *_string_list(data.get("notes"))]) or None

    return AnswerRecord(
        answer_id=stable_id("answer", run_id, question_id),
        run_id=run_id,
        protocol_version=protocol_version,
        question_id=question_id,
        material_id=material["material_id"],
        method=method,
        status=final_status,
        candidate_answer=candidate_answer,
        candidate_acceptable_answers=candidate_acceptable,
        answer=final_answer,
        acceptable_answers=final_acceptable,
        rubric=final_rubric,
        evidence_text=evidence_text,
        evidence_image=evidence_image,
        annotation_model=annotation_model,
        verifier_model=verifier_model,
        verification_notes=verification_notes,
        source=source,
        created_at=utc_now(),
    )


def _materials_block(material: Mapping[str, Any], condition: str, image_path: Path | None) -> str:
    text = material["text"]
    if condition == "TI":
        if image_path is None:
            return f"文本材料：\n{text}\n\n（本次没有提供图像。）"
        return f"文本材料：\n{text}\n\n图像：见附图。"
    if condition == "T":
        return f"文本材料：\n{text}"
    if condition == "I":
        if image_path is None:
            return "（本次没有提供图像。）"
        return "图像：见附图。"
    if condition == "0":
        return "（没有提供任何材料。只依据问题本身作答。）"
    raise ValueError(f"unknown condition: {condition}")


def build_condition_request(
    prompts: PromptSet,
    *,
    material: Mapping[str, Any],
    question: str,
    condition: str,
    image_path: Path | None,
) -> CompletionRequest:
    """Build an isolated request. Removed material and all answer-side data are absent."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    text = material["text"] if condition in {"TI", "T"} else None
    included_image = image_path if condition in {"TI", "I"} else None
    prompt = prompts.render(
        "answer_question",
        question=question,
        materials=_materials_block(material, condition, included_image),
    )
    return CompletionRequest(
        task="answer_question",
        prompt=prompt,
        question=question,
        text=text,
        image_path=str(included_image) if included_image else None,
        condition=condition,
    )


def answer_question(
    provider,
    prompts: PromptSet,
    recorder: CallRecorder,
    *,
    material: Mapping[str, Any],
    method: str,
    question: str,
    condition: str,
    material_image_path: Path | None,
    run_id: str,
    protocol_version: str,
    question_id: str,
    answerer_model: str,
) -> ResponseRecord:
    request = build_condition_request(
        prompts,
        material=material,
        question=question,
        condition=condition,
        image_path=material_image_path,
    )
    status = ResponseStatus.TECHNICAL_FAILURE
    answer: str | None = None
    try:
        completion = recorder.complete(
            provider,
            request,
            role="answerer",
            material_id=material["material_id"],
            question_id=question_id,
            model=answerer_model,
        )
    except ProviderError:
        pass
    else:
        try:
            data = parse_json_object(completion.text)
        except OutputParseError:
            status = ResponseStatus.UNPARSEABLE
        else:
            candidate_status = data.get("status")
            candidate_answer = data.get("answer") if isinstance(data.get("answer"), str) else None
            if candidate_status == ResponseStatus.ANSWERED and candidate_answer and candidate_answer.strip():
                status = ResponseStatus.ANSWERED
                answer = candidate_answer.strip()
            elif candidate_status == ResponseStatus.REFUSED:
                status = ResponseStatus.REFUSED
            elif candidate_status == ResponseStatus.UNPARSEABLE:
                status = ResponseStatus.UNPARSEABLE
                answer = candidate_answer
            else:
                status = ResponseStatus.UNPARSEABLE
    input_manifest = {
        "text_included": request.text is not None,
        "image_included": request.image_path is not None,
        "evidence_included": False,
        "answer_included": False,
        "material_id_included": False,
        "text_sha256": sha256_text(request.text) if request.text is not None else None,
        "image_path": request.image_path,
        "question_sha256": sha256_text(question),
        "prompt_sha256": sha256_text(request.prompt),
    }
    return ResponseRecord(
        response_id=stable_id("response", run_id, question_id, condition),
        run_id=run_id,
        protocol_version=protocol_version,
        question_id=question_id,
        material_id=material["material_id"],
        method=method,
        condition=condition,
        status=status,
        answer=answer,
        answerer_model=answerer_model,
        input_manifest=input_manifest,
        created_at=utc_now(),
    )


_PUNCTUATION = " \t\r\n。．.，,；;：:！!？?、\"'“”‘’（）()[]【】"


def _normalize_answer(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(char for char in normalized if char not in _PUNCTUATION)


def score_response(
    *,
    response: Mapping[str, Any],
    answer: Mapping[str, Any] | None,
    run_id: str,
    protocol_version: str,
    scorer: str,
    question: str | None = None,
    provider=None,
    prompts: PromptSet | None = None,
    recorder: CallRecorder | None = None,
) -> ScoreRecord:
    correct: bool | None = None
    status = ScoreStatus.UNRESOLVED
    rationale = ""
    answer_id = answer["answer_id"] if answer else None

    if response["status"] != ResponseStatus.ANSWERED or not response.get("answer"):
        rationale = f"response_not_answered:{response['status']}"
    elif answer is None:
        rationale = "answer_record_missing"
    elif answer["question_id"] != response["question_id"]:
        rationale = "answer_question_mismatch"
    elif answer["status"] != AnswerStatus.VERIFIED or not answer.get("answer"):
        rationale = "answer_not_verified"
    elif provider is not None:
        if not question or prompts is None or recorder is None:
            raise ValueError("model scoring requires current question, prompts and call recorder")
        # Blind grading: no method, ablation condition or prior judgement in prompt.
        payload = {
            "question": question, "reference_answer": answer["answer"],
            "acceptable_answers": answer.get("acceptable_answers", []),
            "rubric": answer.get("rubric"), "response": response["answer"],
        }
        prompt = prompts.render("score_response", payload=json.dumps(payload, ensure_ascii=False))
        try:
            completion = recorder.complete(
                provider,
                CompletionRequest(task="score_response", prompt=prompt, question=question,
                                  condition=response["condition"],
                                  metadata={"protocol_version": protocol_version,
                                            "answer_id": answer["answer_id"]}),
                role="scorer", question_id=response["question_id"], model=scorer,
            )
            scorer = completion.model
            if completion.finish_reason != "stop":
                raise OutputParseError("judge_completion_not_finished")
            data = parse_json_object(completion.text)
            reason = data.get("rationale")
            if not isinstance(reason, str) or not reason.strip():
                raise OutputParseError("judge_rationale_missing")
            if data.get("status") == "scored" and type(data.get("correct")) is bool:
                correct, status, rationale = data["correct"], ScoreStatus.SCORED, reason
            elif data.get("status") == "unresolved" and data.get("correct") is None:
                rationale = reason
            else:
                raise OutputParseError("judge_schema_invalid")
        except (ProviderError, OutputParseError) as exc:
            rationale = f"model_scoring_unresolved:{exc}"
    else:
        if scorer != "normalized-exact-match-v1":
            raise ValueError("unknown local scorer; model scoring requires a provider")
        candidates = [answer["answer"], *answer.get("acceptable_answers", [])]
        normalized_response = _normalize_answer(response["answer"])
        normalized_candidates = {_normalize_answer(candidate) for candidate in candidates if candidate}
        if normalized_response in normalized_candidates:
            correct = True
            status = ScoreStatus.SCORED
            rationale = "normalized_exact_match"
        else:
            correct = False
            status = ScoreStatus.SCORED
            rationale = "no_acceptable_match"

    return ScoreRecord(
        score_id=stable_id("score", run_id, response["response_id"]),
        run_id=run_id,
        protocol_version=protocol_version,
        question_id=response["question_id"],
        response_id=response["response_id"],
        condition=response["condition"],
        status=status,
        correct=correct,
        rationale=rationale,
        answer_id=answer_id,
        scorer=scorer,
        created_at=utc_now(),
    )


def bdr_state(
    question: Mapping[str, Any] | None,
    validity: Mapping[str, Any] | None,
    scores_by_condition: Mapping[str, bool | None],
) -> tuple[str, str]:
    """Return a tri-state BDR decision and a reason for audit."""
    if question is None:
        return BDRState.UNRESOLVED, "question_pending"
    if question["status"] != QuestionStatus.OK:
        return BDRState.FAIL, f"no_valid_question:{question['status']}"
    validity_status = validity["status"] if validity else ValidityStatus.UNRESOLVED
    if validity_status == ValidityStatus.INVALID:
        return BDRState.FAIL, "validity_invalid"

    c_ti = scores_by_condition.get("TI")
    c_t = scores_by_condition.get("T")
    c_i = scores_by_condition.get("I")
    if c_ti is False:
        return BDRState.FAIL, "TI_incorrect"
    if c_t is True:
        return BDRState.FAIL, "T_correct"
    if c_i is True:
        return BDRState.FAIL, "I_correct"
    if validity_status == ValidityStatus.VALID and c_ti is True and c_t is False and c_i is False:
        return BDRState.PASS, "bimodal_dependency"
    if validity_status != ValidityStatus.VALID:
        return BDRState.UNRESOLVED, "validity_unresolved"
    return BDRState.UNRESOLVED, "condition_unresolved"
