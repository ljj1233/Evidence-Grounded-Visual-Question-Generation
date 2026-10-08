"""Requirement-level view of the frozen Flat verifier (Phase 5B selectors).

The frozen verifier already returns per-requirement labels
(``presuppositions`` / ``reference_constraints`` / ``query_target``); the online
wrapper discards them and keeps only the overall state.  This subclass keeps the
full parse so a selector can use the weakest visual requirement.  Prompt,
checkpoint, decoding and parsing are the frozen Phase-2.1 implementation.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from vvqg.rewards.support.frozen_verifier import parse_prediction
from vvqg.rewards.support.online_flat import STATES, OnlineFlatVerifier

REQUIREMENT_SCORE = {
    "supported": 1.0, "clear": 1.0, "answerable": 1.0,
    "uncertain": 0.5, "ambiguous": 0.5,
    "unsupported": 0.0, "unanswerable": 0.0,
}


def requirement_labels(parsed: dict[str, Any]) -> list[str]:
    structure = parsed.get("structure") or {}
    labels = [node["label"] for node in structure.get("presuppositions", []) if "label" in node]
    labels += [node["label"] for node in structure.get("reference_constraints", []) if "label" in node]
    target = structure.get("query_target")
    if isinstance(target, dict) and "label" in target:
        labels.append(target["label"])
    return labels


def support_summary(parsed: dict[str, Any]) -> dict[str, Any]:
    if parsed.get("status") != "ok":
        return {"flat_state": "FAILED", "support_floor": -1.0, "support_mean": -1.0, "n_requirements": 0, "requirement_labels": []}
    labels = requirement_labels(parsed)
    values = [REQUIREMENT_SCORE[label] for label in labels if label in REQUIREMENT_SCORE]
    state = STATES.get(parsed.get("overall_visual_support"), "FAILED")
    if not values:
        return {"flat_state": state, "support_floor": 0.0, "support_mean": 0.0, "n_requirements": 0, "requirement_labels": labels}
    return {
        "flat_state": state,
        "support_floor": min(values),
        "support_mean": sum(values) / len(values),
        "n_requirements": len(values),
        "requirement_labels": labels,
    }


class StructuredFlatVerifier(OnlineFlatVerifier):
    @torch.inference_mode()
    def verify_structured(self, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        parsed_rows: list[dict[str, Any]] = []
        for start in range(0, len(rows), self.batch_size):
            chunk = rows[start : start + self.batch_size]
            texts = [
                self.processor.apply_chat_template(
                    [{"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.prompt + "\nQuestion: " + row["question"]},
                    ]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for row in chunk
            ]
            inputs = self.processor(
                text=texts, images=[self._image(row["image"]) for row in chunk], padding=True, return_tensors="pt"
            ).to(self.model.device)
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            raws = self.processor.tokenizer.batch_decode(
                generated[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
            )
            parsed_rows.extend(parse_prediction(raw, "flat") for raw in raws)
            print(f"flat {min(start + self.batch_size, len(rows))}/{len(rows)}", flush=True)
        return parsed_rows
