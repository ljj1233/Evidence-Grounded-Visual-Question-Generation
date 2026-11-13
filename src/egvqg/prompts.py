"""Prompt templates loaded from a directory and snapshotted into each run."""

from __future__ import annotations

from pathlib import Path
from string import Template

from .records import canonical_json, sha256_text

PROMPT_NAMES = (
    "qg_plain",
    "qg_strong",
    "qg_evidence",
    "extract_text_evidence",
    "extract_image_evidence",
    "pair_evidence",
    "annotate_answer",
    "verify_answer",
    "validate_question",
    "answer_question",
)


class PromptError(RuntimeError):
    pass


class PromptSet:
    def __init__(self, directory: Path, *, names: tuple[str, ...] = PROMPT_NAMES):
        self.directory = directory
        self._texts: dict[str, str] = {}
        for name in names:
            path = directory / f"{name}.txt"
            if not path.exists():
                raise FileNotFoundError(f"missing prompt template: {path}")
            self._texts[name] = path.read_text(encoding="utf-8")

    def render(self, name: str, **values: str) -> str:
        try:
            template = Template(self._texts[name])
        except KeyError as exc:
            raise PromptError(f"unknown prompt template: {name}") from exc
        try:
            return template.substitute(**values)
        except KeyError as exc:
            raise PromptError(f"prompt {name!r} is missing value {exc.args[0]!r}") from exc

    def variables(self, name: str) -> tuple[str, ...]:
        return tuple(Template(self._texts[name]).get_identifiers())

    def snapshot(self) -> dict[str, str]:
        return dict(self._texts)

    def snapshot_sha256(self) -> str:
        return sha256_text(canonical_json(self._texts))
