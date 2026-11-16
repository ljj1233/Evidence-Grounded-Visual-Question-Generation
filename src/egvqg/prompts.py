"""Prompt templates loaded from a directory and snapshotted into each run."""

from __future__ import annotations

from pathlib import Path
from string import Template

from .records import canonical_json, sha256_text

PROMPT_NAMES = ("plain-v2",)


class PromptError(RuntimeError):
    pass


def prompt_path(directory: Path, name: str) -> Path:
    """Locate `<name>.txt` in the prompt directory or one of its role subdirectories."""
    direct = directory / f"{name}.txt"
    if direct.exists():
        return direct
    for sub in sorted(p for p in directory.iterdir() if p.is_dir()):
        candidate = sub / f"{name}.txt"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing prompt template: {direct} (no subdirectory match)")


class PromptSet:
    def __init__(self, directory: Path, *, names: tuple[str, ...] = PROMPT_NAMES):
        self.directory = directory
        self._texts: dict[str, str] = {}
        for name in names:
            path = prompt_path(directory, name)
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
