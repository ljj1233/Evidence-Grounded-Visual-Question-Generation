"""Shared JSONL record types; schemas describe records without runtime wrapping."""
from typing import TypedDict


class ReferenceRecord(TypedDict):
    image_id: str
    question: str


class PredictionRecord(ReferenceRecord, total=False):
    system_id: str
    image: str
    image_path: str
    generation_id: int
