"""Faithful same-backbone SeeQ reproduction (multi-intent proposal + difficulty filter)."""
from .pipeline import (
    FaithfulSeeQConfig,
    FaithfulSeeQPipeline,
    candidate_key,
    proposal_prompt,
    resolve_round_models,
    schedule,
)
from .filtering import decide, has_difficulty_improved
from .training import dual_format_records, faithful_messages
from .prompts import INTENTS, INTENT_NAMES, SEEQ_COMMIT

__all__ = [
    "FaithfulSeeQConfig", "FaithfulSeeQPipeline", "candidate_key", "proposal_prompt",
    "resolve_round_models", "schedule", "decide", "has_difficulty_improved", "dual_format_records",
    "faithful_messages", "INTENTS", "INTENT_NAMES", "SEEQ_COMMIT",
]
