"""Joint Soft-Evidence QSE v2 exports."""

from vvqg.evidence.assignment import FrozenQuestionEncoder, assignment_loss, hungarian_match, slot_diversity_loss
from vvqg.evidence.model import SoftEvidenceQwen
from vvqg.evidence.model_v2 import SoftEvidenceQwenV2

__all__ = [
    "SoftEvidenceQwen",
    "SoftEvidenceQwenV2",
    "FrozenQuestionEncoder",
    "assignment_loss",
    "hungarian_match",
    "slot_diversity_loss",
]
