"""vvqg — reward-driven Visual Question Generation (Image -> Question).

Frozen task: `Image -> Question`, base model Qwen3-8B (`~/models/Qwen3-8B`).
This package holds the Phase-0 scaffold: data schemas, reward interface +
placeholders, training entry points, evaluation, and model utilities.
"""

__all__ = ["data", "rewards", "training", "evaluation", "models"]
