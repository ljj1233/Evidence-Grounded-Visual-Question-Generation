"""Legacy module path for the frozen answer-likelihood scorer only.

The old counterfactual dependence interface is archived and is not an active
reward or verifier.
"""

from vvqg.rewards.dependence.answer_likelihood import PROMPT, TeacherForcedScorer, answer_logprobs

__all__ = ["PROMPT", "TeacherForcedScorer", "answer_logprobs"]
