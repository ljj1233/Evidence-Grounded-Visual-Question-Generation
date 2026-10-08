"""Counterfactual Evidence Regularization (CER) research pilot.

Supervised, training-time visual-evidence constraint on top of a frozen Base
Full Ours checkpoint.  CER maximizes token-normalized anchor log-probability on
the original image relative to its deterministic shuffled counterfactual, while
staying KL-anchored to the base policy.  No new external reward API, no GRPO.
"""
