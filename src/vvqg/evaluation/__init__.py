"""Active evaluation subsystem for the VQG line.

Layout
------
- ``metrics``  : one canonical implementation per active/diagnostic metric.
- ``common``   : shared I/O and bootstrap statistics.
- ``audit``    : independent reference implementations and audit utilities.

Historical specificity protocols and the deprecated absolute GPT dependence
rubric live under ``archive/evaluation_legacy`` and are not importable from this
public package. See ``docs/evaluation/EVALUATION_ACTIVE_PROTOCOL.md``.
"""

from vvqg.evaluation.metrics import *  # noqa: F401,F403
