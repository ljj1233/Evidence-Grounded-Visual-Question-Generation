import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vvqg.training.online_uvfc_relative_grpo import DEGEN_STREAK, _degenerate  # noqa: E402

THRESHOLDS = {"t_malformed": 0.02, "t_length": 10.0, "t_dup": 0.45}


def row(**overrides):
    base = {"malformed_rate": 0.0, "mean_question_tokens": 8.0, "duplicate_rate": 0.2}
    base.update(overrides)
    return base


def trips(recent):
    return len(recent) >= DEGEN_STREAK and all(_degenerate(r, THRESHOLDS) for r in recent[-DEGEN_STREAK:])


def test_single_spike_does_not_trip():
    assert not trips([row(), row(), row(duplicate_rate=0.5)])


def test_two_consecutive_do_not_trip():
    assert not trips([row(), row(), row(duplicate_rate=0.5), row(duplicate_rate=0.5)])


def test_three_consecutive_trip():
    assert trips([row(), row(duplicate_rate=0.5), row(duplicate_rate=0.5), row(duplicate_rate=0.5)])
