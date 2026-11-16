"""Detail capture: teacher-forced positions are the last prompt token followed by every stored body token."""

import pytest

from egvqg.detail_capture import detail_positions


def test_detail_positions_start_at_last_prompt_token():
    assert detail_positions(5, 3) == [4, 5, 6, 7]


def test_detail_positions_reject_empty_parts():
    with pytest.raises(ValueError):
        detail_positions(0, 3)
    with pytest.raises(ValueError):
        detail_positions(5, 0)
