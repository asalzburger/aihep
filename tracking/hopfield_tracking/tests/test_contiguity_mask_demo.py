import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from contiguity_mask_demo import found_by_mask, max_row_drift, track_rows  # noqa: E402


def test_straight_track_has_zero_drift():
    # an "infinite radius" straight track (approximated here directly) never drifts
    assert max_row_drift([10] * 20) == 0
    assert found_by_mask([10] * 20)


def test_large_radius_track_is_found_by_the_mask():
    rows = track_rows(radius=60.0)
    assert max_row_drift(rows) <= 1
    assert found_by_mask(rows)


def test_small_radius_track_is_missed_by_the_mask():
    rows = track_rows(radius=15.0)
    assert max_row_drift(rows) > 1
    assert not found_by_mask(rows)


def test_smaller_radius_drifts_more_than_larger_radius():
    assert max_row_drift(track_rows(radius=15.0)) > max_row_drift(track_rows(radius=60.0))
