from __future__ import annotations

import pytest

from viz_style.theme import PRESENT, PRINT, Theme, get_theme, set_theme, use_theme


def test_print_matches_pre_existing_default_behavior():
    assert PRINT.show_title is True
    assert PRINT.show_spatial_axes is True


def test_present_drops_titles_and_spatial_axes():
    assert PRESENT.show_title is False
    assert PRESENT.show_spatial_axes is False


def test_default_theme_is_print():
    assert get_theme() is PRINT


def test_set_theme_round_trips():
    set_theme(PRESENT)
    try:
        assert get_theme() is PRESENT
    finally:
        set_theme(PRINT)


def test_use_theme_restores_previous_on_normal_exit():
    assert get_theme() is PRINT
    with use_theme(PRESENT):
        assert get_theme() is PRESENT
    assert get_theme() is PRINT


def test_use_theme_restores_previous_on_exception():
    assert get_theme() is PRINT
    with pytest.raises(ValueError):
        with use_theme(PRESENT):
            assert get_theme() is PRESENT
            raise ValueError("boom")
    assert get_theme() is PRINT


def test_use_theme_nests():
    other = Theme(name="other", show_title=False)
    with use_theme(PRESENT):
        with use_theme(other):
            assert get_theme() is other
        assert get_theme() is PRESENT
    assert get_theme() is PRINT
