from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from viz_style.mpl import save_or_show, style_axes
from viz_style.theme import PRESENT, PRINT


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def test_print_spatial_keeps_everything(ax):
    style_axes(ax, PRINT, spatial=True, title="t", xlabel="x", ylabel="y", legend=True)
    assert ax.get_title() == "t"
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"
    assert all(spine.get_visible() for spine in ax.spines.values())
    assert ax.get_legend() is not None


def test_present_spatial_strips_everything(ax):
    style_axes(ax, PRESENT, spatial=True, title="t", xlabel="x", ylabel="y", legend=True)
    assert ax.get_title() == ""
    assert list(ax.get_xticks()) == []
    assert list(ax.get_yticks()) == []
    assert all(not spine.get_visible() for spine in ax.spines.values())
    # no xlabel/ylabel/legend calls were made once spatial axes are hidden
    assert ax.get_xlabel() == ""
    assert ax.get_ylabel() == ""
    assert ax.get_legend() is None


def test_present_statistical_keeps_axes_labels_and_legend(ax):
    ax.plot([0, 1], [0, 1], label="series")
    style_axes(ax, PRESENT, spatial=False, title="t", xlabel="x", ylabel="y", legend=True)
    # statistical plots (spatial=False) never lose axes/labels/legend
    assert ax.get_title() == ""
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"
    assert all(spine.get_visible() for spine in ax.spines.values())
    assert ax.get_legend() is not None


def test_print_statistical_keeps_title(ax):
    style_axes(ax, PRINT, spatial=False, title="t", xlabel="x", ylabel="y")
    assert ax.get_title() == "t"


def test_style_axes_defaults_to_current_theme(ax, monkeypatch):
    import viz_style.theme as theme_mod

    monkeypatch.setattr(theme_mod, "_current", PRESENT)
    style_axes(ax, spatial=True, title="t")
    assert ax.get_title() == ""


def test_save_or_show_writes_file(tmp_path):
    fig, ax = plt.subplots()
    path = tmp_path / "out.png"
    result = save_or_show(fig, save_path=path, theme=PRINT)
    plt.close(fig)
    assert result == path
    assert path.exists()
