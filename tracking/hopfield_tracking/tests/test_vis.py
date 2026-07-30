import numpy as np
import pandas as pd
from viz_style import PRESENT, PRINT

from hopfield_tracking.cli import run
from hopfield_tracking.vis import plot_iterations


def _segments_and_history():
    hits = pd.DataFrame({"particle_id": [0, 0], "x": [0.0, 0.0], "y": [0.0, 50.0]})
    segments, history, _chains, _score = run(hits, r_c=60.0, r_scale=50.0, seed=0)
    xy = hits[["x", "y"]].to_numpy()
    return xy, segments, history


def test_plot_iterations_print_keeps_title():
    xy, segments, history = _segments_and_history()
    fig = plot_iterations(xy, segments, history, theme=PRINT)
    ax = fig.axes[0]
    assert ax.get_title() != ""


def test_plot_iterations_present_drops_title_and_spines():
    xy, segments, history = _segments_and_history()
    fig = plot_iterations(xy, segments, history, theme=PRESENT)
    ax = fig.axes[0]
    assert ax.get_title() == ""
    assert all(not spine.get_visible() for spine in ax.spines.values())
    # ticks were already unconditionally off before viz_style, in both themes
    assert list(ax.get_xticks()) == []
    assert list(ax.get_yticks()) == []


def test_plot_iterations_default_theme_matches_print():
    xy, segments, history = _segments_and_history()
    fig = plot_iterations(xy, segments, history)
    assert fig.axes[0].get_title() != ""
