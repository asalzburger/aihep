"""The fig.-8-style multi-panel plot: measured points as crosses, "on"
neurons as segments with a small circle at the head indicating direction,
energy/iteration/elapsed-time annotated per panel -- reproducing the
paper's own figure caption almost verbatim.
"""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from .dynamics import RelaxationHistory, DEFAULT_DT_OVER_TAU
from .extract import ON_THRESHOLD, on_segments
from .network import Segment

ON_COLOR = "#0072B2"
HIT_COLOR = "#000000"


def _panel_iterations(n_available: int, n_panels: int = 4) -> list[int]:
    """Evenly spaced iteration indices, always including the first and
    last, like the paper's 4-panel figures."""
    last = n_available - 1
    if last < n_panels - 1:
        return list(range(n_available))
    return sorted({round(i * last / (n_panels - 1)) for i in range(n_panels)})


def _grid_shape(n: int) -> tuple[int, int]:
    """As close to square as possible, e.g. 4 -> 2x2 (the paper's own fig. 8
    layout), 3 -> 2x2 (with one empty slot), 6 -> 2x3."""
    nrows = int(np.ceil(np.sqrt(n)))
    ncols = int(np.ceil(n / nrows))
    return nrows, ncols


def plot_iterations(
    hits_xy: np.ndarray,
    segments: list[Segment],
    history: RelaxationHistory,
    iterations: list[int] | None = None,
    dt_over_tau: float = DEFAULT_DT_OVER_TAU,
    threshold: float = ON_THRESHOLD,
    layout: Literal["grid", "row"] = "grid",
    invert_y: bool = True,
):
    """`layout="grid"` (default) tiles panels as close to a square as
    possible -- 2x2 for the usual 4 panels, matching fig. 8's own layout in
    the paper; `layout="row"` is a single row instead. `invert_y` (default
    True) flips the y-axis so the vertex (the larger-y end of our
    coordinates, following the SVG/image convention of y growing downward)
    renders at the *bottom* with tracks fanning upward, matching the
    paper's own fig. 8 orientation -- matplotlib's default is the opposite
    (y increasing upward), which without this looks upside down next to
    the paper.
    """
    if iterations is None:
        iterations = _panel_iterations(len(history))

    if layout == "grid":
        nrows, ncols = _grid_shape(len(iterations))
    elif layout == "row":
        nrows, ncols = 1, len(iterations)
    else:
        raise ValueError(f"Unknown layout: {layout!r}")

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 4.4 * nrows), squeeze=False)
    flat_axes = axes.flatten(order="C")  # row-major: top-left, top-right, ..., matching the paper's panel order

    for ax, it in zip(flat_axes, iterations):
        ax.scatter(hits_xy[:, 0], hits_xy[:, 1], marker="+", color=HIT_COLOR, zorder=3)
        for seg in on_segments(segments, history.f_v[it], threshold):
            x0, y0 = seg.start_xy
            x1, y1 = seg.end_xy
            ax.plot([x0, x1], [y0, y1], color=ON_COLOR, linewidth=1.2, zorder=2)
            # small open circle at the head, indicating direction (paper's fig. 8 convention)
            ax.plot(x1, y1, marker="o", markerfacecolor="none", markeredgecolor=ON_COLOR, markersize=5, zorder=2)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if invert_y:
            ax.invert_yaxis()
        elapsed = it * dt_over_tau
        ax.set_title(f"Energy  {history.energy[it]:.4f}\nIteration  {it}   T = {elapsed:.1f}τ", fontsize=9)

    for ax in flat_axes[len(iterations) :]:
        ax.axis("off")

    fig.tight_layout()
    return fig
