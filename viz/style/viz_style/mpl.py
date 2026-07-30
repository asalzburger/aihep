"""The one matplotlib-touching module: apply a `Theme` to an `Axes`, and
the shared save-or-show convention. `theme`/`palette` stay matplotlib-free
so importing them never pulls matplotlib in; only importing this module
does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from .theme import Theme, get_theme

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def style_axes(
    ax: "Axes",
    theme: Theme | None = None,
    *,
    spatial: bool,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = False,
    legend_handles: Sequence | None = None,
    legend_loc: str = "best",
) -> None:
    """Apply `theme`'s title/axis chrome to `ax`.

    `spatial=True` marks a detector/sensor/hit-point geometric display:
    when `theme.show_spatial_axes` is False, ticks, spines, axis labels,
    and the legend are all dropped together (only the title, gated
    separately, may still have been suppressed above). `spatial=False`
    marks a statistical/analysis plot (ROC curve, confusion matrix,
    histogram): its axes/labels/legend are never suppressed, regardless of
    theme -- only the title is theme-gated.

    `legend_loc` matters for pixel-parity with each caller's pre-existing
    placement (e.g. "upper right" vs. "lower right") -- pass it explicitly
    rather than relying on the "best" default when a caller already had one.
    """
    theme = theme or get_theme()
    if title and theme.show_title:
        ax.set_title(title)
    if spatial and not theme.show_spatial_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if legend:
        if legend_handles:
            ax.legend(handles=legend_handles, loc=legend_loc, frameon=True, fontsize=8)
        else:
            ax.legend(loc=legend_loc, frameon=True, fontsize=8)


def save_or_show(fig: "Figure", save_path=None, theme: Theme | None = None):
    """`fig.savefig(path, **theme.savefig_kwargs)` if `save_path` is given,
    else `plt.show()` -- the `dpi=150, bbox_inches="tight"` convention
    every CLI already applies, extracted once. Optional: existing CLIs may
    keep their own inline save/show code instead of adopting this."""
    theme = theme or get_theme()
    if save_path:
        fig.savefig(save_path, **theme.savefig_kwargs)
        return save_path
    import matplotlib.pyplot as plt

    plt.show()
    return None
