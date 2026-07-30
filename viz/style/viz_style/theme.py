"""Presentation theme: two orthogonal on/off knobs, no matplotlib import.

`show_title` gates title/suptitle text everywhere. `show_spatial_axes`
gates ticks/spines/axis-labels/legend together, but only for plots that
identify themselves as spatial/geometric displays (detector layers, hit
positions) when they call `viz_style.mpl.style_axes` -- statistical plots
(ROC curves, confusion matrices, histograms) never consult it, since their
axes are the content, not chrome.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Mapping


@dataclass(frozen=True)
class Theme:
    name: str
    show_title: bool = True
    show_spatial_axes: bool = True
    savefig_kwargs: Mapping[str, object] = field(
        default_factory=lambda: {"dpi": 150, "bbox_inches": "tight"}
    )


#: Matches every plotting function's pre-existing default behavior exactly.
PRINT = Theme(name="print")

#: No titles; spatial/geometric displays (not statistical plots) lose their
#: ticks, spines, axis labels, and legend too -- for dropping a figure
#: straight into a slide.
PRESENT = Theme(name="present", show_title=False, show_spatial_axes=False)

_current: Theme = PRINT


def get_theme() -> Theme:
    """The current default theme, used by any `plot_*` call that omits its
    own `theme=` argument."""
    return _current


def set_theme(theme: Theme) -> None:
    """Set the default theme for subsequent calls that omit `theme=`.

    CLIs should not rely on this -- they resolve `--style` to a `Theme` and
    pass it explicitly on every call, so CLI output never depends on
    leftover global state from an earlier call in the same process. This
    is for notebook/script convenience instead.
    """
    global _current
    _current = theme


@contextmanager
def use_theme(theme: Theme) -> Iterator[Theme]:
    """Context manager: set `theme` as current for the block, restoring the
    previous theme afterward even if the block raises."""
    previous = get_theme()
    set_theme(theme)
    try:
        yield theme
    finally:
        set_theme(previous)
