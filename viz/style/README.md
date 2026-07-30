# viz_style

Centralized matplotlib theming and color palette, shared across every
plotting package in this repo ([`sensor`](../../clustering/sensor),
[`tracksim2d`](../../simulator/tracksim2d), [`graphs`](../../tracking/graphs),
[`hopfield_tracking`](../../tracking/hopfield_tracking),
[`multiplicity`](../../clustering/multiplicity)) -- one place to pick
between a "for a talk" and "for the page" look, and one place canonical
colors live instead of five independently hardcoded copies.

## Project layout

| module | contents |
|---|---|
| `viz_style.theme` | `Theme` (two booleans: `show_title`, `show_spatial_axes`), the `PRINT`/`PRESENT` instances, and `get_theme`/`set_theme`/`use_theme`. No matplotlib import. |
| `viz_style.palette` | Plain color/colormap-name string constants -- the canonical home for every hardcoded hex previously duplicated across packages. No matplotlib import, so it (and `theme`) stay importable without pulling matplotlib in, e.g. for a future non-matplotlib (SVG/PIL) consumer. |
| `viz_style.mpl` | `style_axes(ax, theme, *, spatial, ...)` and `save_or_show(fig, save_path, theme)` -- the only matplotlib-touching module. |

## Setup

```bash
cd viz/style
python3 -m venv .venv
.venv/bin/pip install -e . -r requirements.txt
```

## Using it from another package

```bash
# from e.g. clustering/sensor/, alongside its own -e .
.venv/bin/pip install -e ../../viz/style -e ../utils -e . -r requirements.txt
```

## Themes

Two named presets:

- **`PRINT`** (the default returned by `get_theme()`): reproduces every
  consuming package's original, pre-`viz_style` look exactly -- titles,
  axis labels, ticks, spines, and legends all shown.
- **`PRESENT`**: no titles anywhere; and, specifically for
  *spatial/geometric* displays (detector layers, hit positions -- anything
  that calls `style_axes(..., spatial=True)`), no ticks, spines, axis
  labels, or legend either. *Statistical/analysis* plots (ROC curves,
  confusion matrices, histograms -- `spatial=False`) keep their axes,
  labels, and legend in both themes; only their title is theme-gated,
  since a plot like a ROC curve is unreadable without a labeled axis.

Every `plot_*` function across the consuming packages takes an optional
trailing `theme: Theme | None = None` argument, resolved via `theme =
theme or get_theme()`. Pass it explicitly (this is what every CLI's
`--style {print,present}` flag does) or set a session default:

```python
from viz_style import PRESENT, use_theme

with use_theme(PRESENT):
    fig = plot_event(...)   # picks up PRESENT without an explicit theme= kwarg
```

`set_theme(theme)` sets the default for the rest of the process; CLIs never
rely on this (they always pass `theme=` explicitly) so behavior never
depends on state left over from an earlier call.

## Palette

`viz_style.palette` holds every color/colormap constant migrated out of
the five consuming packages -- `CATEGORICAL_OKABE_ITO` (the Okabe-Ito
colorblind-safe 6-hue cycle, previously `tracksim2d.DEFAULT_TRACK_COLORS`),
plus named semantic colors (`TRUTH`, `CLUSTER`, `EDGE`, `HIT`, `VERTEX`,
...) and two sequential colormap names. Every value is a verbatim lift of
a pre-existing hardcoded hex -- migrating a module to import from here
changes zero pixels under `PRINT`.

Reconciling *different* hexes that play the same semantic role (e.g.
`TRUTH` vs. `VERTEX`, or `CLASS_COLORS` vs. `CATEGORICAL_OKABE_ITO`) into
one truly canonical set is a real visual change and hasn't been done here
-- a separately-scoped follow-up.

## Tests

```bash
.venv/bin/python -m pytest tests/
```

Covers `Theme`/`PRINT`/`PRESENT` field values and `get_theme`/`set_theme`/
`use_theme` (including exception-safety and nesting); `style_axes` under
all combinations of `spatial`×theme (the statistical-plot carve-out in
particular); `save_or_show`; and that `viz_style`/`viz_style.palette` never
import matplotlib.
