"""Canonical color/colormap constants -- plain strings, no matplotlib
import, so this module stays importable (e.g. for SVG rendering) without
pulling matplotlib in.

Every value here is a verbatim lift of a pre-existing hardcoded hex from
one of the packages this repo's plotting code lives in -- this pass
centralizes *where* the colors live, not what they are, so migrating a
module to import from here changes zero pixels. The one exception is `ON`,
which used to be an independently hardcoded duplicate of
`CATEGORICAL_OKABE_ITO[0]` (same hex, reinvented) and is now a real alias.

Reconciling genuinely different hexes that play the same semantic role
(e.g. `TRUTH` vs. `VERTEX`, or `CLASS_COLORS` vs. `CATEGORICAL_OKABE_ITO`)
into one truly canonical set is a real visual change and is deliberately
not done here -- a separately-scoped follow-up.
"""

from __future__ import annotations

#: Okabe-Ito colorblind-safe categorical palette (Okabe & Ito, 2008), in
#: tracksim2d's pre-existing cycling order -- the repo's only prior
#: categorical cycle, now canonicalized.
CATEGORICAL_OKABE_ITO = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
)

TRUTH = "#2E86DE"              # was sensor.vis.TRUTH_COLOR
CLUSTER = "#DA4C4C"            # was sensor.vis.CLUSTER_COLOR
CLUSTER_DIGITAL = "#E67E22"    # was sensor.vis.CENTROID_TYPE_COLOR["digital"]
DIGITAL_ON = "#C0392B"         # was sensor.vis.DIGITAL_ON_COLOR
GRID = "0.85"                  # was sensor.vis.GRID_COLOR
LAYER = "#999999"              # was tracksim2d.vis.LAYER_COLOR
HIT = "#000000"                # was tracksim2d.vis.HIT_COLOR / hopfield_tracking.vis.HIT_COLOR
VERTEX = "#009E73"             # was tracksim2d.vis.VERTEX_COLOR
EDGE = "#333333"               # was graphs.vis.EDGE_COLOR
TRUE_EDGE = "#7570B3"          # was graphs.vis.TRUE_EDGE_COLOR
ON = CATEGORICAL_OKABE_ITO[0]  # was hopfield_tracking.vis.ON_COLOR, an
                                # independently hardcoded "#0072B2" -- now a
                                # real alias instead of an accidental dup.

#: Deliberately its own 5-hue set (not the categorical cycle above) -- see
#: the module docstring's note on deferred consolidation.
CLASS_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")

SEQUENTIAL_CHARGE_CMAP = "YlOrRd"      # was sensor.vis.CHARGE_CMAP
SEQUENTIAL_CONFUSION_CMAP = "Blues"    # was multiplicity.evaluate's inline "Blues"
