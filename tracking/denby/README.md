# Denby detector recreation

A first concrete application of [`detector2d`](../../detector/detector2d) /
[`tracksim2d`](../../simulator/tracksim2d): harmonize the detector drawn in
`resources/denby_detector.svg` (13 slightly irregular detection planes),
infer the 4-track event in `resources/denby_detector_event.svg` back into
particle parameters, and re-simulate it through the harmonized detector so
the result overlays the original almost exactly.

Everything stays in the reference figures' own SVG pixel coordinate system
(y grows downward, origin top-left, `viewBox="0 0 924 1074"`) so the
simulated output is directly overlayable with no rescaling.

## What's in the reference SVGs

`denby_detector.svg` and `denby_detector_event.svg` (the latter is the
former plus one event) both contain:

- **13 red dashed horizontal `<path>` elements** spanning the full detector
  width -- these are the actual detection planes (`LineLayer`s). They
  already share one length (709.521 units) but vary in left edge x
  (136.6-157.1) and vertical spacing (53-60 units, mean 58.32).
- **4 curved black "arm" paths with small circles along them** -- decorative
  apparatus art (drift-tube rows) from the original scanned figure, *not*
  the 13 layers. (`denby_overlay.svg` shows our re-simulated hits landing
  almost exactly on these circles too, which is a nice independent sanity
  check that the geometry is self-consistent, even though they aren't used
  as inputs to anything here.)
- (event file only) **4 blue dashed track curves** and **one small green
  dot**.

## Finding the point source

The green dot looks like decoration, but it isn't. Each of the 4 blue
tracks is a genuine circular arc -- fitting a circle (least squares through
densely-sampled points of each track's cubic Bezier) gives an essentially
exact fit, rms residual <= 0.06px:

| track | center | radius |
|---|---|---|
| 0 | (-5860, 1967) | 6375 |
| 1 | (2216, 1585) | 1920 |
| 2 | (988, 595) | 640 |
| 3 | (-1729, 369) | 2214 |

Computing the pairwise circle-circle intersection for all `C(4,2) = 6` pairs
of these fitted circles, every pair agrees on one point: **(~424, ~895)**.
That's exactly the green dot's center (fit the same way: sample its path,
fit a circle, keep only the center). Two independent methods -- "where the
marker is" and "where the tracks' own circles all cross" -- agree to a
couple of pixels, which is strong evidence the 4 tracks really do share one
common vertex there: the point source. `denby_svg.derive_vertex` computes
both and raises if they disagree by more than a few pixels.

(Practical implication of the vertex sitting at y~895, i.e. *below* the
layer stack at y=70-770: the tracks fly upward through the detector, not
downward.)

## Pipeline

| script | reads | writes |
|---|---|---|
| `python/harmonize_detector.py` | `denby_detector.svg` (13 layers), `denby_detector_event.svg` (vertex) | `denby_layers.csv`, `denby_detector_harmonized.svg` |
| `python/fit_event.py` | `denby_detector_event.svg` (4 tracks + vertex) | `denby_event.csv` |
| `python/render_event.py` | `denby_layers.csv`, `denby_event.csv`, `denby_detector_event.svg` (for the overlay background) | `denby_event_simulated.svg`, `denby_overlay.svg` |

`python/denby_svg.py` holds the shared SVG-parsing/circle-fitting helpers
used by all three (`parse_layer_lines`, `parse_event_tracks`,
`fit_circle_to_svg_path`, `derive_vertex`, `extract_svg_body`).

### 1. Harmonize the detector

`harmonize_detector.harmonized_layers()` re-derives the spacing/length
numbers above straight from the SVG (nothing hardcoded) and returns 13
identical, equidistant `LineLayer`s: same x-extent (709.521 units, the
length they already shared), spaced exactly `(769.864-70)/12 = 58.322`
units apart from y=70 to y=769.864. The point source is marked with a small
circle at the derived vertex.

### 2. Fit the event

For each of the 4 tracks, `fit_event.fit_track` recovers
`(x0, y0, phi0, charge, radius)` in `tracksim2d.edm.PARTICLES_COLUMNS`
form:

- `(x0, y0)` = the derived vertex (shared by all 4 rows)
- `radius` = the fitted signed radius. Of the two tangent directions
  possible at the vertex (one per curl sign), the code picks whichever one's
  *forward* sweep reaches the track's farthest-from-vertex sampled point the
  short way around -- the only sensible choice for a track drawn as one
  non-looping arc.
- `phi0` = the initial heading implied by that choice.
- `charge` = **+1 for a left/CCW curl (radius > 0), -1 for right/CW** -- a
  *convention*, not a measurement. "Assume a constant field, stay in SVG
  coordinates" means storing the already-resolved radius directly rather
  than inventing a pt/Bz decomposition: flipping the assumed field direction
  would flip every charge and leave the picture (and this CSV, up to the
  sign column) identical.

Result: `resources/denby_event.csv`, 4 rows, one `event_id`.

### 3. Re-simulate and render

`render_event.py` loads both CSVs, calls
`tracksim2d.simulate.hits_for_particles` (the exact same function any other
`tracksim2d` consumer would use) to get hits on the 13 harmonized layers,
then `tracksim2d.vis.export_svg` twice: once alone
(`denby_event_simulated.svg`), once layered on top of the original
reference figure at reduced opacity (`denby_overlay.svg`) for a one-glance
visual diff.

## How good is the match?

`tests/test_denby.py::test_simulated_hits_match_the_reference_tracks_within_a_few_pixels`
is the objective version of "overlaid, it should match up pretty nicely":
for each track, it compares the simulated hit's x at a given y against the
*original* reference curve's x at that same y (by interpolating the
original Bezier). All 4 tracks stay under 5px of residual on a ~1000px-wide
figure (typically under 3px, sub-pixel for two of them) -- despite the
harmonized layer y-positions being slightly different from the original's.
`resources/denby_overlay.svg` shows this visually: the simulated (colored,
solid) tracks trace almost exactly over the original (light blue, dashed)
ones.

## Running it

```bash
cd tracking/denby
python3 -m venv .venv
.venv/bin/pip install -e ../../detector/detector2d -e ../../simulator/tracksim2d -r requirements.txt

cd python
../.venv/bin/python harmonize_detector.py
../.venv/bin/python fit_event.py
../.venv/bin/python render_event.py
```

## Tests

```bash
.venv/bin/python -m pytest tests/
```

Covers: the original detector really is 13 same-length-but-unevenly-spaced
layers (so harmonization is doing something); the harmonized detector is 13
equidistant, equal-length, equal-extent layers; the derived vertex matches
both the marker dot and the cross-check; the fitted event has one row per
track sharing a vertex; `denby_event.csv` round-trips through
`tracksim2d.io`; and the quantitative pixel-residual match check above.
