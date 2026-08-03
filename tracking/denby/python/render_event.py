"""Re-simulate the Denby event through the harmonized detector and render it.

Reads resources/denby_layers.csv + resources/denby_event.csv (written by
harmonize_detector.py / fit_event.py), computes hits via
detector2d/detectorsim2d exactly like any other detectorsim2d event, and writes:

- resources/denby_event_simulated.svg -- the harmonized detector + simulated
  event, alone, in the same coordinate system/viewBox as the reference
  figures (directly overlayable, no rescaling).
- resources/denby_overlay.svg -- the same thing, drawn on top of the
  original reference figure (dimmed), for a one-glance visual diff.

Run from this directory (`tracking/denby/python/`) with the project venv:

    ../.venv/bin/python render_event.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from denby_svg import extract_svg_body

from detector2d.geometry import LineLayer
from detectorsim2d.simulate import hits_for_particles
from detectorsim2d.vis import export_svg

RESOURCES = Path(__file__).resolve().parent.parent / "resources"
LAYERS_CSV = RESOURCES / "denby_layers.csv"
EVENT_CSV = RESOURCES / "denby_event.csv"
REFERENCE_EVENT_SVG = RESOURCES / "denby_detector_event.svg"
SIMULATED_SVG = RESOURCES / "denby_event_simulated.svg"
OVERLAY_SVG = RESOURCES / "denby_overlay.svg"

VIEWBOX_WIDTH = 924.0
VIEWBOX_HEIGHT = 1074.0


def load_layers(path: str = LAYERS_CSV) -> list[LineLayer]:
    df = pd.read_csv(path)
    return [
        LineLayer(layer_id=int(row.layer_id), p1=(row.x1, row.y1), p2=(row.x2, row.y2), pitch=row.pitch)
        for row in df.itertuples()
    ]


def main() -> None:
    layers = load_layers()
    particles = pd.read_csv(EVENT_CSV)
    hits = hits_for_particles(particles, layers)

    export_svg(layers, particles, hits, SIMULATED_SVG, width=VIEWBOX_WIDTH, height=VIEWBOX_HEIGHT)
    print(f"{len(hits)} hits across {len(layers)} layers for {len(particles)} particles")
    print(f"wrote {SIMULATED_SVG}")

    reference_body = extract_svg_body(open(REFERENCE_EVENT_SVG).read())
    export_svg(
        layers,
        particles,
        hits,
        OVERLAY_SVG,
        width=VIEWBOX_WIDTH,
        height=VIEWBOX_HEIGHT,
        extra_svg=(f'<g id="reference" opacity="0.35">{reference_body}</g>',),
    )
    print(f"wrote {OVERLAY_SVG}")


if __name__ == "__main__":
    main()
