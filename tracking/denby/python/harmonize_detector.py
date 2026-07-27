"""Build the harmonized Denby detector.

The reference figure's 13 red dashed detection-plane lines already share one
length (709.521 units) but vary in left-edge x (136.6-157.1) and vertical
spacing (53-60, mean 58.32). This script re-derives those numbers from the
SVG itself (not hardcoded) and produces 13 identical, equidistant
`LineLayer`s, plus marks the point source (see `denby_svg.derive_vertex` /
README.md for how that's found).

Run from this directory (`tracking/denby/python/`) with the project venv:

    ../.venv/bin/python harmonize_detector.py

Writes ../resources/denby_layers.csv and
../resources/denby_detector_harmonized.svg.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from denby_svg import derive_vertex, parse_layer_lines

from detector2d.geometry import LineLayer
from tracksim2d.edm import HITS_COLUMNS, PARTICLES_COLUMNS
from tracksim2d.vis import export_svg

RESOURCES = Path(__file__).resolve().parent.parent / "resources"
DETECTOR_SVG = RESOURCES / "denby_detector.svg"
EVENT_SVG = RESOURCES / "denby_detector_event.svg"
LAYERS_CSV = RESOURCES / "denby_layers.csv"
HARMONIZED_SVG = RESOURCES / "denby_detector_harmonized.svg"

VIEWBOX_WIDTH = 924.0
VIEWBOX_HEIGHT = 1074.0
PITCH = 10.0  # arbitrary digitization cell size along each layer, in the original px units
VERTEX_MARKER_RADIUS = 6.0
VERTEX_MARKER_COLOR = "#009E73"


def harmonized_layers() -> list[LineLayer]:
    original = parse_layer_lines(open(DETECTOR_SVG).read())
    n = len(original)
    y_first, y_last = original[0][1], original[-1][1]
    spacing = (y_last - y_first) / (n - 1)
    x1 = min(x1 for x1, _, _, _ in original)
    length = sum(x2 - x1 for x1, _, x2, _ in original) / n  # already ~equal; average for robustness
    x2 = x1 + length
    return [
        LineLayer(layer_id=i, p1=(x1, y_first + i * spacing), p2=(x2, y_first + i * spacing), pitch=PITCH)
        for i in range(n)
    ]


def write_layers_csv(layers: list[LineLayer], path: str) -> None:
    with open(path, "w") as fh:
        fh.write("layer_id,x1,y1,x2,y2,pitch\n")
        for layer in layers:
            (x1, y1), (x2, y2) = layer.p1, layer.p2
            fh.write(f"{layer.layer_id},{x1},{y1},{x2},{y2},{layer.pitch}\n")


def main() -> None:
    layers = harmonized_layers()
    write_layers_csv(layers, LAYERS_CSV)

    vertex_x, vertex_y = derive_vertex(open(EVENT_SVG).read())
    vertex_marker = (
        f'<g id="point-source" fill="{VERTEX_MARKER_COLOR}" stroke="none">'
        f'<circle cx="{vertex_x:.3f}" cy="{vertex_y:.3f}" r="{VERTEX_MARKER_RADIUS}"/>'
        f"</g>"
    )
    export_svg(
        layers,
        pd.DataFrame(columns=PARTICLES_COLUMNS),
        pd.DataFrame(columns=HITS_COLUMNS),
        HARMONIZED_SVG,
        width=VIEWBOX_WIDTH,
        height=VIEWBOX_HEIGHT,
        extra_svg=(vertex_marker,),
    )

    spacing = layers[1].p1[1] - layers[0].p1[1]
    print(f"{len(layers)} layers, spacing={spacing:.3f}, length={layers[0].length:.3f}")
    print(f"point source = ({vertex_x:.2f}, {vertex_y:.2f})")
    print(f"wrote {LAYERS_CSV}")
    print(f"wrote {HARMONIZED_SVG}")


if __name__ == "__main__":
    main()
