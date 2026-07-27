"""
Shared helpers for building cluster-overlay SVGs on top of the John Snow
Soho base map, and for loading the extracted dot coordinates.
"""

import csv
import re

import numpy as np

WIDTH, HEIGHT = 4417, 4201
BASE_MAP_PATH = "../resources/Soho_map_raw.svg"
DOTS_CSV = "../resources/dots.csv"
GAUGE_CSV = "../resources/gauge.csv"

# Distinct, colorblind-friendlyish palette for up to 5 clusters.
PALETTE = [
    "#e8000d",  # red
    "#1f77b4",  # blue
    "#2ca02c",  # green
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#17becf",  # cyan (spare)
    "#8c564b",  # brown (spare)
]
NOISE_COLOR = "#999999"  # DBSCAN noise points


def load_dots(csv_path: str = DOTS_CSV):
    """Return list of (x, y) tuples."""
    pts = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pts.append((float(row["x"]), float(row["y"])))
    return pts


def make_svg_to_latlon(gauge_csv: str = GAUGE_CSV):
    """
    Fit an affine map (x, y) -> (lat, long) from the 3 reference points in
    gauge.csv, and return a function performing that conversion.

    Three points fully determine a 2D affine transform (6 DOF), which also
    absorbs the small rotation between the map's pixel axes and true
    north/east - a simple axis-aligned scale would not.
    """
    xs, ys, lats, longs = [], [], [], []
    with open(gauge_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
            lats.append(float(row["lat"]))
            longs.append(float(row["long"]))

    A = np.array([[x, y, 1.0] for x, y in zip(xs, ys)])
    coeff_lat = np.linalg.solve(A, np.array(lats))
    coeff_long = np.linalg.solve(A, np.array(longs))

    def svg_to_latlon(x: float, y: float):
        lat = coeff_lat[0] * x + coeff_lat[1] * y + coeff_lat[2]
        long_ = coeff_long[0] * x + coeff_long[1] * y + coeff_long[2]
        return lat, long_

    return svg_to_latlon


def _base_map_paths(base_map_path: str = BASE_MAP_PATH) -> str:
    """Extract the inner <path .../> elements from the plain line-art base map."""
    content = open(base_map_path, encoding="utf-8").read()
    m = re.search(r"<svg[^>]*>(.*)</svg>", content, re.S)
    return m.group(1)


def _cluster_marker(x: float, y: float, color: str, size: float = 34) -> str:
    """A crosshair/diamond marker used to mark a cluster center."""
    h = size / 2
    return (
        f'<g stroke="{color}" stroke-width="6" fill="none">'
        f'<line x1="{x-h}" y1="{y}" x2="{x+h}" y2="{y}"/>'
        f'<line x1="{x}" y1="{y-h}" x2="{x}" y2="{y+h}"/>'
        f'<circle cx="{x}" cy="{y}" r="{h}" stroke="{color}" stroke-width="5" fill="white" fill-opacity="0.15"/>'
        f"</g>"
    )


def render_cluster_svg(
    out_path: str,
    points,
    labels,
    centers,
    dot_radius: float = 11,
    title: str = None,
    base_map_path: str = BASE_MAP_PATH,
):
    """
    Build an SVG with the base map, dots colored by cluster label, and
    cluster-center markers.

    points: list of (x, y)
    labels: list of int cluster ids aligned with points (-1 = noise, for DBSCAN)
    centers: list of (x, y) cluster centers to mark (any length, incl. 0)
    """
    map_paths = _base_map_paths(base_map_path)

    parts = []
    parts.append(
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg version="1.1" xmlns="http://www.w3.org/2000/svg" '
        f'width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">'
    )
    # white background
    parts.append(f'<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>')
    # base map line art
    parts.append(f'<g id="base_map" opacity="0.55">{map_paths}</g>')

    # dots colored by cluster
    parts.append('<g id="dots">')
    for (x, y), lab in zip(points, labels):
        color = NOISE_COLOR if lab == -1 else PALETTE[lab % len(PALETTE)]
        parts.append(
            f'<circle cx="{x}" cy="{y}" r="{dot_radius}" fill="{color}" '
            f'fill-opacity="0.8" stroke="#000000" stroke-opacity="0.25" stroke-width="1"/>'
        )
    parts.append("</g>")

    # cluster center markers
    parts.append('<g id="centers">')
    for i, (cx, cy) in enumerate(centers):
        color = PALETTE[i % len(PALETTE)]
        parts.append(_cluster_marker(cx, cy, color="#000000"))
    parts.append("</g>")

    if title:
        parts.append(
            f'<text x="40" y="70" font-family="Georgia, serif" font-size="60" '
            f'fill="#000000">{title}</text>'
        )

    parts.append("</svg>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))

    print(f"Wrote {out_path}")
