"""Regex-based extraction of the geometry embedded in the reference Denby
SVGs (`resources/denby_detector.svg`, `resources/denby_detector_event.svg`).

These are hand-authored, tool-exported files (AMDN), not something worth
pulling in a general SVG/DOM library for -- just enough regex + cubic-Bezier
sampling to pull out:

- the 13 red dashed detection-plane lines (`parse_layer_lines`)
- the 4 blue dashed event-track curves and the small green vertex marker
  (`parse_event_tracks`, `parse_vertex_center`), fit to circles with
  `fit_circle_to_svg_path`
"""

from __future__ import annotations

import math
import re

import numpy as np

_LAYER_LINE_RE = re.compile(r'<path id="Path[^"]*" style="fill:#ffffff[^"]*stroke:#ff1[cd]2[45][^"]*" d="([^"]+)"')
_TRACK_RE = re.compile(r'<path id="Path_\d+" style="fill:none;opacity:1;stroke:#0070ba[^"]*" d="([^"]+)"')
_VERTEX_RE = re.compile(r'<path id="[^"]*" style="fill:#8cc63f[^"]*" d="([^"]+)"')

_NUMBER_RE = re.compile(r"-?[\d.]+")
_M_RE = re.compile(r"M\s*(-?[\d.]+),(-?[\d.]+)\s*(.*)", re.DOTALL)
_C_SEGMENT_RE = re.compile(
    r"C\s*(-?[\d.]+),(-?[\d.]+),(-?[\d.]+),(-?[\d.]+),(-?[\d.]+),(-?[\d.]+)"
)


def extract_svg_body(svg_text: str) -> str:
    """The inner markup of an SVG file, stripped of the XML declaration and
    the outer `<svg ...>...</svg>` wrapper -- for embedding one SVG's content
    inside another (e.g. a dimmed reference layer under a new overlay)."""
    return re.search(r"<svg[^>]*>(.*)</svg>", svg_text, re.DOTALL).group(1)


def parse_layer_lines(svg_text: str) -> list[tuple[float, float, float, float]]:
    """The 13 red dashed detection-plane lines, in document order, as
    ``(x1, y1, x2, y2)`` (each is a degenerate straight `C` curve; only the
    first and last point of the `d` string are needed)."""
    lines = []
    for d in _LAYER_LINE_RE.findall(svg_text):
        nums = [float(n) for n in _NUMBER_RE.findall(d)]
        lines.append((nums[0], nums[1], nums[-2], nums[-1]))
    return lines


def parse_event_tracks(svg_text: str) -> list[str]:
    """Raw path `d` strings of the 4 blue dashed event tracks, document order."""
    return _TRACK_RE.findall(svg_text)


def parse_vertex_path(svg_text: str) -> str:
    """Raw path `d` string of the small green vertex-marker blob."""
    return _VERTEX_RE.search(svg_text).group(1)


def _parse_path_d(d: str) -> tuple[tuple[float, float], list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]]:
    """Parse `M x,y C x,y,x,y,x,y C x,y,x,y,x,y ...` into (start, [(c1, c2, end), ...])."""
    m = _M_RE.match(d.strip())
    start = (float(m.group(1)), float(m.group(2)))
    segments = [
        ((float(a), float(b)), (float(c), float(e)), (float(f), float(g)))
        for a, b, c, e, f, g in _C_SEGMENT_RE.findall(m.group(3))
    ]
    return start, segments


def sample_cubic_path(d: str, n_per_segment: int = 25) -> np.ndarray:
    """Densely sample every cubic-Bezier segment of an SVG path `d` string,
    returning an (N, 2) array of points that lie on the curve."""
    start, segments = _parse_path_d(d)
    points = [start]
    cur = start
    t = np.linspace(0.0, 1.0, n_per_segment)
    mt = 1.0 - t
    for c1, c2, end in segments:
        x = mt**3 * cur[0] + 3 * mt**2 * t * c1[0] + 3 * mt * t**2 * c2[0] + t**3 * end[0]
        y = mt**3 * cur[1] + 3 * mt**2 * t * c1[1] + 3 * mt * t**2 * c2[1] + t**3 * end[1]
        points.extend(zip(x[1:], y[1:]))
        cur = end
    return np.array(points)


def fit_circle(points: np.ndarray) -> tuple[float, float, float]:
    """Algebraic (Kasa) least-squares circle fit. Returns (cx, cy, radius)."""
    x, y = points[:, 0], points[:, 1]
    a = np.column_stack([x, y, np.ones_like(x)])
    b = x**2 + y**2
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r = np.sqrt(sol[2] + cx**2 + cy**2)
    return cx, cy, r


def fit_circle_to_svg_path(d: str, n_per_segment: int = 25) -> tuple[float, float, float]:
    return fit_circle(sample_cubic_path(d, n_per_segment))


def parse_vertex_center(svg_text: str) -> tuple[float, float]:
    """Center of the small green vertex-marker blob (its radius is meaningless)."""
    cx, cy, _ = fit_circle_to_svg_path(parse_vertex_path(svg_text))
    return cx, cy


def _circle_intersections(
    c1: tuple[float, float, float], c2: tuple[float, float, float]
) -> list[tuple[float, float]]:
    x1, y1, r1 = c1
    x2, y2, r2 = c2
    d = math.hypot(x2 - x1, y2 - y1)
    if d == 0.0 or d > r1 + r2 or d < abs(r1 - r2):
        return []
    a = (r1**2 - r2**2 + d**2) / (2.0 * d)
    h2 = r1**2 - a**2
    if h2 < 0.0:
        return []
    h = math.sqrt(h2)
    xm, ym = x1 + a * (x2 - x1) / d, y1 + a * (y2 - y1) / d
    return [
        (xm + h * (y2 - y1) / d, ym - h * (x2 - x1) / d),
        (xm - h * (y2 - y1) / d, ym + h * (x2 - x1) / d),
    ]


def derive_vertex(svg_text: str, tol: float = 5.0) -> tuple[float, float]:
    """The point source: the green marker dot's center, cross-checked against
    the pairwise circle-circle intersections of the 4 event tracks' own
    fitted circles (see tracking/denby/README.md for the full derivation --
    the two independent methods agree to within a couple pixels, which is
    strong evidence the 4 tracks really do share one common vertex there)."""
    dot_center = parse_vertex_center(svg_text)
    circles = [fit_circle_to_svg_path(d) for d in parse_event_tracks(svg_text)]

    candidates = []
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            points = _circle_intersections(circles[i], circles[j])
            if points:
                candidates.append(min(points, key=lambda p: math.hypot(p[0] - dot_center[0], p[1] - dot_center[1])))

    if candidates:
        fit_x = float(np.mean([p[0] for p in candidates]))
        fit_y = float(np.mean([p[1] for p in candidates]))
        disagreement = math.hypot(fit_x - dot_center[0], fit_y - dot_center[1])
        if disagreement > tol:
            raise ValueError(
                f"circle-fit vertex ({fit_x:.1f}, {fit_y:.1f}) disagrees with the marker dot "
                f"{dot_center} by {disagreement:.1f}px (> {tol}px tolerance)"
            )
    return dot_center
