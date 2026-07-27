"""Visualization: a matplotlib event display, and a dependency-free raw-SVG
exporter.

The SVG exporter is deliberately hand-rolled rather than built on
matplotlib's own SVG backend: circular tracks are emitted as *exact* native
SVG elliptical-arc commands (no polyline sampling), and the caller controls
the coordinate system directly (crucial for the Denby recreation, which has
to reuse the reference figure's own pixel coordinates so the two overlay
without any rescaling).
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from detector2d.geometry import CircleLayer, LineLayer, Trajectory
from matplotlib.patches import Circle as MplCircle

from .simulate import trajectory_for_row

#: Okabe-Ito colorblind-safe categorical palette, one color per particle.
DEFAULT_TRACK_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
LAYER_COLOR = "#999999"
HIT_COLOR = "#000000"
VERTEX_COLOR = "#009E73"


def plot_event(particles, hits, layers, event_id: int, track_length: float = 100.0):
    """Matplotlib event display: layers (dashed gray), one colored trajectory
    per particle (drawn out to its farthest hit, or ``track_length`` if it
    has none), hits as outlined markers, vertices as stars."""
    event_particles = particles[particles["event_id"] == event_id]
    event_hits = hits[hits["event_id"] == event_id]

    fig, ax = plt.subplots(figsize=(7, 7))

    for layer in layers:
        if isinstance(layer, LineLayer):
            (x1, y1), (x2, y2) = layer.p1, layer.p2
            ax.plot([x1, x2], [y1, y2], color=LAYER_COLOR, linestyle="--", linewidth=1.0, zorder=1)
        elif isinstance(layer, CircleLayer):
            cx, cy = layer.center
            ax.add_patch(
                MplCircle(
                    (cx, cy), layer.radius, fill=False, edgecolor=LAYER_COLOR, linestyle="--", linewidth=1.0, zorder=1
                )
            )

    for i, (_, particle) in enumerate(event_particles.iterrows()):
        color = DEFAULT_TRACK_COLORS[i % len(DEFAULT_TRACK_COLORS)]
        trajectory = trajectory_for_row(particle)
        particle_hits = event_hits[event_hits["particle_id"] == particle["particle_id"]]
        s_end = float(particle_hits["path_length"].max()) if len(particle_hits) else track_length

        s_values = np.linspace(0.0, s_end, 100)
        xs, ys = zip(*(trajectory.position(s) for s in s_values))
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=1.5,
            zorder=2,
            label=f"particle {int(particle['particle_id'])} (q={particle['charge']:+.0f})",
        )
        ax.plot(*trajectory.position(0.0), marker="*", color=VERTEX_COLOR, markersize=10, zorder=4)
        if len(particle_hits):
            ax.scatter(particle_hits["x"], particle_hits["y"], color=color, edgecolors=HIT_COLOR, s=40, zorder=3)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_title(f"event {event_id}: {len(event_particles)} particle(s), {len(event_hits)} hit(s)")
    if len(event_particles):
        ax.legend(loc="best", frameon=True, fontsize=8)
    fig.tight_layout()
    return fig


def _layer_svg(layer) -> str:
    if isinstance(layer, LineLayer):
        (x1, y1), (x2, y2) = layer.p1, layer.p2
        return f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}"/>'
    if isinstance(layer, CircleLayer):
        cx, cy = layer.center
        return f'<circle cx="{cx:.3f}" cy="{cy:.3f}" r="{layer.radius:.3f}"/>'
    raise TypeError(f"Unknown layer type: {type(layer)!r}")


def _arc_path_d(trajectory: Trajectory, s_end: float) -> str:
    """SVG path `d` for the trajectory from s=0 to s=s_end: a plain `L`ine
    segment if straight, otherwise one or more native `A`rc commands (each
    spanning at most pi radians, so the large-arc-flag is always 0 and the
    sweep-flag alone fully determines the arc -- exact, no sampling)."""
    x0, y0 = trajectory.position(0.0)
    if trajectory.is_straight:
        x1, y1 = trajectory.position(s_end)
        return f"M {x0:.3f},{y0:.3f} L {x1:.3f},{y1:.3f}"

    r = abs(trajectory.radius)
    sweep = 1 if trajectory.radius > 0 else 0
    max_chunk = math.pi * r * 0.999
    n_chunks = max(1, math.ceil(abs(s_end) / max_chunk)) if s_end else 1

    d = f"M {x0:.3f},{y0:.3f}"
    for i in range(1, n_chunks + 1):
        x, y = trajectory.position(s_end * i / n_chunks)
        d += f" A {r:.3f},{r:.3f} 0 0,{sweep} {x:.3f},{y:.3f}"
    return d


def export_svg(
    layers,
    particles,
    hits,
    path: str | Path,
    width: float,
    height: float,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    event_id: int | None = None,
    default_track_length: float = 100.0,
    hit_radius: float = 3.0,
    layer_color: str = LAYER_COLOR,
    layer_stroke_width: float = 1.5,
    layer_dasharray: str = "6,3",
    track_colors: tuple[str, ...] = DEFAULT_TRACK_COLORS,
    hit_color: str = HIT_COLOR,
    vertex_color: str = VERTEX_COLOR,
    vertex_radius: float = 3.0,
    draw_vertices: bool = True,
    extra_svg: tuple[str, ...] = (),
) -> None:
    """Write a self-contained SVG of a detector layout + event to ``path``.

    ``x_offset, y_offset, width, height`` set the viewBox directly in the
    same coordinate system as ``layers``/``particles``/``hits`` -- pass the
    reference figure's own viewBox to make the result directly overlayable.
    ``extra_svg`` is inserted verbatim right after the opening `<svg>` tag
    (e.g. to lay a reference image/paths underneath at reduced opacity).
    """
    if event_id is not None:
        particles = particles[particles["event_id"] == event_id]
        hits = hits[hits["event_id"] == event_id]

    parts = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{x_offset} {y_offset} {width} {height}" width="{width}" height="{height}">',
    ]
    parts.extend(extra_svg)

    parts.append(
        f'<g id="layers" fill="none" stroke="{layer_color}" '
        f'stroke-width="{layer_stroke_width}" stroke-dasharray="{layer_dasharray}">'
    )
    parts.extend(_layer_svg(layer) for layer in layers)
    parts.append("</g>")

    parts.append('<g id="tracks" fill="none" stroke-width="2">')
    vertex_points = []
    for i, (_, particle) in enumerate(particles.iterrows()):
        color = track_colors[i % len(track_colors)]
        trajectory = trajectory_for_row(particle)
        particle_hits = hits[hits["particle_id"] == particle["particle_id"]]
        s_end = float(particle_hits["path_length"].max()) if len(particle_hits) else default_track_length
        parts.append(f'<path d="{_arc_path_d(trajectory, s_end)}" stroke="{color}"/>')
        vertex_points.append(trajectory.position(0.0))
    parts.append("</g>")

    parts.append(f'<g id="hits" fill="{hit_color}" stroke="none">')
    parts.extend(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{hit_radius}"/>' for x, y in zip(hits["x"], hits["y"]))
    parts.append("</g>")

    if draw_vertices and vertex_points:
        parts.append(f'<g id="vertices" fill="{vertex_color}" stroke="none">')
        parts.extend(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{vertex_radius}"/>' for x, y in vertex_points)
        parts.append("</g>")

    parts.append("</svg>")
    Path(path).write_text("\n".join(parts))
