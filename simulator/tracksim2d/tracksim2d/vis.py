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
from viz_style import Theme, palette
from viz_style.mpl import style_axes

from .simulate import boundary_crossing_s, trajectory_for_row

#: Okabe-Ito colorblind-safe categorical palette, one color per particle.
DEFAULT_TRACK_COLORS = palette.CATEGORICAL_OKABE_ITO
LAYER_COLOR = palette.LAYER
HIT_COLOR = palette.HIT
VERTEX_COLOR = palette.VERTEX


def _track_end_s(
    trajectory: Trajectory, particle_hits, track_length: float, tracker_boundary: float | None
) -> float:
    """How far to draw ``trajectory``: out to its farthest hit (or
    ``track_length`` if it has none), capped at the tracker boundary
    crossing (if any, via :func:`tracksim2d.simulate.boundary_crossing_s`) so
    a curved arc doesn't loop back inward past the point where it's left the
    tracker volume. ``hits_for_particles`` already applies this same cutoff
    when producing hits, so this only matters for the drawn curve itself
    (e.g. a particle with no hits, drawn out to ``track_length``)."""
    s_end = float(particle_hits["path_length"].max()) if len(particle_hits) else track_length
    boundary_s = boundary_crossing_s(trajectory, tracker_boundary)
    if boundary_s is not None:
        s_end = min(s_end, boundary_s)
    return s_end


def plot_event(
    particles,
    hits,
    layers,
    event_id: int,
    track_length: float = 100.0,
    tracker_boundary: float | None = None,
    theme: Theme | None = None,
):
    """Matplotlib event display: layers, one colored trajectory per particle
    (drawn out to its farthest hit, or ``track_length`` if it has none,
    capped at ``tracker_boundary`` if given -- see :func:`_track_end_s`),
    hits as outlined markers, vertices as stars.

    A `LineLayer` is an individual physical sensor (e.g. one module of a
    `detector:` `mode: detailed` barrel ring, or a hand-listed `layers:`
    plane), drawn as a solid gray line. A `CircleLayer` is the idealized bare
    surface of a whole layer (`mode: simplified`, no individual sensors to
    show), drawn dashed to mark it as a stand-in rather than real hardware.
    """
    event_particles = particles[particles["event_id"] == event_id]
    event_hits = hits[hits["event_id"] == event_id]

    fig, ax = plt.subplots(figsize=(7, 7))

    for layer in layers:
        if isinstance(layer, LineLayer):
            (x1, y1), (x2, y2) = layer.p1, layer.p2
            ax.plot([x1, x2], [y1, y2], color=LAYER_COLOR, linestyle="-", linewidth=1.0, zorder=1)
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
        s_end = _track_end_s(trajectory, particle_hits, track_length, tracker_boundary)

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

    ax.set_aspect("equal")
    style_axes(
        ax, theme, spatial=True,
        title=f"event {event_id}: {len(event_particles)} particle(s), {len(event_hits)} hit(s)",
        xlabel="x", ylabel="y", legend=bool(len(event_particles)),
    )
    fig.tight_layout()
    return fig


def _layer_svg(layer, dasharray: str) -> str:
    """A `LineLayer` is an individual physical sensor, drawn solid (inherits
    the group's stroke, no dasharray). A `CircleLayer` is the idealized bare
    surface of a whole layer with no individual sensors to show, drawn
    dashed (``dasharray``) to mark it as a stand-in rather than real
    hardware -- see :func:`plot_event`'s docstring for the same convention."""
    if isinstance(layer, LineLayer):
        (x1, y1), (x2, y2) = layer.p1, layer.p2
        return f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}"/>'
    if isinstance(layer, CircleLayer):
        cx, cy = layer.center
        return f'<circle cx="{cx:.3f}" cy="{cy:.3f}" r="{layer.radius:.3f}" stroke-dasharray="{dasharray}"/>'
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
    layer_dasharray: str = "6,3",  # applied to CircleLayer surfaces only; LineLayer sensors are always solid
    track_colors: tuple[str, ...] = DEFAULT_TRACK_COLORS,
    hit_color: str = HIT_COLOR,
    vertex_color: str = VERTEX_COLOR,
    vertex_radius: float = 3.0,
    draw_vertices: bool = True,
    extra_svg: tuple[str, ...] = (),
    tracker_boundary: float | None = None,
) -> None:
    """Write a self-contained SVG of a detector layout + event to ``path``.

    ``x_offset, y_offset, width, height`` set the viewBox directly in the
    same coordinate system as ``layers``/``particles``/``hits`` -- pass the
    reference figure's own viewBox to make the result directly overlayable.
    ``extra_svg`` is inserted verbatim right after the opening `<svg>` tag
    (e.g. to lay a reference image/paths underneath at reduced opacity).
    ``tracker_boundary``, if given, caps how far a drawn arc extends -- see
    :func:`_track_end_s`.
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

    parts.append(f'<g id="layers" fill="none" stroke="{layer_color}" stroke-width="{layer_stroke_width}">')
    parts.extend(_layer_svg(layer, layer_dasharray) for layer in layers)
    parts.append("</g>")

    parts.append('<g id="tracks" fill="none" stroke-width="2">')
    vertex_points = []
    for i, (_, particle) in enumerate(particles.iterrows()):
        color = track_colors[i % len(track_colors)]
        trajectory = trajectory_for_row(particle)
        particle_hits = hits[hits["particle_id"] == particle["particle_id"]]
        s_end = _track_end_s(trajectory, particle_hits, default_track_length, tracker_boundary)
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
