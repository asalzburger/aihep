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
from detector2d.calorimeter import CaloRing
from detector2d.geometry import CircleLayer, LineLayer, Trajectory
from matplotlib.colors import Normalize
from matplotlib.patches import Circle as MplCircle
from matplotlib.patches import PathPatch, Wedge
from matplotlib.path import Path as MplPath
from viz_style import Theme, palette
from viz_style.mpl import style_axes

from .simulate import boundary_crossing_s, path_for_row, trajectory_for_row

#: Okabe-Ito colorblind-safe categorical palette, one color per particle.
DEFAULT_TRACK_COLORS = palette.CATEGORICAL_OKABE_ITO
LAYER_COLOR = palette.LAYER
CALO_COLOR = palette.CALO
MUON_COLOR = palette.MUON
HIT_COLOR = palette.HIT
VERTEX_COLOR = palette.VERTEX
ENERGY_CMAP = palette.SEQUENTIAL_CHARGE_CMAP

#: Per-system slab/background fill and deposit-energy colormap, so the three
#: subsystems read apart at a glance: a pale volume tint behind each one,
#: with its own energy deposits overlaid in a matching saturated hue.
VOLUME_COLORS = {"ecal": palette.ECAL_VOLUME, "hcal": palette.HCAL_VOLUME, "muon": palette.MUON_VOLUME}
ENERGY_CMAPS = {"ecal": palette.SEQUENTIAL_ECAL_CMAP, "hcal": palette.SEQUENTIAL_HCAL_CMAP}


def _track_end_s(
    path,
    particle_hits,
    track_length: float,
    tracker_boundary: float | None,
    particle_deposits=None,
) -> float:
    """How far to draw ``path``: out to whichever it reaches last, its
    farthest hit or its farthest calorimeter deposit (or ``track_length`` if
    it has neither), capped at the tracker boundary crossing (if any, via
    :func:`detectorsim2d.simulate.boundary_crossing_s`) so a curved arc doesn't
    loop back inward past the point where it's left the tracker volume, and at
    the path's own end.

    Deposits matter here because a neutral EM particle leaves *no* hits at all
    -- a photon drawn only to its hits would not be drawn at all, when in fact
    it flew straight to the ECAL and stopped there.
    """
    reach = float(particle_hits["path_length"].max()) if len(particle_hits) else 0.0
    if particle_deposits is not None and len(particle_deposits):
        radius = float(np.hypot(particle_deposits["x"], particle_deposits["y"]).max())
        deposit_s = boundary_crossing_s(path, radius)
        if deposit_s is not None:
            reach = max(reach, deposit_s)

    s_end = reach if reach > 0.0 else track_length
    boundary_s = boundary_crossing_s(path, tracker_boundary)
    if boundary_s is not None:
        s_end = min(s_end, boundary_s)
    return min(s_end, getattr(path, "total_length", math.inf))


def _layer_style(layer) -> tuple[str, str, float]:
    """``(color, linestyle, linewidth)`` for a layer, by subsystem. A
    `LineLayer` is an individual physical sensor, drawn solid; a bare
    `CircleLayer` is the idealized surface of a whole layer with no individual
    sensors to show, drawn dashed to mark it as a stand-in for hardware."""
    if isinstance(layer, CaloRing):
        return (CALO_COLOR, "-", 0.6)
    if layer.system == "muon":
        return (MUON_COLOR, "-", 1.4)
    return (LAYER_COLOR, "-" if isinstance(layer, LineLayer) else "--", 1.0)


def _muon_rings(layers) -> dict[int, list[tuple[float, float]]]:
    """One entry per muon 'plane' (all `LineLayer`s sharing a `layer_id`,
    e.g. the 8 sides of one octagon station), giving that plane's polygon
    vertices in angular order around the origin. Adjacent sides share a
    corner, so deduping + angle-sorting the sides' own endpoints recovers
    the polygon exactly -- no separate n_sides/apothem needed here."""
    groups: dict[int, list[tuple[float, float]]] = {}
    for layer in layers:
        if isinstance(layer, LineLayer) and layer.system == "muon":
            groups.setdefault(layer.layer_id, []).extend([layer.p1, layer.p2])

    rings = {}
    for layer_id, points in groups.items():
        vertices: list[tuple[float, float]] = []
        for p in points:
            if not any(math.hypot(p[0] - q[0], p[1] - q[1]) < 1e-6 for q in vertices):
                vertices.append(p)
        vertices.sort(key=lambda p: math.atan2(p[1], p[0]))
        rings[layer_id] = vertices
    return rings


def _closed_ring_path(vertices, reverse: bool = False) -> tuple[list[tuple[float, float]], list[int]]:
    """One closed subpath's vertices/codes for `matplotlib.path.Path`,
    optionally reversed (opposite winding is what punches a hole in a
    compound path under the nonzero fill rule)."""
    pts = list(reversed(vertices)) if reverse else list(vertices)
    pts = pts + [pts[0]]
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(pts) - 2) + [MplPath.CLOSEPOLY]
    return pts, codes


def _muon_background_patch(layers) -> PathPatch | None:
    """The muon system's own octagonal (or however many-sided) background
    volume, from its outermost plane's polygon down to its innermost one's
    -- an 'octagonal annulus', not a circular one, since real muon stations
    are flat-sided chambers, not a cylinder."""
    rings = _muon_rings(layers)
    if not rings:
        return None

    def mean_radius(vertices):
        return sum(math.hypot(x, y) for x, y in vertices) / len(vertices)

    ordered = sorted(rings.values(), key=mean_radius)
    outer_vertices = ordered[-1]
    outer_pts, outer_codes = _closed_ring_path(outer_vertices)

    if len(ordered) > 1:
        inner_pts, inner_codes = _closed_ring_path(ordered[0], reverse=True)
        path = MplPath(outer_pts + inner_pts, outer_codes + inner_codes)
    else:
        # only one plane found (e.g. a single hand-built station): no inner
        # boundary to punch a hole with, so just fill the solid polygon.
        path = MplPath(outer_pts, outer_codes)

    return PathPatch(path, facecolor=VOLUME_COLORS["muon"], edgecolor="none", zorder=0.5)


def _draw_layers(ax, layers) -> None:
    muon_background = _muon_background_patch(layers)
    if muon_background is not None:
        ax.add_patch(muon_background)

    for layer in layers:
        color, linestyle, linewidth = _layer_style(layer)
        if isinstance(layer, CaloRing):
            # a calorimeter layer has real radial depth: draw the slab it
            # occupies (tinted by subsystem), so its cells have somewhere to live
            ax.add_patch(
                Wedge(
                    layer.center, layer.radius + 0.5 * layer.thickness, 0.0, 360.0,
                    width=layer.thickness, facecolor=VOLUME_COLORS.get(layer.system, "none"),
                    edgecolor=color, linewidth=linewidth, zorder=1,
                )
            )
        elif isinstance(layer, LineLayer):
            (x1, y1), (x2, y2) = layer.p1, layer.p2
            ax.plot([x1, x2], [y1, y2], color=color, linestyle=linestyle, linewidth=linewidth, zorder=1)
        elif isinstance(layer, CircleLayer):
            cx, cy = layer.center
            ax.add_patch(
                MplCircle(
                    (cx, cy), layer.radius, fill=False, edgecolor=color,
                    linestyle=linestyle, linewidth=linewidth, zorder=1,
                )
            )


def _rings_by_layer(layers) -> dict[tuple[str, int], CaloRing]:
    return {
        (ring.system, ring.layer_id): ring for ring in layers if isinstance(ring, CaloRing)
    }


def _draw_deposits(ax, deposits, layers) -> None:
    """Each deposit as a filled wedge covering its own cell, shaded by energy
    -- the calorimeter's actual readout granularity, rather than a point.
    Colored per subsystem (ECAL red, HCAL orange -- see `ENERGY_CMAPS`) so a
    shower's system is legible at a glance, on top of that system's own pale
    volume tint."""
    rings = _rings_by_layer(layers)
    if not len(deposits) or not rings:
        return
    cmaps = {system: plt.get_cmap(name) for system, name in ENERGY_CMAPS.items()}
    default_cmap = plt.get_cmap(ENERGY_CMAP)
    norm = Normalize(vmin=0.0, vmax=float(deposits["energy"].max()) or 1.0)

    for _, deposit in deposits.iterrows():
        ring = rings.get((deposit["system"], int(deposit["layer_id"])))
        if ring is None:
            continue
        low, high = ring.cell_edges(int(deposit["cell_id"]))
        # fade with energy as well as coloring by it: a shower's Gaussian tail
        # cells carry a per-mille of its energy, and drawing them at full
        # opacity makes every shower look several times wider than it is
        shade = float(norm(deposit["energy"]))
        cmap = cmaps.get(deposit["system"], default_cmap)
        ax.add_patch(
            Wedge(
                ring.center, ring.radius + 0.5 * ring.thickness,
                math.degrees(low), math.degrees(high), width=ring.thickness,
                facecolor=cmap(shade), alpha=0.15 + 0.85 * shade,
                edgecolor="none", zorder=2,
            )
        )


def _particle_label(particle) -> str:
    species = particle.get("species")
    if species is None or (isinstance(species, float) and math.isnan(species)):
        return f"particle {int(particle['particle_id'])} (q={particle['charge']:+.0f})"
    energy = particle.get("energy")
    suffix = f", E={energy:.0f}" if energy is not None and not math.isnan(energy) else ""
    return f"{species}{suffix}"


def plot_event(
    particles,
    hits,
    layers,
    event_id: int,
    track_length: float = 100.0,
    tracker_boundary: float | None = None,
    theme: Theme | None = None,
    deposits=None,
    field=None,
    world_radius: float | None = None,
    max_path_length: float | None = None,
):
    """Matplotlib event display: layers, one colored trajectory per particle
    (drawn out to its farthest hit or deposit, or ``track_length`` if it has
    neither, capped at ``tracker_boundary`` if given -- see
    :func:`_track_end_s`), hits as outlined markers, calorimeter deposits as
    energy-shaded cells, vertices as stars.

    A `LineLayer` is an individual physical sensor (e.g. one module of a
    `detector:` `mode: detailed` barrel ring, a muon chamber, or a hand-listed
    `layers:` plane), drawn as a solid line. A `CircleLayer` is the idealized
    bare surface of a whole layer (`mode: simplified`, no individual sensors to
    show), drawn dashed to mark it as a stand-in rather than real hardware. A
    `CaloRing` is drawn as the radial slab it occupies, tinted by subsystem
    (ECAL pale yellow-green, HCAL pale blue -- `VOLUME_COLORS`) with its
    energy deposits overlaid in a matching saturated hue (ECAL red, HCAL
    orange -- `ENERGY_CMAPS`), so its cells have somewhere to sit and which
    system they belong to is legible at a glance. The muon system gets one
    overall pale background volume the same way, behind its individual
    chamber planes.

    Pass ``field`` (a :class:`~detector2d.field.FieldRegions`) to draw the
    piecewise trajectory the particle actually followed -- bending one way in
    the tracker, straight through the calorimeters, bending the other way in
    the muon system. Without it, each particle is drawn as the single arc its
    stored ``radius`` describes.
    """
    event_particles = particles[particles["event_id"] == event_id]
    event_hits = hits[hits["event_id"] == event_id]
    event_deposits = None if deposits is None else deposits[deposits["event_id"] == event_id]

    fig, ax = plt.subplots(figsize=(7, 7))

    _draw_layers(ax, layers)
    if event_deposits is not None:
        _draw_deposits(ax, event_deposits, layers)

    for i, (_, particle) in enumerate(event_particles.iterrows()):
        color = DEFAULT_TRACK_COLORS[i % len(DEFAULT_TRACK_COLORS)]
        path = path_for_row(particle, field, world_radius, max_path_length)
        particle_hits = event_hits[event_hits["particle_id"] == particle["particle_id"]]
        particle_deposits = (
            None
            if event_deposits is None
            else event_deposits[event_deposits["particle_id"] == particle["particle_id"]]
        )
        s_end = _track_end_s(path, particle_hits, track_length, tracker_boundary, particle_deposits)

        s_values = np.linspace(0.0, s_end, 300)
        xs, ys = zip(*(path.position(s) for s in s_values))
        ax.plot(xs, ys, color=color, linewidth=1.5, zorder=3, label=_particle_label(particle))
        ax.plot(*path.position(0.0), marker="*", color=VERTEX_COLOR, markersize=10, zorder=5)
        if len(particle_hits):
            ax.scatter(
                particle_hits["x"], particle_hits["y"],
                color=color, edgecolors=HIT_COLOR, s=40, zorder=4,
            )

    ax.set_aspect("equal")
    summary = f"{len(event_particles)} particle(s), {len(event_hits)} hit(s)"
    if event_deposits is not None:
        summary += f", {len(event_deposits)} deposit(s)"
    style_axes(
        ax, theme, spatial=True,
        title=f"event {event_id}: {summary}",
        xlabel="x", ylabel="y", legend=bool(len(event_particles)),
    )
    fig.tight_layout()
    return fig


def plot_lego(
    deposits,
    layers,
    event_id: int,
    theme: Theme | None = None,
    systems: tuple[str, ...] = ("ecal", "hcal"),
    phi_range: tuple[float, float] | None = None,
):
    """The calorimeter unrolled: azimuth across, one row per sampling layer,
    each cell shaded by its energy.

    Shows the longitudinal profile (each layer dimmer than the one before it)
    and the lateral spread (each layer wider) at a glance, which the x/y
    display cannot.

    ``phi_range`` (a ``(low, high)`` pair in degrees) zooms in on one shower.
    Zoom in far enough and individual cell edges are drawn -- which is the
    only way to *see* the ECAL's half-cell stagger, since at full scale a
    256-cell ring's half cell is well under a degree.
    """
    event_deposits = deposits[deposits["event_id"] == event_id]
    rings = [r for r in layers if isinstance(r, CaloRing) and r.system in systems]
    rings.sort(key=lambda ring: ring.radius)
    if not rings:
        raise ValueError(f"no calorimeter rings for systems {systems!r} in this layout")

    summed = event_deposits.groupby(["system", "layer_id", "cell_id"], as_index=False)["energy"].sum()
    energies = {
        (row["system"], int(row["layer_id"]), int(row["cell_id"])): row["energy"]
        for _, row in summed.iterrows()
    }

    fig, ax = plt.subplots(figsize=(10, 4))
    cmap = plt.get_cmap(ENERGY_CMAP)
    vmax = float(summed["energy"].max()) if len(summed) else 1.0
    norm = Normalize(vmin=0.0, vmax=vmax or 1.0)

    low_deg, high_deg = phi_range if phi_range is not None else (0.0, 360.0)
    # only outline individual cells once few enough are on screen to resolve
    # them; otherwise the outlines merge into a solid smear
    show_edges = {
        ring.layer_id: (high_deg - low_deg) / math.degrees(ring.dphi) <= 80 for ring in rings
    }

    for row_index, ring in enumerate(rings):
        for cell in range(ring.n_phi):
            edge_low, edge_high = ring.cell_edges(cell)
            center = math.degrees(0.5 * (edge_low + edge_high)) % 360.0
            if not (low_deg - 1.0 <= center <= high_deg + 1.0):
                continue
            energy = energies.get((ring.system, ring.layer_id, cell), 0.0)
            ax.bar(
                x=center,
                height=0.9,
                width=math.degrees(edge_high - edge_low),
                bottom=row_index + 0.05,
                color=cmap(norm(energy)) if energy > 0 else "none",
                edgecolor=CALO_COLOR if show_edges[ring.layer_id] else "none",
                linewidth=0.3,
                align="center",
            )

    ax.set_xlim(low_deg, high_deg)
    ax.set_ylim(0.0, len(rings))
    ax.set_yticks([i + 0.5 for i in range(len(rings))])
    ax.set_yticklabels([f"{ring.system} L{ring.layer_id}" for ring in rings])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="energy")
    # spatial=False: unrolling azimuth turns this into an analysis plot, and
    # it is unreadable without its phi axis and layer labels even in `present`
    style_axes(
        ax, theme, spatial=False,
        title=f"event {event_id}: calorimeter cells, unrolled in azimuth",
        xlabel="phi [deg]",
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
    if isinstance(layer, CaloRing):
        # real hardware with radial depth, so solid, and drawn as the two
        # bounding circles of the slab it occupies
        cx, cy = layer.center
        return "".join(
            f'<circle cx="{cx:.3f}" cy="{cy:.3f}" r="{layer.radius + sign * 0.5 * layer.thickness:.3f}"/>'
            for sign in (-1, 1)
        )
    if isinstance(layer, CircleLayer):
        cx, cy = layer.center
        return f'<circle cx="{cx:.3f}" cy="{cy:.3f}" r="{layer.radius:.3f}" stroke-dasharray="{dasharray}"/>'
    raise TypeError(f"Unknown layer type: {type(layer)!r}")


def _arc_commands(trajectory: Trajectory, s_start: float, s_end: float) -> str:
    """SVG path commands continuing from ``s_start`` to ``s_end`` along one
    arc (no leading `M`): a plain `L`ine if straight, otherwise one or more
    native `A`rc commands (each spanning at most pi radians, so the
    large-arc-flag is always 0 and the sweep-flag alone fully determines the
    arc -- exact, no sampling)."""
    span = s_end - s_start
    if trajectory.is_straight:
        x1, y1 = trajectory.position(s_end)
        return f" L {x1:.3f},{y1:.3f}"

    r = abs(trajectory.radius)
    sweep = 1 if trajectory.radius > 0 else 0
    max_chunk = math.pi * r * 0.999
    n_chunks = max(1, math.ceil(abs(span) / max_chunk)) if span else 1

    commands = ""
    for i in range(1, n_chunks + 1):
        x, y = trajectory.position(s_start + span * i / n_chunks)
        commands += f" A {r:.3f},{r:.3f} 0 0,{sweep} {x:.3f},{y:.3f}"
    return commands


def _arc_path_d(path, s_end: float) -> str:
    """SVG path `d` from s=0 to s=s_end. A segmented path emits one run of
    commands per segment, so a track that bends, straightens, then bends the
    other way stays a single exact `<path>` -- no polyline sampling anywhere."""
    x0, y0 = path.position(0.0)
    d = f"M {x0:.3f},{y0:.3f}"

    segments = getattr(path, "segments", None)
    if segments is None:
        return d + _arc_commands(path, 0.0, s_end)

    for segment in segments:
        if segment.s_start >= s_end:
            break
        stop = min(segment.s_end, s_end)
        d += _arc_commands(segment.trajectory, 0.0, stop - segment.s_start)
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
    field=None,
    world_radius: float | None = None,
    max_path_length: float | None = None,
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
        # `track`, not `path` -- `path` is this function's output file argument
        track = path_for_row(particle, field, world_radius, max_path_length)
        particle_hits = hits[hits["particle_id"] == particle["particle_id"]]
        s_end = _track_end_s(track, particle_hits, default_track_length, tracker_boundary)
        parts.append(f'<path d="{_arc_path_d(track, s_end)}" stroke="{color}"/>')
        vertex_points.append(track.position(0.0))
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
