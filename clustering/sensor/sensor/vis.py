"""Matplotlib visualization of simulated events and clusters.

Static, print-style figures (not the interactive-dashboard case the dataviz
skill mostly targets) but the same principles apply where they're relevant:
a perceptually-uniform sequential colormap for charge (never a rainbow map),
one fixed accent color per categorical overlay (cluster boxes vs. truth
tracks), and a legend whenever more than one series is on the plot.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401 -- registers the "3d" projection
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from viz_style import Theme, get_theme, palette
from viz_style.mpl import style_axes

from .analysis import CENTROID_COLUMNS, compute_residuals
from .sim import DetectorConfig, charge_endpoints, true_center_position

CLUSTER_COLOR = palette.CLUSTER
CLUSTER_OUTLINE_COLOR = "black"
TRUTH_COLOR = palette.TRUTH
CHARGE_CMAP = palette.SEQUENTIAL_CHARGE_CMAP  # sequential: low charge yellow -> high charge red
DIGITAL_ON_COLOR = palette.DIGITAL_ON
GRID_COLOR = palette.GRID
SENSOR_BOX_COLOR = palette.LAYER  # same structural gray as detector2d/tracksim2d's own layer drawing
DEFAULT_READOUT_THRESHOLD = 0.15  # pixels with charge <= this are not "read out"
DEFAULT_3D_ZOOM = (20, 20)  # pixel window around the event's tracks, for a proportioned slab illustration

# Centroid-type identity, shared between the per-event overlay (plot_event)
# and the residual/summary plots: charge-weighted reuses the existing
# cluster-box red, digital gets its own hue so the two never collide.
CENTROID_TYPE_COLOR = {"charge": CLUSTER_COLOR, "digital": palette.CLUSTER_DIGITAL}
CENTROID_TYPE_LABEL = {"charge": "charge-weighted", "digital": "digital"}
CENTROID_TYPE_MARKER = {"charge": "D", "digital": "s"}
TRUE_POSITION_MARKER = "*"
ENTRY_MARKER = "o"  # truth track entry point (charge-collection segment's z=0 end)
EXIT_MARKER = "^"  # truth track exit point (z=thickness end) -- a filled triangle


def _clip_window(center: float, size: int, n_max: int) -> tuple[int, int]:
    """size-pixel window centered as closely as possible on `center` (a
    continuous pixel-index-space position -- see `_footprint_center`, not
    necessarily an integer), shifted to fit inside [0, n_max)."""
    lo = round(center - size / 2)
    lo = max(0, min(lo, max(n_max - size, 0)))
    return lo, lo + size


def _footprint_center(ix: pd.Series, iy: pd.Series) -> tuple[float, float]:
    """(center_ix, center_iy) of a set of hit pixels' bounding box, in
    continuous pixel-index space -- pixel i spans [i, i+1), so a footprint
    occupying ix in [ix_min, ix_max] is centered at (ix_min+ix_max+1)/2.
    Deliberately the geometric center of the *pixel footprint*, not a
    charge-weighted centroid -- the latter skews toward whichever pixel
    happens to be charge-heavier, which can badly off-center the zoom
    window for an asymmetric-charge cluster."""
    center_ix = (ix.min() + ix.max() + 1) / 2.0
    center_iy = (iy.min() + iy.max() + 1) / 2.0
    return center_ix, center_iy


def _zoom_pixel_bounds(
    event_hits: pd.DataFrame, event_clusters: pd.DataFrame, detector: DetectorConfig, zoom: tuple[int, int]
) -> tuple[int, int, int, int]:
    """(ix_lo, ix_hi, iy_lo, iy_hi) window of `zoom` pixels, centered as
    closely as possible on the event's largest cluster's actual pixel
    footprint (falls back to the grid center if there is none)."""
    nx, ny = zoom
    if not event_clusters.empty:
        target_id = event_clusters.sort_values("charge_sum", ascending=False).iloc[0]["cluster_id"]
        target_hits = event_hits[event_hits["cluster_id"] == target_id]
        center_ix, center_iy = _footprint_center(target_hits["ix"], target_hits["iy"])
    else:
        center_ix, center_iy = detector.n_pixels_x / 2.0, detector.n_pixels_y / 2.0
    ix_lo, ix_hi = _clip_window(center_ix, nx, detector.n_pixels_x)
    iy_lo, iy_hi = _clip_window(center_iy, ny, detector.n_pixels_y)
    return ix_lo, ix_hi, iy_lo, iy_hi


def _cluster_outline_segments(
    ix: list[int], iy: list[int], pitch_x: float, pitch_y: float
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Boundary edges of the union of pixel squares in a cluster.

    An edge between two grid cells is only kept if exactly one side is a
    cluster pixel, so the result traces the true (possibly concave or
    donut-shaped) outline rather than the axis-aligned bounding box.
    """
    pixels = set(zip(ix, iy))
    segments = []
    for px, py in pixels:
        x0, x1 = px * pitch_x, (px + 1) * pitch_x
        y0, y1 = py * pitch_y, (py + 1) * pitch_y
        if (px - 1, py) not in pixels:
            segments.append(((x0, y0), (x0, y1)))
        if (px + 1, py) not in pixels:
            segments.append(((x1, y0), (x1, y1)))
        if (px, py - 1) not in pixels:
            segments.append(((x0, y0), (x1, y0)))
        if (px, py + 1) not in pixels:
            segments.append(((x0, y1), (x1, y1)))
    return segments


def plot_event(
    hits: pd.DataFrame,
    clusters: pd.DataFrame,
    truth: pd.DataFrame,
    detector: DetectorConfig,
    event_id: int,
    zoom: tuple[int, int] | None = None,
    grid: bool = False,
    readout_threshold: float = DEFAULT_READOUT_THRESHOLD,
    digital: bool = False,
    centroid_types: tuple[str, ...] = ("charge",),
    theme: Theme | None = None,
):
    """Plot one event's pixel grid, cluster boxes, and truth tracks.

    zoom: if given, (nx, ny) pixel window shown, centered as closely as
        possible on the event's largest cluster's actual pixel footprint
        (by summed charge) instead of the full sensor -- not a
        charge-weighted centroid, which would skew off-center toward
        whichever pixel happens to be charge-heavier.
    grid: if True, overlay light pixel-boundary gridlines.
    readout_threshold: pixels with charge at or below this are treated as
        not read out (dropped from the display and from cluster boxes),
        mimicking a real front-end's readout threshold.
    digital: if True, show surviving pixels as flat on/off instead of
        charge-graded color.
    centroid_types: which reconstructed centroid(s) to mark per cluster —
        any of "charge" (charge-weighted) and "digital" (unweighted); each
        truth particle's true position (its own track, evaluated at the
        sensor's mid-thickness plane) is always marked alongside them.
    theme: print (default) or present -- see viz_style.
    """
    event_hits = hits[hits["event_id"] == event_id]
    event_hits = event_hits[event_hits["charge"] > readout_threshold]
    event_clusters = clusters[clusters["event_id"] == event_id]
    event_clusters = event_clusters[event_clusters["cluster_id"].isin(event_hits["cluster_id"])]
    event_truth = truth[truth["event_id"] == event_id]

    charge_grid = np.zeros((detector.n_pixels_x, detector.n_pixels_y))
    charge_grid[event_hits["ix"].to_numpy(), event_hits["iy"].to_numpy()] = event_hits["charge"].to_numpy()

    x_edges = np.arange(detector.n_pixels_x + 1) * detector.pitch_x_um
    y_edges = np.arange(detector.n_pixels_y + 1) * detector.pitch_y_um

    if zoom is not None:
        ix_lo, ix_hi, iy_lo, iy_hi = _zoom_pixel_bounds(event_hits, event_clusters, detector, zoom)
        x_view = (ix_lo * detector.pitch_x_um, ix_hi * detector.pitch_x_um)
        y_view = (iy_lo * detector.pitch_y_um, iy_hi * detector.pitch_y_um)
    else:
        x_view = (x_edges[0], x_edges[-1])
        y_view = (y_edges[0], y_edges[-1])

    fig, ax = plt.subplots(figsize=(7, 7))
    if digital:
        digital_display = np.ma.masked_equal((charge_grid.T > 0).astype(float), 0.0)
        ax.pcolormesh(x_edges, y_edges, digital_display, cmap=ListedColormap([DIGITAL_ON_COLOR]), shading="flat")
    else:
        charge_display = np.ma.masked_equal(charge_grid.T, 0.0)
        mesh = ax.pcolormesh(x_edges, y_edges, charge_display, cmap=CHARGE_CMAP, shading="flat")
        resolved_theme = theme or get_theme()
        colorbar_label = "charge" if resolved_theme.show_spatial_axes else None
        fig.colorbar(mesh, ax=ax, label=colorbar_label, fraction=0.046, pad=0.04)

    if grid:
        # Explicit line segments (not matplotlib's tick/grid machinery) so
        # every pixel border in view is guaranteed to be drawn, regardless
        # of how many pixels are visible.
        xs_in_view = x_edges[(x_edges >= x_view[0]) & (x_edges <= x_view[1])]
        ys_in_view = y_edges[(y_edges >= y_view[0]) & (y_edges <= y_view[1])]
        ax.vlines(xs_in_view, y_view[0], y_view[1], color=GRID_COLOR, linewidth=0.6, zorder=1)
        ax.hlines(ys_in_view, x_view[0], x_view[1], color=GRID_COLOR, linewidth=0.6, zorder=1)

    for _, ix, iy in event_hits.groupby("cluster_id")[["ix", "iy"]].agg(list).itertuples():
        segments = _cluster_outline_segments(ix, iy, detector.pitch_x_um, detector.pitch_y_um)
        ax.add_collection(LineCollection(segments, colors=CLUSTER_OUTLINE_COLOR, linewidths=2, zorder=3))

    for _, cluster_row in event_clusters.iterrows():
        for centroid_type in centroid_types:
            x_col, y_col = CENTROID_COLUMNS[centroid_type]
            ax.plot(
                cluster_row[x_col],
                cluster_row[y_col],
                marker=CENTROID_TYPE_MARKER[centroid_type],
                color=CENTROID_TYPE_COLOR[centroid_type],
                markersize=7,
                markeredgecolor="black",
                markeredgewidth=0.5,
                zorder=4,
            )

    for _, row in event_truth.iterrows():
        p0, p1 = charge_endpoints(
            row["x0_um"], row["y0_um"], row["dxdz"], row["dydz"], detector.thickness_um, detector.lorentz_slope
        )
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=TRUTH_COLOR, linewidth=1.5, zorder=3)
        ax.plot(*p0, marker=ENTRY_MARKER, color=TRUTH_COLOR, markersize=4, zorder=3)
        ax.plot(*p1, marker=EXIT_MARKER, color=TRUTH_COLOR, markersize=6, zorder=3)

        true_x, true_y = true_center_position(
            row["x0_um"], row["y0_um"], row["dxdz"], row["dydz"], detector.thickness_um
        )
        ax.plot(
            true_x, true_y,
            marker=TRUE_POSITION_MARKER, color=TRUTH_COLOR, markersize=11,
            markeredgecolor="black", markeredgewidth=0.5, zorder=4,
        )

    ax.set_xlim(*x_view)
    ax.set_ylim(*y_view)
    ax.set_aspect("equal")
    mode = " [digital]" if digital else ""
    title = (
        f"event {event_id}{mode}: {len(event_hits)} hit pixels "
        f"(readout > {readout_threshold}), {event_hits['cluster_id'].nunique()} cluster(s)"
    )

    handles = [
        plt.Line2D([0], [0], color=CLUSTER_COLOR, linewidth=2, label="cluster bounding box"),
        plt.Line2D([0], [0], color=TRUTH_COLOR, linewidth=1.5, marker=ENTRY_MARKER, markersize=4, label="truth track"),
        plt.Line2D(
            [0], [0], color=TRUTH_COLOR, marker=TRUE_POSITION_MARKER, markersize=9, linewidth=0,
            markeredgecolor="black", markeredgewidth=0.5, label="true position (slab center)",
        ),
    ]
    for centroid_type in centroid_types:
        handles.append(
            plt.Line2D(
                [0], [0], color=CENTROID_TYPE_COLOR[centroid_type], marker=CENTROID_TYPE_MARKER[centroid_type],
                markersize=7, linewidth=0, markeredgecolor="black", markeredgewidth=0.5,
                label=f"reconstructed ({CENTROID_TYPE_LABEL[centroid_type]})",
            )
        )
    if digital:
        handles.insert(0, Patch(facecolor=DIGITAL_ON_COLOR, label="hit (on)"))
    style_axes(
        ax, theme, spatial=True, title=title, xlabel="x [um]", ylabel="y [um]",
        legend=True, legend_handles=handles, legend_loc="upper right",
    )

    fig.tight_layout()
    return fig


def _box_faces(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> list[np.ndarray]:
    """The 6 quad faces of an axis-aligned box spanning [x0, x1] x [y0, y1] x
    [z0, z1], each as a (4, 3) array of corners -- shared by the outer
    sensor-slab box and every per-pixel "hit voxel" in `plot_event_3d`."""
    corners = np.array(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ]
    )
    return [
        corners[[0, 1, 2, 3]],  # bottom (z0)
        corners[[4, 5, 6, 7]],  # top (z1)
        corners[[0, 1, 5, 4]],
        corners[[2, 3, 7, 6]],
        corners[[1, 2, 6, 5]],
        corners[[0, 3, 7, 4]],
    ]


def plot_event_3d(
    hits: pd.DataFrame,
    truth: pd.DataFrame,
    detector: DetectorConfig,
    event_id: int,
    zoom: tuple[int, int] | None = None,
    readout_threshold: float = DEFAULT_READOUT_THRESHOLD,
):
    """Presenter-only 3D illustration: a somewhat transparent sensor slab
    (the event's truth particle(s) drawn as straight lines traversing it,
    entry face at z=0 to readout face at z=thickness_um, each marked with a
    dot at entry and a filled triangle at exit, same convention as
    `plot_event`). Every hit pixel
    (post-`readout_threshold`) is drawn as a fully colored 3D block spanning
    the whole thickness -- the "3D pixel" the track actually traversed --
    filled with the same charge colormap `plot_event` uses (no colorbar/
    palette shown, colors computed directly rather than via a mappable).
    Each block's own top face sits on the readout (top) surface, so
    together they read as the 2D charge/cluster image projected onto the
    top of the slab; a very light pixel-boundary grid covers the rest of
    that top surface for context. No cluster boxes, no reconstructed
    centroids, no title/axes/legend -- unlike `plot_event`, this is always
    rendered presenter-style; there is no `theme` argument.

    zoom: (nx, ny) pixel window shown, centered on the event's largest
        cluster's actual pixel footprint (default: `DEFAULT_3D_ZOOM`), same
        as `plot_event`'s --zoom; falls back to the truth particles' mean
        entry point if there are no hits above `readout_threshold`. The
        full sensor is many times wider/taller than it is thick, so an
        unzoomed box would look like a flat pancake with no visible tilt.
    readout_threshold: pixels with charge at or below this are not marked
        as hit, same convention as `plot_event`.
    """
    event_truth = truth[truth["event_id"] == event_id]
    if event_truth.empty:
        raise ValueError(f"No truth particles for event_id={event_id}")
    event_hits = hits[(hits["event_id"] == event_id) & (hits["charge"] > readout_threshold)]

    nx, ny = zoom if zoom is not None else DEFAULT_3D_ZOOM
    half_x = nx / 2 * detector.pitch_x_um
    half_y = ny / 2 * detector.pitch_y_um
    if len(event_hits):
        # centered on the largest cluster's actual pixel footprint, same as
        # `plot_event`'s --zoom -- not the truth entry points, which can sit
        # well away from where the (possibly drifted/angled) track actually
        # deposits charge.
        target_id = event_hits.groupby("cluster_id")["charge"].sum().idxmax()
        target_hits = event_hits[event_hits["cluster_id"] == target_id]
        center_ix, center_iy = _footprint_center(target_hits["ix"], target_hits["iy"])
        cx, cy = center_ix * detector.pitch_x_um, center_iy * detector.pitch_y_um
    else:
        cx, cy = event_truth["x0_um"].mean(), event_truth["y0_um"].mean()
    x_lo, x_hi = cx - half_x, cx + half_x
    y_lo, y_hi = cy - half_y, cy + half_y
    z_lo, z_hi = 0.0, detector.thickness_um

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection="3d")

    box = Poly3DCollection(
        _box_faces(x_lo, x_hi, y_lo, y_hi, z_lo, z_hi),
        facecolor=mcolors.to_rgba(SENSOR_BOX_COLOR, alpha=0.08),
        edgecolor=mcolors.to_rgba(SENSOR_BOX_COLOR, alpha=0.6),
        linewidths=0.8,
    )
    ax.add_collection3d(box)

    # Very light pixel grid across the zoom window, on the top (readout)
    # surface -- the "2D view" lands on the top face, not floating inside.
    ix_lo = int(x_lo // detector.pitch_x_um)
    ix_hi = int(np.ceil(x_hi / detector.pitch_x_um))
    iy_lo = int(y_lo // detector.pitch_y_um)
    iy_hi = int(np.ceil(y_hi / detector.pitch_y_um))
    x_edges = np.arange(ix_lo, ix_hi + 1) * detector.pitch_x_um
    y_edges = np.arange(iy_lo, iy_hi + 1) * detector.pitch_y_um
    grid_segments = [[(gx, y_lo, z_hi), (gx, y_hi, z_hi)] for gx in x_edges]
    grid_segments += [[(x_lo, gy, z_hi), (x_hi, gy, z_hi)] for gy in y_edges]
    ax.add_collection3d(
        Line3DCollection(grid_segments, colors=mcolors.to_rgba(GRID_COLOR, alpha=0.75), linewidths=0.4)
    )

    # Hit pixels: the full 3D pixel volume the track traversed, solid-filled
    # with the charge colormap (computed directly into per-face facecolors,
    # not a mappable, so there's no colorbar/palette added to the figure).
    # Each voxel's own top face sits on the readout surface, so together
    # they form the projected 2D cluster-charge image on top of the slab.
    window_hits = event_hits[
        event_hits["x_center_um"].between(x_lo, x_hi) & event_hits["y_center_um"].between(y_lo, y_hi)
    ]
    if len(window_hits):
        cmap = plt.get_cmap(CHARGE_CMAP)
        norm = mcolors.Normalize(vmin=0.0, vmax=window_hits["charge"].max())
        voxel_faces, voxel_colors = [], []
        for _, hit in window_hits.iterrows():
            hx0, hx1 = hit["x_center_um"] - detector.pitch_x_um / 2, hit["x_center_um"] + detector.pitch_x_um / 2
            hy0, hy1 = hit["y_center_um"] - detector.pitch_y_um / 2, hit["y_center_um"] + detector.pitch_y_um / 2
            color = cmap(norm(hit["charge"]))
            faces = _box_faces(hx0, hx1, hy0, hy1, z_lo, z_hi)
            voxel_faces.extend(faces)
            voxel_colors.extend([color] * len(faces))
        ax.add_collection3d(Poly3DCollection(voxel_faces, facecolors=voxel_colors, edgecolors="none"))

    for _, row in event_truth.iterrows():
        x0, y0 = row["x0_um"], row["y0_um"]
        x1, y1 = x0 + z_hi * row["dxdz"], y0 + z_hi * row["dydz"]
        ax.plot([x0, x1], [y0, y1], [z_lo, z_hi], color=TRUTH_COLOR, linewidth=2.5, zorder=5)
        ax.plot([x0], [y0], [z_lo], marker=ENTRY_MARKER, color=TRUTH_COLOR, markersize=6, zorder=6)
        ax.plot([x1], [y1], [z_hi], marker=EXIT_MARKER, color=TRUTH_COLOR, markersize=9, zorder=6)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_zlim(z_lo, z_hi)
    ax.set_box_aspect((x_hi - x_lo, y_hi - y_lo, z_hi - z_lo))
    ax.view_init(elev=18, azim=-50)
    ax.set_axis_off()

    fig.tight_layout()
    return fig


def plot_cluster_summary(clusters: pd.DataFrame, theme: Theme | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(clusters["n_pixels"], bins=np.arange(0.5, clusters["n_pixels"].max() + 1.5), color=CLUSTER_COLOR)
    style_axes(axes[0], theme, spatial=False, title="cluster size", xlabel="cluster size [pixels]", ylabel="count")

    axes[1].hist(clusters["charge_sum"], bins=30, color=CLUSTER_COLOR)
    style_axes(axes[1], theme, spatial=False, title="cluster charge", xlabel="cluster charge", ylabel="count")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def plot_residual(
    clusters: pd.DataFrame,
    truth: pd.DataFrame,
    detector: DetectorConfig,
    types: tuple[str, ...] = ("charge",),
    axis: tuple[str, ...] = ("x", "y"),
    bins: int = 50,
    hits: pd.DataFrame | None = None,
    contributions: pd.DataFrame | None = None,
    theme: Theme | None = None,
):
    """Histogram(s) of reconstructed-centroid minus true-position residuals,
    one subplot per requested axis.

    types: one or both of "charge" (charge-weighted centroid, the default)
        and "digital" (unweighted centroid) — pass both to overlay them for
        a direct comparison of the two reconstruction schemes.
    axis: one or both of "x", "y".
    hits, contributions: if both given, truth particles are matched to
        clusters via the exact charge-contribution link instead of
        nearest-position (see `analysis.match_clusters_to_truth`) — the
        correct choice for overlapping multi-particle events.

    The true position is each truth particle's own track evaluated at the
    sensor's mid-thickness plane (see `sim.geometry.true_center_position`).
    """
    residuals_by_type = {
        t: compute_residuals(clusters, truth, detector, type=t, hits=hits, contributions=contributions)
        for t in types
    }

    fig, axes = plt.subplots(1, len(axis), figsize=(5.5 * len(axis), 4), squeeze=False)
    axes = axes[0]

    for ax, a in zip(axes, axis):
        for t in types:
            values = residuals_by_type[t][f"residual_{a}_um"]
            ax.hist(
                values,
                bins=bins,
                color=CENTROID_TYPE_COLOR[t],
                alpha=0.6 if len(types) > 1 else 1.0,
                label=f"{CENTROID_TYPE_LABEL[t]} (μ={values.mean():.2f}, σ={values.std():.2f})",
            )
        ax.axvline(0.0, color="0.3", linewidth=1, linestyle="--", zorder=1)
        style_axes(
            ax, theme, spatial=False, title=f"{a} residual",
            xlabel=f"{a} residual: reconstructed - true [um]", ylabel="count", legend=True,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig
