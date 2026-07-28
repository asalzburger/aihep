"""Matplotlib visualization of simulated events and clusters.

Static, print-style figures (not the interactive-dashboard case the dataviz
skill mostly targets) but the same principles apply where they're relevant:
a perceptually-uniform sequential colormap for charge (never a rainbow map),
one fixed accent color per categorical overlay (cluster boxes vs. truth
tracks), and a legend whenever more than one series is on the plot.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

from .analysis import CENTROID_COLUMNS, compute_residuals
from .sim import DetectorConfig, charge_endpoints, true_center_position

CLUSTER_COLOR = "#DA4C4C"
TRUTH_COLOR = "#2E86DE"
CHARGE_CMAP = "YlOrRd"  # sequential: low charge yellow -> high charge red
DIGITAL_ON_COLOR = "#C0392B"
GRID_COLOR = "0.85"
DEFAULT_READOUT_THRESHOLD = 0.15  # pixels with charge <= this are not "read out"

# Centroid-type identity, shared between the per-event overlay (plot_event)
# and the residual/summary plots: charge-weighted reuses the existing
# cluster-box red, digital gets its own hue so the two never collide.
CENTROID_TYPE_COLOR = {"charge": CLUSTER_COLOR, "digital": "#E67E22"}
CENTROID_TYPE_LABEL = {"charge": "charge-weighted", "digital": "digital"}
CENTROID_TYPE_MARKER = {"charge": "D", "digital": "s"}
TRUE_POSITION_MARKER = "*"


def _clip_window(center: int, size: int, n_max: int) -> tuple[int, int]:
    """size-pixel window centered on `center`, shifted to fit inside [0, n_max)."""
    lo = center - size // 2
    lo = max(0, min(lo, max(n_max - size, 0)))
    return lo, lo + size


def _zoom_pixel_bounds(
    event_clusters: pd.DataFrame, detector: DetectorConfig, zoom: tuple[int, int]
) -> tuple[int, int, int, int]:
    """(ix_lo, ix_hi, iy_lo, iy_hi) window of `zoom` pixels centered on the
    event's largest cluster (falls back to the grid center if there is
    none)."""
    nx, ny = zoom
    if not event_clusters.empty:
        target = event_clusters.sort_values("charge_sum", ascending=False).iloc[0]
        center_ix = int(target["x_centroid_um"] // detector.pitch_x_um)
        center_iy = int(target["y_centroid_um"] // detector.pitch_y_um)
    else:
        center_ix, center_iy = detector.n_pixels_x // 2, detector.n_pixels_y // 2
    ix_lo, ix_hi = _clip_window(center_ix, nx, detector.n_pixels_x)
    iy_lo, iy_hi = _clip_window(center_iy, ny, detector.n_pixels_y)
    return ix_lo, ix_hi, iy_lo, iy_hi


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
):
    """Plot one event's pixel grid, cluster boxes, and truth tracks.

    zoom: if given, (nx, ny) pixel window shown, centered on the event's
        largest cluster (by charge) instead of the full sensor.
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
        ix_lo, ix_hi, iy_lo, iy_hi = _zoom_pixel_bounds(event_clusters, detector, zoom)
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
        fig.colorbar(mesh, ax=ax, label="charge", fraction=0.046, pad=0.04)

    if grid:
        # Explicit line segments (not matplotlib's tick/grid machinery) so
        # every pixel border in view is guaranteed to be drawn, regardless
        # of how many pixels are visible.
        xs_in_view = x_edges[(x_edges >= x_view[0]) & (x_edges <= x_view[1])]
        ys_in_view = y_edges[(y_edges >= y_view[0]) & (y_edges <= y_view[1])]
        ax.vlines(xs_in_view, y_view[0], y_view[1], color=GRID_COLOR, linewidth=0.6, zorder=1)
        ax.hlines(ys_in_view, x_view[0], x_view[1], color=GRID_COLOR, linewidth=0.6, zorder=1)

    for _, ix, iy in event_hits.groupby("cluster_id")[["ix", "iy"]].agg(list).itertuples():
        x0, x1 = min(ix) * detector.pitch_x_um, (max(ix) + 1) * detector.pitch_x_um
        y0, y1 = min(iy) * detector.pitch_y_um, (max(iy) + 1) * detector.pitch_y_um
        ax.add_patch(
            Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=CLUSTER_COLOR, linewidth=2)
        )

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
        ax.plot(*p0, marker="o", color=TRUTH_COLOR, markersize=4, zorder=3)
        ax.plot(*p1, marker="x", color=TRUTH_COLOR, markersize=6, zorder=3)

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
    ax.set_xlabel("x [um]")
    ax.set_ylabel("y [um]")
    ax.set_aspect("equal")
    mode = " [digital]" if digital else ""
    ax.set_title(
        f"event {event_id}{mode}: {len(event_hits)} hit pixels "
        f"(readout > {readout_threshold}), {event_hits['cluster_id'].nunique()} cluster(s)"
    )

    handles = [
        plt.Line2D([0], [0], color=CLUSTER_COLOR, linewidth=2, label="cluster bounding box"),
        plt.Line2D([0], [0], color=TRUTH_COLOR, linewidth=1.5, marker="o", markersize=4, label="truth track"),
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
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=8)

    fig.tight_layout()
    return fig


def plot_cluster_summary(clusters: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(clusters["n_pixels"], bins=np.arange(0.5, clusters["n_pixels"].max() + 1.5), color=CLUSTER_COLOR)
    axes[0].set_xlabel("cluster size [pixels]")
    axes[0].set_ylabel("count")
    axes[0].set_title("cluster size")

    axes[1].hist(clusters["charge_sum"], bins=30, color=CLUSTER_COLOR)
    axes[1].set_xlabel("cluster charge")
    axes[1].set_ylabel("count")
    axes[1].set_title("cluster charge")

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
        ax.set_xlabel(f"{a} residual: reconstructed - true [um]")
        ax.set_ylabel("count")
        ax.set_title(f"{a} residual")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8, frameon=True)

    fig.tight_layout()
    return fig
