"""Illustrative reproduction of Denby (1988) fig. 9: Darbo & Heck's
"contiguity trigger" for the Delphi TPC -- a *much* simpler, non-Hopfield
idea, included here only as a contrast to the rest of this package.

Hits are mapped onto a rectangular "Image Memory" (IM) grid (row =
transverse position, column = depth). A fixed *contiguity mask* connects
each node to its vertical neighbors in the same column, and to its
right-hand neighbor's vertical neighbors -- i.e. it tolerates at most one
row of drift per column. A track is "found" if it forms a continuous path
of "on" nodes across the grid under that mask.

Because the mask has one fixed, hardwired tolerance, it finds a straighter
(higher-pT) track and misses a more curved (lower-pT) one -- exactly the
paper's own point (fig. 9/10), and the reason section 9 calls a "true"
neural approach more general: curvature just doesn't matter to a Hopfield
network the way it matters to a fixed mask.

Run (only needs matplotlib):
    ../.venv/bin/python examples/contiguity_mask_demo.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

N_COLUMNS = 20
N_ROWS = 21
CENTER_ROW = N_ROWS // 2
MAX_DRIFT = 1  # the mask's fixed tolerance: connects a node to same-column
# neighbors, and to its right-hand neighbor's vertical neighbors -- i.e. at
# most 1 row of drift per column is "contiguous".

OUTPUT_PNG = Path(__file__).with_name("contiguity_mask_demo.png")


def track_rows(radius: float, n_columns: int = N_COLUMNS, center_row: int = CENTER_ROW) -> list[int]:
    """Row occupied at each column for a circular arc of the given radius
    (in column units), starting at the center row heading across the grid.
    Smaller radius = more curved = more row-drift per column."""
    rows = []
    for col in range(n_columns):
        dy = radius - np.sqrt(max(radius**2 - col**2, 0.0))
        rows.append(int(round(center_row - dy)))
    return rows


def max_row_drift(rows: list[int]) -> int:
    return max((abs(b - a) for a, b in zip(rows[:-1], rows[1:])), default=0)


def found_by_mask(rows: list[int], max_drift: int = MAX_DRIFT) -> bool:
    """Whether `rows` is a continuous path under the fixed contiguity mask."""
    return max_row_drift(rows) <= max_drift


def plot_demo(tracks: dict[str, list[int]], max_drift: int = MAX_DRIFT):
    fig, ax = plt.subplots(figsize=(6, 6))
    for row in range(N_ROWS):
        ax.plot(range(N_COLUMNS), [row] * N_COLUMNS, ".", color="0.85", markersize=3, zorder=1)

    for label, rows in tracks.items():
        found = found_by_mask(rows, max_drift)
        color = "#0072B2" if found else "#D55E00"
        marker_label = f"{label} ({'found' if found else 'missed'})"
        ax.plot(range(N_COLUMNS), rows, marker="o", markersize=4, color=color, label=marker_label, zorder=2)

    ax.invert_yaxis()
    ax.set_xlabel("column (depth)")
    ax.set_ylabel("row (transverse position)")
    ax.set_title(f"Contiguity mask trigger (max row-drift = {max_drift})")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def main() -> None:
    tracks = {
        "pT=2.0 (straighter, R=60)": track_rows(radius=60.0),
        "pT=1.0 (curved, R=15)": track_rows(radius=15.0),
    }
    for label, rows in tracks.items():
        print(f"{label}: max row-drift={max_row_drift(rows)}, found={found_by_mask(rows)}")

    fig = plot_demo(tracks)
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
