"""Run the Hopfield track finder (`hopfield_tracking`) on Denby's own
4-track event, reproducing the paper's fig. 8(a) end to end: starting from
a random state, watch the network relax into the same 4 tracks last
session's `render_event.py` drew directly from the fitted parameters.

Reads resources/denby_layers.csv + resources/denby_event.csv (same as
render_event.py), computes hits via tracksim2d exactly as any other
tracksim2d consumer would, and writes:

- resources/denby_hopfield_hits.csv -- the plain (x, y, particle_id,
  layer_id) table `hopfield_tracking` actually consumes (no other
  detector2d/tracksim2d-specific columns), for reference/reuse. `layer_id`
  is what lets it exclude same-layer candidate segments -- two hits on the
  same detector plane can never be the two ends of one physical track
  segment, see hopfield_tracking/network.py::build_segments.
- resources/denby_hopfield_fig8.png -- the fig.-8-style multi-panel figure.

Run from this directory (`tracking/denby/python/`) with the project venv
(which needs `hopfield_tracking` installed alongside detector2d/tracksim2d,
see ../README.md):

    ../.venv/bin/python run_hopfield_on_denby_event.py
"""

from __future__ import annotations

import pandas as pd
from render_event import RESOURCES, load_layers

from hopfield_tracking.cli import run
from hopfield_tracking.vis import plot_iterations
from tracksim2d.simulate import hits_for_particles

HITS_CSV = RESOURCES / "denby_hopfield_hits.csv"
FIGURE_PNG = RESOURCES / "denby_hopfield_fig8.png"

SEED = 3  # typical of the mean outcome across seeds; see hopfield_tracking/README.md


def main() -> None:
    layers = load_layers()
    particles = pd.read_csv(RESOURCES / "denby_event.csv")
    hits = hits_for_particles(particles, layers)

    hopfield_hits = hits[["particle_id", "layer_id", "x", "y"]].reset_index(drop=True)
    hopfield_hits.to_csv(HITS_CSV, index=False)
    print(f"wrote {HITS_CSV} ({len(hopfield_hits)} hits, {hopfield_hits['particle_id'].nunique()} true tracks)")

    segments, history, chains, score = run(hopfield_hits, seed=SEED)
    print(f"{len(segments)} candidate segments")
    print(f"converged in {len(history) - 1} iteration(s), final energy {history.energy[-1]:.4f}")
    print(f"found {len(chains)} chain(s)")
    print(f"score: {score}")

    xy = hopfield_hits[["x", "y"]].to_numpy()
    fig = plot_iterations(xy, segments, history)
    fig.savefig(FIGURE_PNG, dpi=150, bbox_inches="tight")
    print(f"wrote {FIGURE_PNG}")


if __name__ == "__main__":
    main()
