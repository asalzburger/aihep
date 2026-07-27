"""Run `hopfield_tracking` on every event in resources/denby_random_events.csv
(written by generate_random_events.py) and report perfect/confused rates by
track multiplicity -- the quantitative, systematic analogue of the paper's
fig. 8(a) (a hand-picked event that reconstructs perfectly) vs. fig. 8(b)
(one that doesn't): does reconstruction quality degrade as more tracks
compete for the same hits, the way the paper describes qualitatively?

Writes resources/denby_sweep_results.csv (one row per event) and prints a
per-multiplicity summary.

Run from this directory (`tracking/denby/python/`) with the project venv:

    ../.venv/bin/python run_hopfield_sweep.py
"""

from __future__ import annotations

import pandas as pd
from generate_random_events import EVENTS_CSV
from render_event import RESOURCES

from hopfield_tracking.cli import run

RESULTS_CSV = RESOURCES / "denby_sweep_results.csv"
SEED = 0


def main() -> None:
    events = pd.read_csv(EVENTS_CSV)

    rows = []
    for event_id, event_hits in events.groupby("event_id"):
        multiplicity = int(event_hits["multiplicity"].iloc[0])
        hits = event_hits[["particle_id", "layer_id", "x", "y"]].reset_index(drop=True)
        try:
            _, history, _, score = run(hits, seed=SEED)
        except ValueError:
            continue  # not enough hits on some track to calibrate R_c; skip
        rows.append(
            dict(
                event_id=event_id,
                multiplicity=multiplicity,
                n_iterations=len(history) - 1,
                **score,
            )
        )

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_CSV, index=False)
    print(f"wrote {RESULTS_CSV} ({len(results)} events)")

    summary = results.groupby("multiplicity").agg(
        n_events=("event_id", "size"),
        perfect_rate=("perfect", "mean"),
        mean_exact_matches=("n_exact_matches", "mean"),
        mean_true_tracks=("n_true_tracks", "mean"),
    )
    print(summary)


if __name__ == "__main__":
    main()
