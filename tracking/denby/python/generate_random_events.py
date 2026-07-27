"""Generate a batch of synthetic multi-track events (1-6 tracks each,
matching Denby's own "the network was tried on a set of simulated events
with from 1 to 6 tracks") through the harmonized Denby detector -- reusing
`tracksim2d`'s existing particle-gun simulation and `detector2d`'s field
helper rather than writing new generation code (task item 4).

This is a batch of *demonstration/evaluation* events for
`run_hopfield_sweep.py` to run the network on repeatedly -- not a
train/val/test split. The Hopfield network here has no gradient-trained
weights (`T_ij` is computed analytically per event from its own hit
geometry, same as the paper); "dataset" means a held-out set of events to
demonstrate/evaluate the one fixed algorithm on, the same role fig. 8(a)
vs. 8(b) plays in the paper (a hand-picked easy case and a hand-picked hard
one) -- except swept systematically instead of by hand. See
../hopfield_tracking/README.md.

Writes resources/denby_random_events.csv: columns event_id, multiplicity,
particle_id, layer_id, x, y.

Run from this directory (`tracking/denby/python/`) with the project venv:

    ../.venv/bin/python generate_random_events.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from render_event import RESOURCES, load_layers

from detector2d.field import signed_radius
from tracksim2d.simulate import hits_for_particles

EVENTS_CSV = RESOURCES / "denby_random_events.csv"

# Same vertex as the paper's own recovered event (tracking/denby/README.md),
# so tracks fan into the harmonized 13-layer stack the same way.
VERTEX_X, VERTEX_Y = 424.25, 894.536009
PHI_MIN, PHI_MAX = -2.3, -1.0  # radians: "mostly upward, fanning left-right", like the real event
PT_MIN, PT_MAX = 1.0, 15.0
BZ = 0.005  # with K below, gives signed radii ~650-10000px: the same order as the real event's 640-6375
K = 0.2998
CHARGES = (-1.0, 1.0)
MULTIPLICITIES = (1, 2, 3, 4, 5, 6)
EVENTS_PER_MULTIPLICITY = 15
SEED = 0


def _sample_particle(rng: np.random.Generator, event_id: int, particle_id: int) -> dict:
    phi0 = rng.uniform(PHI_MIN, PHI_MAX)
    charge = float(rng.choice(CHARGES))
    pt = rng.uniform(PT_MIN, PT_MAX)
    radius = signed_radius(pt, charge, BZ, K)
    return dict(
        event_id=event_id, particle_id=particle_id, x0=VERTEX_X, y0=VERTEX_Y, phi0=phi0, charge=charge, radius=radius
    )


def generate_events(rng: np.random.Generator, layers) -> pd.DataFrame:
    hit_rows = []
    event_id = 0
    for multiplicity in MULTIPLICITIES:
        for _ in range(EVENTS_PER_MULTIPLICITY):
            particles = pd.DataFrame([_sample_particle(rng, event_id, pid) for pid in range(multiplicity)])
            hits = hits_for_particles(particles, layers)
            for _, hit in hits.iterrows():
                hit_rows.append(
                    dict(
                        event_id=event_id,
                        multiplicity=multiplicity,
                        particle_id=int(hit["particle_id"]),
                        layer_id=int(hit["layer_id"]),
                        x=hit["x"],
                        y=hit["y"],
                    )
                )
            event_id += 1
    return pd.DataFrame(hit_rows)


def main() -> None:
    layers = load_layers()
    events = generate_events(np.random.default_rng(SEED), layers)
    events.to_csv(EVENTS_CSV, index=False)

    print(f"wrote {EVENTS_CSV}")
    print(f"{events['event_id'].nunique()} events, {len(events)} hits total")
    print(events.drop_duplicates("event_id").groupby("multiplicity").size().rename("n_events"))


if __name__ == "__main__":
    main()
