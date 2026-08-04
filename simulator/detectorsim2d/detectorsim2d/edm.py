"""Event data model (EDM): the transient per-event table schemas shared by
`simulate` (producer), `io` (persistence), and `vis` (consumer).

Three columnar tables, joined by `event_id` (`hits`/`deposits` also by
`particle_id`):

- **particles** — one row per simulated particle (ground truth: identity,
  vertex, direction, charge, energy, resolved bend radius)
- **hits** — one row per *sensitive* layer a particle actually crossed
  (tracker and muon system: position measurements)
- **deposits** — one row per calorimeter cell a particle put energy into
  (ECAL and HCAL: energy measurements)

The hits/deposits split is the detector's own: a tracking layer answers
"where", a calorimeter cell answers "how much".
"""

from __future__ import annotations

PARTICLES_COLUMNS = [
    "event_id",
    "particle_id",
    "species",
    "pdg",
    "x0",
    "y0",
    "phi0",
    "charge",
    "energy",
    "radius",
    "jet_id",
    "is_b_jet",
]
"""`species`/`pdg` name the particle (see detectorsim2d.species); both may be
absent/NaN for a hand-built table that only knows kinematics, in which case
the particle is treated as a bare charged stub with no calorimeter response.

`energy` is the particle's energy; in this 2D toy everything is transverse,
so it doubles as `pt` and is what `signed_radius` bends.

`radius` is NaN or infinite for a straight track, a finite signed value for
an arc (positive curls left/CCW, negative curls right/CW) -- see
detector2d.geometry.Trajectory and detectorsim2d.simulate.trajectory_for_row. It
is the radius in the *innermost* field region; with a piecewise field the
particle's actual radius changes region by region (see
detector2d.propagate)."""

"""`jet_id`/`is_b_jet` are ground truth from `gun.mode == "jets"`: which jet
axis (0-based, unique within the event) a particle was assigned to, and
whether that axis was sampled as a "b-jet" (see
`ParticleGunConfig.b_jet_fraction`). `jet_id` is -1 and `is_b_jet` is `False`
for every other gun mode, and for a hand-built table that predates jets
(pandas reads a missing value back as NaN, not -1/False, on a CSV round
trip -- callers that care should treat NaN the same as "no jet")."""

HITS_COLUMNS = [
    "event_id",
    "particle_id",
    "system",
    "layer_id",
    "hit_id",
    "x",
    "y",
    "s_local",
    "path_length",
]
"""`system` is the subsystem the layer belongs to ("tracker", "muon"), taken
straight off the layer (see detector2d.geometry.CircleLayer).

`s_local` is the hit's position along its layer, in length units (distance
from p1 for a LineLayer, arc length around the center for a CircleLayer);
`path_length` is the arc length along the particle's own trajectory to reach
the hit."""

DEPOSITS_COLUMNS = [
    "event_id",
    "particle_id",
    "system",
    "layer_id",
    "cell_id",
    "x",
    "y",
    "s_local",
    "energy",
]
"""One row per (particle, calorimeter cell) -- i.e. *truth level*: two
particles showering into the same cell give two rows, which is what makes
the table usable as ground truth for cluster-splitting exercises. Use
`detectorsim2d.response.sum_cells` for the particle-blind view a real
reconstruction would see.

`system` is "ecal"/"hcal"; `cell_id` is the azimuthal cell index within that
layer (see detector2d.calorimeter.CaloRing.cell_index); `x`/`y` and `s_local`
locate the cell center on its ring; `energy` is the energy deposited in that
cell by that particle."""
