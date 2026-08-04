"""Event data model (EDM): the transient per-event table schemas produced by
`reconstruct` (producer) and consumed by `io`/downstream examples.

Two columnar tables, joined by `event_id`/`particle_id` back to the
`detectorsim2d` particles table they were reconstructed from:

- **tracks** -- one row per charged particle (including muons -- see the
  module docstring of :mod:`detectorreco2d.reconstruct`), smeared `d0`/
  `phi0`/`pt`.
- **clusters** -- one row per showering (EM/hadronic) particle, smeared
  total calorimeter energy.

Both carry `jet_id`/`is_b_jet` straight through from the particles table as
*truth* context (see `detectorsim2d.edm.PARTICLES_COLUMNS`) -- useful for
validating/labeling a flavor-tagging exercise, not something a real
reconstruction would know. Likewise `species`: a real tracker/calorimeter
does not hand you particle identity for free, but carrying it through here
saves re-deriving "was this a muon" from muon-system hits, which is future
work, not this package's.
"""

from __future__ import annotations

TRACKS_COLUMNS = [
    "event_id",
    "particle_id",
    "jet_id",
    "is_b_jet",
    "species",
    "charge",
    "d0_true",
    "d0",
    "phi0_true",
    "phi0",
    "pt_true",
    "pt",
]
"""`d0` is the signed transverse impact parameter (see
`detector2d.geometry.Trajectory.d0`); `phi0` the initial direction; `pt` the
transverse momentum (== `energy` in this 2D toy, see
`detectorsim2d.edm.PARTICLES_COLUMNS`). The `_true` columns are the
unsmeared truth values the smeared ones were drawn around -- kept side by
side so a resolution/pull plot needs no join back to the particles table."""

CLUSTERS_COLUMNS = [
    "event_id",
    "particle_id",
    "jet_id",
    "is_b_jet",
    "species",
    "energy_true",
    "energy",
]
"""One row per EM/hadronic-showering particle (a muon's MIP trail is not a
"cluster" -- see the module docstring of `detectorreco2d.reconstruct`);
`energy_true` sums that particle's truth `detectorsim2d` calorimeter
deposits (across every cell/layer/system it hit), `energy` is that total
after the cluster-energy resolution smear."""
