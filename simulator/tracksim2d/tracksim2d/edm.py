"""Event data model (EDM): the transient per-event table schemas shared by
`simulate` (producer), `io` (persistence), and `vis` (consumer).

Two columnar tables, joined by `event_id` (`hits` also by `particle_id`):

- **particles** — one row per simulated particle (ground truth: vertex,
  direction, charge, resolved bend radius)
- **hits** — one row per layer a particle actually crossed
"""

from __future__ import annotations

PARTICLES_COLUMNS = ["event_id", "particle_id", "x0", "y0", "phi0", "charge", "radius"]
"""radius is NaN or infinite for a straight track, a finite signed value for
an arc (positive curls left/CCW, negative curls right/CW) -- see
detector2d.geometry.Trajectory and tracksim2d.simulate.trajectory_for_row."""

HITS_COLUMNS = ["event_id", "particle_id", "layer_id", "hit_id", "x", "y", "s_local", "path_length"]
"""s_local is the hit's position along its layer, in length units (distance
from p1 for a LineLayer, arc length around the center for a CircleLayer);
path_length is the arc length along the particle's own trajectory to reach
the hit."""
