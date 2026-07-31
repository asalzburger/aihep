import math

import numpy as np
import pandas as pd
import pytest
from detector2d.barrel import build_barrel_modules, module_reach
from detector2d.geometry import CircleLayer, LineLayer, Trajectory

from tracksim2d.config import FieldConfig, ParticleGunConfig, SimConfig
from tracksim2d.edm import PARTICLES_COLUMNS
from tracksim2d.simulate import boundary_crossing_s, hits_for_particles, sample_particles, simulate_events


def _particle_row(**overrides):
    row = dict(event_id=0, particle_id=0, x0=0.0, y0=0.0, phi0=0.0, charge=1.0, radius=math.nan)
    row.update(overrides)
    return row


def test_straight_particle_hits_expected_line_layers():
    particles = pd.DataFrame([_particle_row()], columns=PARTICLES_COLUMNS)
    layers = [
        LineLayer(layer_id=0, p1=(10.0, -5.0), p2=(10.0, 5.0)),
        LineLayer(layer_id=1, p1=(20.0, -5.0), p2=(20.0, 5.0)),
    ]
    hits = hits_for_particles(particles, layers)
    assert len(hits) == 2
    assert list(hits.sort_values("path_length")["layer_id"]) == [0, 1]
    assert list(hits.sort_values("path_length")["x"]) == pytest.approx([10.0, 20.0])


def test_curved_particle_hits_circle_layer():
    # trajectory circle: center (0,0) r=5 (particle starts at (0,-5) heading +x).
    # layer: circle center (6,0) r=5 -- classic 3-4-5 intersection at (3, +-4).
    particles = pd.DataFrame([_particle_row(x0=0.0, y0=-5.0, phi0=0.0, radius=5.0)], columns=PARTICLES_COLUMNS)
    layers = [CircleLayer(layer_id=0, center=(6.0, 0.0), radius=5.0)]
    hits = hits_for_particles(particles, layers)
    assert len(hits) == 1  # only the earliest crossing is kept
    x, y = hits.iloc[0]["x"], hits.iloc[0]["y"]
    assert (round(x, 6), round(y, 6)) in {(3.0, -4.0), (3.0, 4.0)}
    assert math.hypot(x - 6.0, y - 0.0) == pytest.approx(5.0)


def test_particle_with_nan_radius_is_straight():
    particles = pd.DataFrame([_particle_row(radius=math.nan)], columns=PARTICLES_COLUMNS)
    layers = [LineLayer(layer_id=0, p1=(10.0, -5.0), p2=(10.0, 5.0))]
    hits = hits_for_particles(particles, layers)
    assert len(hits) == 1
    assert hits.iloc[0]["x"] == pytest.approx(10.0)
    assert hits.iloc[0]["y"] == pytest.approx(0.0)


def test_detailed_barrel_overlap_produces_a_genuine_double_hit():
    """A straight particle aimed into the angular overlap between two
    neighboring barrel modules crosses both -- hits_for_particles needs no
    special-casing for this: it just loops over every module object (both
    happen to share layer_id=0) and keeps whichever ones the trajectory
    actually crosses."""
    radius, half_length, tilt_deg, overlap_fraction = 68.0, 4.0, 10.0, 0.15
    tilt = math.radians(tilt_deg)
    modules = build_barrel_modules(
        layer_id=0, radius=radius, half_length=half_length, tilt=tilt, overlap_fraction=overlap_fraction
    )
    n = len(modules)
    delta = 2.0 * math.pi / n
    reach = module_reach(radius, 2 * half_length, tilt)
    phi_in_overlap = (delta + reach) / 2.0  # inside module 0's overlap with module 1

    particles = pd.DataFrame([_particle_row(phi0=phi_in_overlap)], columns=PARTICLES_COLUMNS)
    hits = hits_for_particles(particles, modules)

    assert len(hits) == 2
    assert set(hits["layer_id"]) == {0}
    # both hits belong to the same physical crossing, not two unrelated ones
    assert (hits["path_length"].max() - hits["path_length"].min()) < 2 * half_length


def test_boundary_crossing_s_is_none_without_a_boundary():
    trajectory = Trajectory(x0=0.0, y0=0.0, phi0=0.0, radius=None)
    assert boundary_crossing_s(trajectory, None) is None


def test_boundary_crossing_s_finds_straight_track_crossing():
    trajectory = Trajectory(x0=0.0, y0=0.0, phi0=0.0, radius=None)
    assert boundary_crossing_s(trajectory, 25.0) == pytest.approx(25.0)


def test_boundary_crossing_s_none_when_trajectory_never_reaches_it():
    # a tight loop of radius 5 (max reach 2*5=10 from the origin) never
    # reaches a boundary of radius 50
    trajectory = Trajectory(x0=0.0, y0=0.0, phi0=0.0, radius=5.0)
    assert boundary_crossing_s(trajectory, 50.0) is None


def test_hits_for_particles_drops_the_loop_back_crossing_beyond_the_boundary():
    # a curved trajectory starting at the origin always crosses a concentric
    # circular layer twice (see detector2d.barrel's module docstring): once
    # on the way out, and again after looping back through on the way in.
    # radius=10 -> max reach 20, so a layer at radius=18 is only reachable
    # via that second, loop-back crossing (its "first" and only crossing
    # happens late, close to the half-loop point) -- with no boundary, that
    # crossing is still kept; with a boundary tighter than it, it must be
    # dropped instead of drawing the particle back in from outside the
    # tracker volume.
    particles = pd.DataFrame([_particle_row(radius=10.0)], columns=PARTICLES_COLUMNS)
    layers = [CircleLayer(layer_id=0, center=(0.0, 0.0), radius=18.0)]

    hits_unbounded = hits_for_particles(particles, layers, tracker_boundary=None)
    assert len(hits_unbounded) == 1

    hits_bounded = hits_for_particles(particles, layers, tracker_boundary=15.0)
    assert len(hits_bounded) == 0


def test_hits_for_particles_keeps_hits_within_the_boundary():
    particles = pd.DataFrame([_particle_row()], columns=PARTICLES_COLUMNS)  # straight, along +x
    layers = [LineLayer(layer_id=0, p1=(10.0, -5.0), p2=(10.0, 5.0))]
    hits = hits_for_particles(particles, layers, tracker_boundary=50.0)
    assert len(hits) == 1
    assert hits.iloc[0]["x"] == pytest.approx(10.0)


def test_simulate_events_passes_tracker_boundary_through_config():
    # pt=1.0, k=0.2998 (default), bz chosen so the resolved radius is
    # exactly 10.0 -- same loop-back setup as the test above, but exercised
    # through the full simulate_events(config) path rather than calling
    # hits_for_particles directly.
    k = FieldConfig().k
    bz = 1.0 / (k * 10.0)
    layers = [CircleLayer(layer_id=0, center=(0.0, 0.0), radius=18.0)]
    config = SimConfig(
        layers=layers,
        magnetic_field=FieldConfig(bz=bz, k=k),
        gun=ParticleGunConfig(n_particles=1, phi_min=0.0, phi_max=0.0, charges=(1.0,), pt_min=1.0, pt_max=1.0),
        n_events=1,
        seed=1,
        tracker_boundary=15.0,
    )
    particles, hits, _deposits = simulate_events(config)
    assert particles.iloc[0]["radius"] == pytest.approx(10.0)
    assert len(hits) == 0  # the only crossing is the loop-back one, dropped by the boundary


def test_sample_particles_reproducible_with_same_seed():
    config = SimConfig(
        layers=[],
        magnetic_field=FieldConfig(bz=1.0),
        gun=ParticleGunConfig(n_particles=4, pt_min=1.0, pt_max=5.0),
        n_events=1,
        seed=7,
    )
    rows_a = sample_particles(np.random.default_rng(config.seed), config, event_id=0)
    rows_b = sample_particles(np.random.default_rng(config.seed), config, event_id=0)
    assert rows_a == rows_b
    assert len(rows_a) == 4


def test_simulate_events_end_to_end():
    layers = [LineLayer(layer_id=0, p1=(10.0, -50.0), p2=(10.0, 50.0), pitch=1.0)]
    config = SimConfig(
        layers=layers,
        magnetic_field=FieldConfig(bz=0.0),  # no field -> straight tracks, guaranteed to hit
        gun=ParticleGunConfig(n_particles=5, phi_min=-0.1, phi_max=0.1),
        n_events=3,
        seed=42,
    )
    particles, hits, _deposits = simulate_events(config)
    assert len(particles) == 15  # 3 events * 5 particles
    assert np.isinf(particles["radius"]).all()  # bz=0 -> straight (signed_radius returns inf)
    assert len(hits) == 15  # every straight, shallow-angle particle crosses the one layer
    assert set(hits["event_id"]) == {0, 1, 2}
