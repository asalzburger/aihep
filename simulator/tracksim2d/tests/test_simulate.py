import math

import numpy as np
import pandas as pd
import pytest
from detector2d.geometry import CircleLayer, LineLayer

from tracksim2d.config import FieldConfig, ParticleGunConfig, SimConfig
from tracksim2d.edm import PARTICLES_COLUMNS
from tracksim2d.simulate import hits_for_particles, sample_particles, simulate_events


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
    particles, hits = simulate_events(config)
    assert len(particles) == 15  # 3 events * 5 particles
    assert np.isinf(particles["radius"]).all()  # bz=0 -> straight (signed_radius returns inf)
    assert len(hits) == 15  # every straight, shallow-angle particle crosses the one layer
    assert set(hits["event_id"]) == {0, 1, 2}
