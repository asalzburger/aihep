import pandas as pd
import pytest
from detector2d.geometry import LineLayer

from tracksim2d.config import FieldConfig, ParticleGunConfig, SimConfig
from tracksim2d.io import read_run, write_run
from tracksim2d.simulate import simulate_events


@pytest.mark.parametrize("fmt", ["csv", "arrow"])
def test_write_read_round_trip(tmp_path, fmt):
    config = SimConfig(
        layers=[LineLayer(layer_id=0, p1=(10.0, -50.0), p2=(10.0, 50.0))],
        magnetic_field=FieldConfig(bz=0.5),
        gun=ParticleGunConfig(n_particles=3),
        n_events=4,
        seed=99,
    )
    particles, hits = simulate_events(config)

    write_run(tmp_path, fmt, particles, hits)
    particles2, hits2 = read_run(tmp_path, fmt)

    pd.testing.assert_frame_equal(particles, particles2, check_dtype=False)
    pd.testing.assert_frame_equal(hits, hits2, check_dtype=False)
