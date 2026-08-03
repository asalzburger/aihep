import pandas as pd
import pytest
from detector2d.calorimeter import build_calo_stack
from detector2d.geometry import CircleLayer, LineLayer

from detectorsim2d.config import FieldConfig, ParticleGunConfig, SimConfig
from detectorsim2d.edm import DEPOSITS_COLUMNS
from detectorsim2d.io import read_deposits, read_run, write_run
from detectorsim2d.response import ResponseConfig
from detectorsim2d.simulate import simulate_events


@pytest.mark.parametrize("fmt", ["csv", "arrow"])
def test_write_read_round_trip(tmp_path, fmt):
    config = SimConfig(
        layers=[LineLayer(layer_id=0, p1=(10.0, -50.0), p2=(10.0, 50.0))],
        magnetic_field=FieldConfig(bz=0.5),
        gun=ParticleGunConfig(n_particles=3),
        n_events=4,
        seed=99,
    )
    particles, hits, _deposits = simulate_events(config)

    write_run(tmp_path, fmt, particles, hits)
    particles2, hits2 = read_run(tmp_path, fmt)

    pd.testing.assert_frame_equal(particles, particles2, check_dtype=False)
    pd.testing.assert_frame_equal(hits, hits2, check_dtype=False)


def _calo_config():
    return SimConfig(
        layers=[CircleLayer(layer_id=0, center=(0.0, 0.0), radius=100.0)]
        + build_calo_stack(100, 210.0, 3, 30.0, 64, system="ecal")
        + build_calo_stack(200, 300.0, 2, 80.0, 32, system="hcal"),
        magnetic_field=FieldConfig(bz=0.0),  # straight tracks, guaranteed to reach the calo
        gun=ParticleGunConfig(
            n_particles=3, species=("electron", "photon", "pi+"), pt_min=100.0, pt_max=200.0
        ),
        response=ResponseConfig(stochastic={}),
        n_events=2,
        seed=99,
    )


@pytest.mark.parametrize("fmt", ["csv", "arrow"])
def test_deposits_round_trip(tmp_path, fmt):
    particles, hits, deposits = simulate_events(_calo_config())
    assert len(deposits) > 0

    write_run(tmp_path, fmt, particles, hits, deposits)
    deposits2 = read_deposits(tmp_path, fmt)
    pd.testing.assert_frame_equal(deposits, deposits2, check_dtype=False)


@pytest.mark.parametrize("fmt", ["csv", "arrow"])
def test_read_run_stays_a_two_tuple_even_with_deposits_written(tmp_path, fmt):
    """`tracking/graphs` unpacks exactly two values from read_run, so adding a
    calorimeter must not change its arity."""
    particles, hits, deposits = simulate_events(_calo_config())
    write_run(tmp_path, fmt, particles, hits, deposits)

    result = read_run(tmp_path, fmt)
    assert len(result) == 2
    pd.testing.assert_frame_equal(result[1], hits, check_dtype=False)


@pytest.mark.parametrize("fmt", ["csv", "arrow"])
def test_reading_deposits_from_a_tracker_only_run_gives_an_empty_table(tmp_path, fmt):
    particles, hits, _deposits = simulate_events(
        SimConfig(layers=[LineLayer(layer_id=0, p1=(10.0, -50.0), p2=(10.0, 50.0))], n_events=1)
    )
    write_run(tmp_path, fmt, particles, hits)  # no deposits argument

    deposits = read_deposits(tmp_path, fmt)
    assert len(deposits) == 0
    assert list(deposits.columns) == DEPOSITS_COLUMNS
    assert not (tmp_path / f"deposits.{fmt}").exists()
