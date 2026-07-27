import pandas as pd
import pytest

from sensor.cli import run_simulation
from sensor.io import read_run, write_run
from sensor.sim import SimConfig


def _base_config(**overrides) -> SimConfig:
    config = SimConfig()
    config.seed = 12345
    for path, value in overrides.items():
        section, field = path.split(".")
        setattr(getattr(config, section), field, value)
    return config


def test_perpendicular_track_hits_one_pixel():
    config = _base_config(**{"particle.angle_spread": 0.0, "detector.lorentz_slope": 0.0})
    hits, clusters, truth = run_simulation(config)

    assert len(hits) == 1
    assert len(clusters) == 1
    assert clusters.iloc[0]["n_pixels"] == 1
    assert clusters.iloc[0]["charge_sum"] == pytest.approx(1.0)


def test_lorentz_drift_elongates_cluster_in_x():
    baseline = _base_config(**{"particle.angle_spread": 0.0, "detector.lorentz_slope": 0.0})
    drifted = _base_config(**{"particle.angle_spread": 0.0, "detector.lorentz_slope": 1.0})
    # keep the same vertex/direction draw for both by using the same seed
    _, clusters_baseline, _ = run_simulation(baseline)
    _, clusters_drifted, _ = run_simulation(drifted)

    assert clusters_baseline.iloc[0]["x_span_pixels"] == 1
    # thickness=150um, lorentz_slope=1.0 -> 150um shift -> 6 pixels at pitch_x=25um
    assert clusters_drifted.iloc[0]["x_span_pixels"] > 1


def test_multi_particle_event_produces_multiple_truth_rows():
    config = _base_config(**{"multi.n_particles": 5})
    hits, clusters, truth = run_simulation(config)

    assert len(truth) == 5
    assert (truth["event_id"] == 0).all()
    assert sorted(truth["particle_id"]) == list(range(5))


@pytest.mark.parametrize("fmt", ["csv", "arrow"])
def test_write_read_round_trip(tmp_path, fmt):
    config = _base_config()
    config.n_events = 10
    hits, clusters, truth = run_simulation(config)

    write_run(tmp_path, fmt, hits, clusters, truth)
    hits2, clusters2, truth2 = read_run(tmp_path, fmt)

    pd.testing.assert_frame_equal(hits, hits2, check_dtype=False)
    pd.testing.assert_frame_equal(clusters, clusters2, check_dtype=False)
    pd.testing.assert_frame_equal(truth, truth2, check_dtype=False)
