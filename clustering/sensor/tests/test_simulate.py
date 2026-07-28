import numpy as np
import pandas as pd
import pytest

from sensor.cli import run_simulation
from sensor.io import read_run, write_run
from sensor.sim import SimConfig, simulate_events


def _base_config(**overrides) -> SimConfig:
    config = SimConfig()
    config.seed = 12345
    for path, value in overrides.items():
        section, field = path.split(".")
        setattr(getattr(config, section), field, value)
    return config


def test_perpendicular_track_hits_one_pixel():
    config = _base_config(**{"particle.angle_spread": 0.0, "detector.lorentz_slope": 0.0})
    hits, clusters, truth, contributions = run_simulation(config)

    assert len(hits) == 1
    assert len(clusters) == 1
    assert clusters.iloc[0]["n_pixels"] == 1
    assert clusters.iloc[0]["charge_sum"] == pytest.approx(1.0)
    # single particle, single pixel: its whole deposit lands in that one pixel
    assert len(contributions) == 1
    assert contributions.iloc[0]["particle_id"] == 0
    assert contributions.iloc[0]["charge"] == pytest.approx(1.0)


def test_lorentz_drift_elongates_cluster_in_x():
    baseline = _base_config(**{"particle.angle_spread": 0.0, "detector.lorentz_slope": 0.0})
    drifted = _base_config(**{"particle.angle_spread": 0.0, "detector.lorentz_slope": 1.0})
    # keep the same vertex/direction draw for both by using the same seed
    _, clusters_baseline, _, _ = run_simulation(baseline)
    _, clusters_drifted, _, _ = run_simulation(drifted)

    assert clusters_baseline.iloc[0]["x_span_pixels"] == 1
    # thickness=150um, lorentz_slope=1.0 -> 150um shift -> 6 pixels at pitch_x=25um
    assert clusters_drifted.iloc[0]["x_span_pixels"] > 1


def test_cluster_columns_include_digital_centroid():
    config = _base_config(**{"particle.angle_spread": 0.0, "detector.lorentz_slope": 0.0})
    _, clusters, _, _ = run_simulation(config)

    # single-pixel cluster: charge-weighted and digital centroids coincide
    assert clusters.iloc[0]["x_centroid_digital_um"] == pytest.approx(clusters.iloc[0]["x_centroid_um"])
    assert clusters.iloc[0]["y_centroid_digital_um"] == pytest.approx(clusters.iloc[0]["y_centroid_um"])


def test_readout_threshold_removes_low_charge_clusters():
    config = _base_config(**{"particle.angle_spread": 0.0, "detector.lorentz_slope": 0.0})
    config.readout_threshold = 1.5  # above the ~1.0 charge deposited by a perpendicular track
    hits, clusters, _, _ = run_simulation(config)

    assert hits.empty
    assert clusters.empty


def test_contributions_sum_to_the_combined_grid():
    # contributions is the per-particle decomposition of the same combined
    # grid digitize_events later reads from; the two must agree exactly.
    config = _base_config(**{"multi.n_particles": 3})
    grids, _, contributions = simulate_events(config, rng=np.random.default_rng(config.seed))

    grid = grids[0]
    for (ix, iy), group in contributions.groupby(["ix", "iy"]):
        assert group["charge"].sum() == pytest.approx(grid[int(ix), int(iy)])


def test_multi_particle_event_produces_multiple_truth_rows():
    config = _base_config(**{"multi.n_particles": 5})
    hits, clusters, truth, contributions = run_simulation(config)

    assert len(truth) == 5
    assert (truth["event_id"] == 0).all()
    assert sorted(truth["particle_id"]) == list(range(5))
    # every truth particle deposited charge somewhere
    assert sorted(contributions["particle_id"].unique()) == list(range(5))


def test_n_particles_fixed_mode_is_always_exact():
    config = _base_config(**{"multi.n_particles": 4})
    config.multi.n_particles_mode = "fixed"
    config.n_events = 20
    _, _, truth, _ = run_simulation(config)

    assert (truth.groupby("event_id").size() == 4).all()


def test_n_particles_uniform_mode_stays_within_range_and_varies():
    config = _base_config()
    config.multi.n_particles_mode = "uniform"
    config.multi.n_particles_min = 1
    config.multi.n_particles = 3
    config.n_events = 200
    _, _, truth, _ = run_simulation(config)

    counts = truth.groupby("event_id").size()
    assert counts.min() >= 1
    assert counts.max() <= 3
    assert counts.nunique() > 1  # varies across events, not accidentally constant


@pytest.mark.parametrize("fmt", ["csv", "arrow"])
def test_write_read_round_trip(tmp_path, fmt):
    config = _base_config()
    config.n_events = 10
    hits, clusters, truth, contributions = run_simulation(config)

    write_run(tmp_path, fmt, hits, clusters, truth, contributions)
    hits2, clusters2, truth2, contributions2 = read_run(tmp_path, fmt)

    pd.testing.assert_frame_equal(hits, hits2, check_dtype=False)
    pd.testing.assert_frame_equal(clusters, clusters2, check_dtype=False)
    pd.testing.assert_frame_equal(truth, truth2, check_dtype=False)
    pd.testing.assert_frame_equal(contributions, contributions2, check_dtype=False)
