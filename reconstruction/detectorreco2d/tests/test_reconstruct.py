import math

import numpy as np
import pandas as pd
import pytest
from detector2d.geometry import Trajectory
from detectorsim2d.edm import DEPOSITS_COLUMNS, PARTICLES_COLUMNS

from detectorreco2d.config import ClusterResolution, RecoConfig, Resolution, TrackResolution
from detectorreco2d.edm import CLUSTERS_COLUMNS
from detectorreco2d.reconstruct import reconstruct, reconstruct_clusters, reconstruct_tracks, resolution

NO_SMEAR = RecoConfig()  # a=b=0 everywhere -> exact passthrough


def _particle_row(**overrides):
    row = dict(
        event_id=0,
        particle_id=0,
        species=np.nan,
        pdg=np.nan,
        x0=0.0,
        y0=0.0,
        phi0=0.0,
        charge=1.0,
        energy=10.0,
        radius=math.nan,
        jet_id=-1,
        is_b_jet=False,
    )
    row.update(overrides)
    return row


def _particles_df(rows):
    return pd.DataFrame(rows, columns=PARTICLES_COLUMNS)


def _deposit_row(**overrides):
    row = dict(
        event_id=0, particle_id=0, system="ecal", layer_id=100, cell_id=0, x=0.0, y=0.0, s_local=0.0, energy=1.0
    )
    row.update(overrides)
    return row


def _deposits_df(rows):
    return pd.DataFrame(rows, columns=DEPOSITS_COLUMNS)


def test_resolution_shrinks_towards_the_floor_as_x_grows():
    assert resolution(a=1.0, b=10.0, x=1.0) == pytest.approx(11.0)
    assert resolution(a=1.0, b=10.0, x=10.0) == pytest.approx(2.0)
    assert resolution(a=1.0, b=10.0, x=1000.0) == pytest.approx(1.01)


def test_resolution_falls_back_to_the_floor_for_non_positive_x():
    assert resolution(a=1.0, b=10.0, x=0.0) == 1.0
    assert resolution(a=1.0, b=10.0, x=-5.0) == 1.0


def test_reconstruct_tracks_only_keeps_charged_particles():
    particles = _particles_df(
        [
            _particle_row(particle_id=0, charge=1.0),
            _particle_row(particle_id=1, charge=0.0),
            _particle_row(particle_id=2, charge=-1.0),
        ]
    )
    tracks = reconstruct_tracks(particles, NO_SMEAR)
    assert sorted(tracks["particle_id"]) == [0, 2]


def test_reconstruct_tracks_includes_muons_like_any_other_charged_particle():
    particles = _particles_df([_particle_row(species="mu-", pdg=13, charge=-1.0)])
    tracks = reconstruct_tracks(particles, NO_SMEAR)
    assert len(tracks) == 1
    assert tracks.iloc[0]["species"] == "mu-"


def test_reconstruct_tracks_with_zero_resolution_reproduces_truth_exactly():
    particles = _particles_df([_particle_row(x0=3.0, y0=-1.0, phi0=0.4, energy=25.0, radius=math.nan)])
    tracks = reconstruct_tracks(particles, NO_SMEAR)
    row = tracks.iloc[0]
    expected_d0 = Trajectory(x0=3.0, y0=-1.0, phi0=0.4).d0
    assert row["d0_true"] == pytest.approx(expected_d0)
    assert row["d0"] == pytest.approx(expected_d0)
    assert row["phi0_true"] == pytest.approx(0.4)
    assert row["phi0"] == pytest.approx(0.4)
    assert row["pt_true"] == pytest.approx(25.0)
    assert row["pt"] == pytest.approx(25.0)


def test_reconstruct_tracks_smears_around_the_truth_value():
    config = RecoConfig(track=TrackResolution(d0=Resolution(a=0.5, b=0.0)))
    particles = _particles_df(
        [_particle_row(particle_id=i, x0=0.0, y0=0.0, phi0=0.0, energy=10.0) for i in range(2000)]
    )
    tracks = reconstruct_tracks(particles, config, rng=np.random.default_rng(0))
    residuals = tracks["d0"] - tracks["d0_true"]
    assert residuals.mean() == pytest.approx(0.0, abs=0.05)
    assert residuals.std() == pytest.approx(0.5, rel=0.15)


def test_reconstruct_tracks_pt_resolution_shrinks_with_pt():
    config = RecoConfig(track=TrackResolution(pt=Resolution(a=0.0, b=100.0)))
    low_pt = _particles_df([_particle_row(particle_id=i, energy=10.0) for i in range(500)])
    high_pt = _particles_df([_particle_row(particle_id=i, energy=1000.0) for i in range(500)])
    low = reconstruct_tracks(low_pt, config, rng=np.random.default_rng(1))
    high = reconstruct_tracks(high_pt, config, rng=np.random.default_rng(2))
    assert (low["pt"] - low["pt_true"]).std() > (high["pt"] - high["pt_true"]).std()


def test_reconstruct_tracks_pt_never_goes_non_positive_even_with_huge_smearing():
    config = RecoConfig(track=TrackResolution(pt=Resolution(a=1000.0, b=0.0)))
    particles = _particles_df([_particle_row(particle_id=i, energy=1.0) for i in range(200)])
    tracks = reconstruct_tracks(particles, config, rng=np.random.default_rng(3))
    assert (tracks["pt"] > 0).all()


def test_reconstruct_tracks_carries_jet_truth_through():
    particles = _particles_df([_particle_row(jet_id=2, is_b_jet=True)])
    tracks = reconstruct_tracks(particles, NO_SMEAR)
    assert tracks.iloc[0]["jet_id"] == 2
    assert tracks.iloc[0]["is_b_jet"] == True  # noqa: E712 (numpy bool, `is True` is too strict)


def test_reconstruct_tracks_defaults_missing_jet_columns_to_no_jet():
    particles = _particles_df([_particle_row()]).drop(columns=["jet_id", "is_b_jet"])
    tracks = reconstruct_tracks(particles, NO_SMEAR)
    assert tracks.iloc[0]["jet_id"] == -1
    assert tracks.iloc[0]["is_b_jet"] == False  # noqa: E712


def test_reconstruct_clusters_sums_deposits_per_particle():
    particles = _particles_df([_particle_row(species="electron", pdg=11, charge=-1.0)])
    deposits = _deposits_df([_deposit_row(cell_id=0, energy=4.0), _deposit_row(cell_id=1, energy=6.0)])
    clusters = reconstruct_clusters(particles, deposits, NO_SMEAR)
    assert len(clusters) == 1
    assert clusters.iloc[0]["energy_true"] == pytest.approx(10.0)
    assert clusters.iloc[0]["energy"] == pytest.approx(10.0)


def test_reconstruct_clusters_excludes_muons():
    particles = _particles_df([_particle_row(species="mu-", pdg=13, charge=-1.0)])
    deposits = _deposits_df([_deposit_row(system="ecal", energy=0.3), _deposit_row(system="hcal", energy=0.8)])
    clusters = reconstruct_clusters(particles, deposits, NO_SMEAR)
    assert len(clusters) == 0


def test_reconstruct_clusters_excludes_bare_species_free_stubs():
    particles = _particles_df([_particle_row(species=np.nan, pdg=np.nan)])
    deposits = _deposits_df([_deposit_row(energy=1.0)])
    clusters = reconstruct_clusters(particles, deposits, NO_SMEAR)
    assert len(clusters) == 0


def test_reconstruct_clusters_sums_across_systems_for_a_charged_hadron():
    # a charged hadron leaves a MIP trail in the ECAL *and* a shower in the
    # HCAL -- both count toward one cluster (the honest first-pass
    # simplification, see the module docstring).
    particles = _particles_df([_particle_row(species="pi+", pdg=211, charge=1.0)])
    deposits = _deposits_df([_deposit_row(system="ecal", energy=0.3), _deposit_row(system="hcal", energy=9.7)])
    clusters = reconstruct_clusters(particles, deposits, NO_SMEAR)
    assert clusters.iloc[0]["energy_true"] == pytest.approx(10.0)


def test_reconstruct_clusters_energy_never_goes_negative():
    config = RecoConfig(cluster=ClusterResolution(energy=Resolution(a=1000.0, b=0.0)))
    particles = _particles_df(
        [_particle_row(particle_id=i, species="photon", pdg=22, charge=0.0) for i in range(200)]
    )
    deposits = _deposits_df([_deposit_row(particle_id=i, energy=1.0) for i in range(200)])
    clusters = reconstruct_clusters(particles, deposits, config, rng=np.random.default_rng(4))
    assert (clusters["energy"] >= 0).all()


def test_reconstruct_clusters_empty_deposits_returns_an_empty_table_with_the_right_columns():
    particles = _particles_df([_particle_row(species="photon", pdg=22, charge=0.0)])
    clusters = reconstruct_clusters(particles, _deposits_df([]), NO_SMEAR)
    assert len(clusters) == 0
    assert list(clusters.columns) == CLUSTERS_COLUMNS


def test_reconstruct_returns_tracks_and_clusters_together():
    particles = _particles_df([_particle_row(species="electron", pdg=11, charge=-1.0)])
    deposits = _deposits_df([_deposit_row(energy=5.0)])
    tracks, clusters = reconstruct(particles, deposits, NO_SMEAR)
    assert len(tracks) == 1
    assert len(clusters) == 1
