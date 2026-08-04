import numpy as np
import pandas as pd
import pytest
from detectorreco2d.config import RecoConfig
from detectorreco2d.edm import TRACKS_COLUMNS
from detectorsim2d.config import FieldConfig, ParticleGunConfig, SimConfig
from detector2d.geometry import CircleLayer

from flavor_tagging.pipeline import RECO_CONFIG_PATH, SIM_CONFIG_PATH, reconstruct_run, run_pipeline, simulate_jets, summarize_jets


def _small_sim_config(**gun_overrides) -> SimConfig:
    gun = ParticleGunConfig(
        mode="jets",
        n_particles=60,
        jet_count_min=3,
        jet_count_max=3,
        jet_cone_sigma=0.05,
        pt_min=50.0,
        pt_max=300.0,
        species=("electron", "positron", "photon", "pi+", "pi-"),
        b_jet_fraction=0.5,
        b_jet_decay_length_min=10.0,
        b_jet_decay_length_max=90.0,
        b_jet_track_boost=0.2,
        b_jet_pt_boost=0.15,
        jet_muon_fraction=0.02,
        b_jet_muon_fraction=0.15,
    )
    for key, value in gun_overrides.items():
        setattr(gun, key, value)
    return SimConfig(
        layers=[CircleLayer(layer_id=i, center=(0.0, 0.0), radius=r) for i, r in enumerate((50.0, 100.0, 150.0))],
        magnetic_field=FieldConfig(bz=0.5),
        gun=gun,
        n_events=20,
        seed=123,
        world_radius=800.0,
        max_path_length=4000.0,
    )


def test_configs_paths_exist_on_disk():
    assert SIM_CONFIG_PATH.exists()
    assert RECO_CONFIG_PATH.exists()


def test_simulate_jets_produces_b_jets_and_light_jets():
    particles, hits, deposits = simulate_jets(_small_sim_config())
    in_a_jet = particles[particles["jet_id"] != -1]
    flavors = in_a_jet.groupby(["event_id", "jet_id"])["is_b_jet"].first()
    assert flavors.any()  # at least one b-jet
    assert not flavors.all()  # and at least one light jet


def test_reconstruct_run_returns_tracks_and_clusters_with_jet_truth():
    particles, _hits, deposits = simulate_jets(_small_sim_config())
    tracks, clusters = reconstruct_run(particles, deposits, RecoConfig())
    assert list(tracks.columns) == TRACKS_COLUMNS
    assert (tracks["jet_id"] != -1).any()
    assert tracks["is_b_jet"].isin([True, False]).all()


def test_run_pipeline_is_reproducible_with_an_explicit_seed():
    sim_config = _small_sim_config()
    particles1, _hits1, deposits1, tracks1, clusters1 = run_pipeline(sim_config, RecoConfig(), seed=7)
    particles2, _hits2, deposits2, tracks2, clusters2 = run_pipeline(sim_config, RecoConfig(), seed=7)
    pd.testing.assert_frame_equal(particles1, particles2, check_dtype=False)
    pd.testing.assert_frame_equal(tracks1, tracks2, check_dtype=False)


def test_summarize_jets_counts_tracks_and_muons_per_jet():
    tracks = pd.DataFrame(
        [
            dict(event_id=0, particle_id=0, jet_id=0, is_b_jet=False, species="pi+", charge=1.0,
                 d0_true=0, d0=0, phi0_true=0, phi0=0, pt_true=10, pt=10),
            dict(event_id=0, particle_id=1, jet_id=0, is_b_jet=False, species="electron", charge=-1.0,
                 d0_true=0, d0=0, phi0_true=0, phi0=0, pt_true=10, pt=10),
            dict(event_id=0, particle_id=2, jet_id=1, is_b_jet=True, species="mu-", charge=-1.0,
                 d0_true=0, d0=0, phi0_true=0, phi0=0, pt_true=10, pt=10),
            dict(event_id=0, particle_id=3, jet_id=1, is_b_jet=True, species="pi+", charge=1.0,
                 d0_true=0, d0=0, phi0_true=0, phi0=0, pt_true=10, pt=10),
            dict(event_id=0, particle_id=4, jet_id=-1, is_b_jet=False, species="electron", charge=-1.0,
                 d0_true=0, d0=0, phi0_true=0, phi0=0, pt_true=10, pt=10),
        ],
        columns=TRACKS_COLUMNS,
    )
    summary = summarize_jets(tracks)
    assert len(summary) == 2  # jet_id -1 is excluded

    light = summary[summary["jet_id"] == 0].iloc[0]
    assert light["n_tracks"] == 2
    assert light["n_muons"] == 0
    assert light["is_b_jet"] == False  # noqa: E712

    b_jet = summary[summary["jet_id"] == 1].iloc[0]
    assert b_jet["n_tracks"] == 2
    assert b_jet["n_muons"] == 1
    assert b_jet["is_b_jet"] == True  # noqa: E712


def test_summarize_jets_of_an_empty_tracks_table_is_empty():
    summary = summarize_jets(pd.DataFrame(columns=TRACKS_COLUMNS))
    assert len(summary) == 0
    assert list(summary.columns) == ["event_id", "jet_id", "is_b_jet", "n_tracks", "n_muons"]
