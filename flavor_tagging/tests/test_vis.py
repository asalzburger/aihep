"""Smoke tests: each plotting function runs end to end and produces a real
file, on tiny synthetic tracks/clusters tables (no need to run the full
simulate+reconstruct pipeline just to exercise matplotlib)."""

import matplotlib

matplotlib.use("Agg")  # headless: these tests only check that a file gets written

import numpy as np
import pandas as pd
import pytest
from detectorreco2d.edm import CLUSTERS_COLUMNS, TRACKS_COLUMNS

from flavor_tagging.vis import (
    make_validation_plots,
    plot_cluster_energy,
    plot_muon_multiplicity,
    plot_track_d0,
    plot_track_multiplicity,
)


def _tracks_df(n_light: int = 40, n_b_jet: int = 20, rng: np.random.Generator | None = None) -> pd.DataFrame:
    rng = rng or np.random.default_rng(0)
    rows = []
    particle_id = 0
    for jet_id, (is_b_jet, n) in enumerate([(False, n_light), (True, n_b_jet)]):
        for _ in range(n):
            species = "mu-" if rng.random() < (0.15 if is_b_jet else 0.02) else "pi+"
            pt = rng.uniform(50.0, 300.0)
            rows.append(
                dict(
                    event_id=0,
                    particle_id=particle_id,
                    jet_id=jet_id,
                    is_b_jet=is_b_jet,
                    species=species,
                    charge=1.0,
                    d0_true=0.0,
                    d0=rng.normal(0.0, 5.0 if is_b_jet else 0.5),
                    phi0_true=0.0,
                    phi0=0.0,
                    pt_true=pt,
                    pt=pt,
                )
            )
            particle_id += 1
    return pd.DataFrame(rows, columns=TRACKS_COLUMNS)


def _clusters_df(n: int = 50, rng: np.random.Generator | None = None) -> pd.DataFrame:
    rng = rng or np.random.default_rng(1)
    energy_true = rng.uniform(10.0, 200.0, size=n)
    rows = [
        dict(
            event_id=0,
            particle_id=i,
            jet_id=0,
            is_b_jet=False,
            species="photon",
            energy_true=e,
            energy=max(0.0, rng.normal(e, 5.0)),
        )
        for i, e in enumerate(energy_true)
    ]
    return pd.DataFrame(rows, columns=CLUSTERS_COLUMNS)


def test_plot_track_d0_writes_a_file(tmp_path):
    plot_track_d0(_tracks_df(), save_path=tmp_path / "d0.png")
    assert (tmp_path / "d0.png").exists()


def test_plot_cluster_energy_writes_a_file(tmp_path):
    plot_cluster_energy(_clusters_df(), save_path=tmp_path / "energy.png")
    assert (tmp_path / "energy.png").exists()


def test_plot_track_multiplicity_writes_a_file(tmp_path):
    plot_track_multiplicity(_tracks_df(), save_path=tmp_path / "n_tracks.png")
    assert (tmp_path / "n_tracks.png").exists()


def test_plot_muon_multiplicity_writes_a_file(tmp_path):
    plot_muon_multiplicity(_tracks_df(), save_path=tmp_path / "n_muons.png")
    assert (tmp_path / "n_muons.png").exists()


def test_make_validation_plots_writes_all_four(tmp_path):
    paths = make_validation_plots(_tracks_df(), _clusters_df(), tmp_path)
    assert len(paths) == 4
    for path in paths.values():
        assert path.exists()
