import math

import numpy as np
import pandas as pd
import pytest
from detectorsim2d.edm import DEPOSITS_COLUMNS, PARTICLES_COLUMNS

from detectorreco2d.config import RecoConfig
from detectorreco2d.edm import CLUSTERS_COLUMNS
from detectorreco2d.io import read_clusters, read_run, read_tracks, write_run
from detectorreco2d.reconstruct import reconstruct


def _particles_df():
    row = dict(
        event_id=0,
        particle_id=0,
        species="electron",
        pdg=11,
        x0=1.0,
        y0=2.0,
        phi0=0.3,
        charge=-1.0,
        energy=10.0,
        radius=math.nan,
        jet_id=-1,
        is_b_jet=False,
    )
    return pd.DataFrame([row], columns=PARTICLES_COLUMNS)


def _deposits_df():
    row = dict(
        event_id=0, particle_id=0, system="ecal", layer_id=100, cell_id=0, x=0.0, y=0.0, s_local=0.0, energy=5.0
    )
    return pd.DataFrame([row], columns=DEPOSITS_COLUMNS)


@pytest.mark.parametrize("fmt", ["csv", "arrow"])
def test_write_read_round_trip(tmp_path, fmt):
    tracks, clusters = reconstruct(_particles_df(), _deposits_df(), RecoConfig(), rng=np.random.default_rng(0))

    write_run(tmp_path, fmt, tracks, clusters)
    tracks2, clusters2 = read_run(tmp_path, fmt)

    pd.testing.assert_frame_equal(tracks, tracks2, check_dtype=False)
    pd.testing.assert_frame_equal(clusters, clusters2, check_dtype=False)


@pytest.mark.parametrize("fmt", ["csv", "arrow"])
def test_reading_clusters_from_a_clusterless_run_gives_an_empty_table(tmp_path, fmt):
    tracks, _clusters = reconstruct(_particles_df(), _deposits_df(), RecoConfig(), rng=np.random.default_rng(0))
    write_run(tmp_path, fmt, tracks)  # no clusters argument

    clusters = read_clusters(tmp_path, fmt)
    assert len(clusters) == 0
    assert list(clusters.columns) == CLUSTERS_COLUMNS
    assert not (tmp_path / f"clusters.{fmt}").exists()


@pytest.mark.parametrize("fmt", ["csv", "arrow"])
def test_read_run_stays_a_two_tuple_even_without_clusters(tmp_path, fmt):
    tracks, _clusters = reconstruct(_particles_df(), _deposits_df(), RecoConfig(), rng=np.random.default_rng(0))
    write_run(tmp_path, fmt, tracks)

    result = read_run(tmp_path, fmt)
    assert len(result) == 2
    pd.testing.assert_frame_equal(result[0], tracks, check_dtype=False)


@pytest.mark.parametrize("fmt", ["csv", "arrow"])
def test_read_tracks_matches_write_run(tmp_path, fmt):
    tracks, clusters = reconstruct(_particles_df(), _deposits_df(), RecoConfig(), rng=np.random.default_rng(0))
    write_run(tmp_path, fmt, tracks, clusters)
    pd.testing.assert_frame_equal(read_tracks(tmp_path, fmt), tracks, check_dtype=False)
