import pandas as pd
import pytest

from splitting.pipeline import apply_splitter
from splitting.truth_splitter import NO_CONTRIBUTION_KEY, TruthSplitter, dominant_particle_per_pixel


def _hit(event_id, ix, iy, cluster_id, charge):
    return dict(
        event_id=event_id,
        ix=ix,
        iy=iy,
        x_center_um=(ix + 0.5) * 25.0,
        y_center_um=(iy + 0.5) * 50.0,
        charge=charge,
        cluster_id=cluster_id,
    )


def _contribution(event_id, particle_id, ix, iy, charge):
    return dict(event_id=event_id, particle_id=particle_id, ix=ix, iy=iy, charge=charge)


def test_dominant_particle_per_pixel_picks_the_larger_contributor():
    contributions = pd.DataFrame(
        [
            _contribution(0, particle_id=0, ix=5, iy=5, charge=0.3),
            _contribution(0, particle_id=1, ix=5, iy=5, charge=0.7),
        ]
    )
    dominant = dominant_particle_per_pixel(contributions)

    assert len(dominant) == 1
    assert dominant.iloc[0]["particle_id"] == 1


def test_truth_splitter_separates_a_cleanly_shared_cluster():
    # one connected cluster of 5 pixels: 2 from particle 0, 2 from particle
    # 1, and one pure-noise pixel with no truth contribution at all (a
    # noise pixel that happened to land next to the real cluster and got
    # glued on by connectivity -- this does happen in real runs, see
    # resources/p123).
    hits = pd.DataFrame(
        [
            _hit(0, 0, 0, cluster_id=0, charge=1.0),
            _hit(0, 1, 0, cluster_id=0, charge=1.0),
            _hit(0, 2, 0, cluster_id=0, charge=1.0),
            _hit(0, 3, 0, cluster_id=0, charge=1.0),
            _hit(0, 4, 0, cluster_id=0, charge=0.05),
        ]
    )
    contributions = pd.DataFrame(
        [
            _contribution(0, particle_id=0, ix=0, iy=0, charge=1.0),
            _contribution(0, particle_id=0, ix=1, iy=0, charge=1.0),
            _contribution(0, particle_id=1, ix=2, iy=0, charge=1.0),
            _contribution(0, particle_id=1, ix=3, iy=0, charge=1.0),
            # no contribution row for pixel (4, 0): pure noise
        ]
    )

    new_hits, new_clusters = apply_splitter(TruthSplitter(), hits, pd.DataFrame(), contributions)

    assert len(new_hits) == 5  # no pixels dropped
    assert new_hits["charge"].sum() == pytest.approx(4.05)
    assert len(new_clusters) == 3  # particle 0's pixels, particle 1's pixels, the noise pixel

    by_size = new_clusters.sort_values("n_pixels").reset_index(drop=True)
    assert list(by_size["n_pixels"]) == [1, 2, 2]
    assert by_size.iloc[0]["charge_sum"] == pytest.approx(0.05)  # the noise-only cluster
    assert sorted(by_size["charge_sum"].iloc[1:]) == pytest.approx([2.0, 2.0])

    # every split cluster with real contributions is purely one particle
    merged = new_hits[["event_id", "ix", "iy", "cluster_id"]].merge(
        contributions, on=["event_id", "ix", "iy"], how="inner"
    )
    contributors_per_cluster = merged.groupby(["event_id", "cluster_id"])["particle_id"].nunique()
    assert (contributors_per_cluster == 1).all()


def test_truth_splitter_key_is_sentinel_for_unattributed_pixels():
    hits = pd.DataFrame([_hit(0, 0, 0, cluster_id=0, charge=0.05)])
    contributions = pd.DataFrame(columns=["event_id", "particle_id", "ix", "iy", "charge"])

    key = TruthSplitter().split_key(hits, pd.DataFrame(), contributions)

    assert key.iloc[0] == NO_CONTRIBUTION_KEY
