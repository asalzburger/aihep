import numpy as np
import pandas as pd
import pytest

from hopfield_tracking.coefficients import build_weight_matrix
from hopfield_tracking.dynamics import relax
from hopfield_tracking.extract import chain_tracks, on_segments, score_against_truth
from hopfield_tracking.network import Segment, build_segments, mean_consecutive_hit_distance


def _seg(index, start_id, end_id):
    return Segment(index=index, start_id=start_id, end_id=end_id, start_xy=(0.0, 0.0), end_xy=(0.0, 0.0))


def test_on_segments_thresholds_by_output_value():
    segments = [_seg(0, 0, 1), _seg(1, 1, 2)]
    active = on_segments(segments, f_v=np.array([0.5, 0.05]), threshold=0.1)
    assert active == [segments[0]]


def test_chain_tracks_follows_a_simple_chain():
    segments = [_seg(0, 0, 1), _seg(1, 1, 2), _seg(2, 2, 3)]
    chains = chain_tracks(segments)
    assert chains == [[0, 1, 2, 3]]


def test_chain_tracks_finds_multiple_independent_chains():
    segments = [_seg(0, 0, 1), _seg(1, 1, 2), _seg(2, 10, 11)]
    chains = chain_tracks(segments)
    assert sorted(chains) == [[0, 1, 2], [10, 11]]


def test_chain_tracks_truncates_at_a_bifurcation():
    # point 1 has two outgoing "on" segments -- illegal, so the chain from 0
    # stops at 1 rather than picking one arbitrarily.
    segments = [_seg(0, 0, 1), _seg(1, 1, 2), _seg(2, 1, 3)]
    chains = chain_tracks(segments)
    assert [0, 1] in chains


def test_score_against_truth_perfect():
    chains = [[0, 1, 2], [3, 4]]
    truth = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1}
    score = score_against_truth(chains, truth)
    assert score == dict(n_true_tracks=2, n_found_chains=2, n_exact_matches=2, perfect=True)


def test_score_against_truth_partial():
    chains = [[0, 1]]  # missing point 2 of the true 3-point track
    truth = {0: 0, 1: 0, 2: 0}
    score = score_against_truth(chains, truth)
    assert score["perfect"] is False
    assert score["n_exact_matches"] == 0


def test_end_to_end_perfect_reconstruction_of_a_well_separated_fan():
    """Regression guard for the calibrated defaults (coefficients.DEFAULT_TYPE1_SCALE,
    dynamics.DEFAULT_GAIN/DEFAULT_INIT_SPREAD): three widely-separated straight
    tracks from a common vertex must reconstruct exactly, the same clean case
    used to calibrate those defaults in the first place (see README.md)."""
    rows = []
    for particle_id, phi in enumerate([1.2, 1.6, 2.0]):  # radians, ~23 degrees apart
        for k in range(1, 9):
            rows.append(dict(particle_id=particle_id, x=k * 60.0 * np.cos(phi), y=k * 60.0 * np.sin(phi)))
    hits = pd.DataFrame(rows)

    x, y = hits["x"].to_numpy(), hits["y"].to_numpy()
    mean_r = mean_consecutive_hit_distance(hits)
    r_c = 1.5 * mean_r
    segments = build_segments(x, y, r_c)
    t = build_weight_matrix(segments, r_c=r_c, r_scale=mean_r, inhibition=-0.5, use_type2=False)

    history = relax(t, n_iterations=100, gain=4.0, rng=np.random.default_rng(0), energy_tol=1e-9)
    active = on_segments(segments, history.f_v[-1])
    chains = chain_tracks(active)
    score = score_against_truth(chains, dict(enumerate(hits["particle_id"])))

    assert score["perfect"]
