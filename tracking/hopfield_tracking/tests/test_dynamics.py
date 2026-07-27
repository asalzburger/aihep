import numpy as np
import pytest

from hopfield_tracking.dynamics import energy, relax, sigmoid, step


def test_sigmoid_is_0_5_at_center():
    assert sigmoid(np.array([0.5]), gain=8.0)[0] == pytest.approx(0.5)


def test_sigmoid_clips_to_0_and_1():
    v = np.array([-10.0, 0.0, 0.5, 1.0, 10.0])
    f = sigmoid(v, gain=8.0)
    assert f.min() >= 0.0
    assert f.max() <= 1.0
    assert f[0] == 0.0
    assert f[-1] == 1.0


def test_sigmoid_is_monotonic():
    v = np.linspace(0, 1, 50)
    f = sigmoid(v, gain=4.0)
    assert np.all(np.diff(f) >= 0)


def test_energy_matches_manual_calculation():
    t = np.array([[0.0, 2.0], [2.0, 0.0]])
    f_v = np.array([0.5, 0.25])
    # f^T T f = f0*T01*f1 + f1*T10*f0 = 0.5*2*0.25 + 0.25*2*0.5 = 0.5; E = -1/2 * 0.5
    assert energy(t, f_v) == pytest.approx(-0.25)


def test_step_matches_euler_formula():
    t = np.array([[0.0, 1.0], [1.0, 0.0]])
    v = np.array([0.5, 0.5])
    f_v = np.array([0.5, 0.5])
    result = step(v, t, f_v, dt_over_tau=0.5)
    # input = T @ f_v = [0.5, 0.5]; v_new = v + 0.5*(input - v) = v (already at the fixed point)
    assert result == pytest.approx([0.5, 0.5])


def test_relax_records_history_including_initial_state():
    t = np.zeros((3, 3))
    history = relax(t, n_iterations=5, v0=np.full(3, 0.5))
    assert len(history) >= 1
    assert len(history.v) == len(history.f_v) == len(history.energy)


def test_relax_stops_early_once_energy_plateaus():
    t = np.zeros((4, 4))  # no coupling at all -> decays to a fixed point in one step
    history = relax(t, n_iterations=100, v0=np.full(4, 0.5), energy_tol=1e-6)
    assert len(history) < 100


def test_relax_saturates_a_self_reinforcing_chain_to_all_on():
    # a 3-segment straight chain with a strong-enough type-1-like coupling
    # should saturate fully on, from a near-center random start.
    t = np.array(
        [
            [0.0, 1.5, 0.0],
            [1.5, 0.0, 1.5],
            [0.0, 1.5, 0.0],
        ]
    )
    history = relax(t, n_iterations=100, gain=8.0, rng=np.random.default_rng(0), energy_tol=1e-9)
    assert history.f_v[-1] == pytest.approx([1.0, 1.0, 1.0], abs=1e-6)
