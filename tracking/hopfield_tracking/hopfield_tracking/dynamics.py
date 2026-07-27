"""Continuous Hopfield dynamics (Denby 1988, section 5's general formalism,
reused for track finding in section 8):

    tau dv_i/dt = sum_j T_ij f(v_j) - v_i          (relaxation)
    E = -1/2 sum_ij T_ij f(v_i) f(v_j)              (energy)

discretized by explicit Euler with the paper's own step size, Delta t =
0.5 tau. `f` is a sigmoid; the paper uses a piecewise-linear approximation
(their fig. 7) "to keep the computing time and memory requirements low" --
`sigmoid` below is a symmetric clipped-linear stand-in for that curve. The
energy is guaranteed non-increasing for true asynchronous (one-neuron-at-a-
time) Hopfield updates; this synchronous Euler discretization only
approximates that, and can occasionally tick up between steps, though it
still reliably relaxes downward overall (see `tests/test_dynamics.py`).

**Why the initial `v` is narrow, and why `type1_scale` in `coefficients.py`
must exceed 1**: a completely correct chain of segments, all reinforcing
each other at the paper's own coefficient magnitude, is only *neutrally*
stable -- for an interior segment with two neighbors, the true fixed point
input exactly equals its own value, but a finite chain's end segments (only
one neighbor) can't sustain themselves at that same level, and that
weakness cascades inward. Linearizing the dynamics around `v=0` shows the
"everything on" solution only grows (rather than decays back to the trivial
"everything off" solution) when the largest eigenvalue of `T` exceeds 1 --
which requires scaling the paper's `type1` formula up by roughly a factor
of ~2-3 in our normalized units. Once that holds, starting `v` broadly
across `[0, 1]` combined with a steep `gain` just saturates neurons to 0/1
based on their *initial* random sign rather than letting the network's
actual structure decide anything -- so we start instead in a narrow band
around the sigmoid's unstable center (0.5) and let the (now genuinely
amplifying) dynamics differentiate real signal from noise over several
iterations, the same "start near the middle, let structure win" mechanism
the paper describes for its TSP network.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_GAIN = 4.0
DEFAULT_DT_OVER_TAU = 0.5
#: initial v is drawn from Uniform(0.5 - INIT_SPREAD, 0.5 + INIT_SPREAD),
#: i.e. close to the sigmoid's unstable center rather than spread across
#: [0, 1] -- see the module docstring's "why a narrow init" note.
DEFAULT_INIT_SPREAD = 0.1


def sigmoid(v: np.ndarray, gain: float = DEFAULT_GAIN) -> np.ndarray:
    """Piecewise-linear response function: a clipped linear ramp through
    (0.5, 0.5) with slope `gain`, flat at 0 and 1 outside the ramp -- the
    same "central linear region with plateaux" shape as the paper's fig. 7,
    just symmetric for simplicity."""
    return np.clip(gain * (v - 0.5) + 0.5, 0.0, 1.0)


def energy(t: np.ndarray, f_v: np.ndarray) -> float:
    """E = -1/2 f(v)^T T f(v)."""
    return float(-0.5 * f_v @ t @ f_v)


def step(v: np.ndarray, t: np.ndarray, f_v: np.ndarray, dt_over_tau: float = DEFAULT_DT_OVER_TAU) -> np.ndarray:
    """One explicit-Euler update of tau dv/dt = sum_j T_ij f(v_j) - v_i."""
    input_current = t @ f_v
    return v + dt_over_tau * (input_current - v)


@dataclass
class RelaxationHistory:
    """Per-iteration record of the network's state, index 0 = the random
    initial condition. Everything `hopfield_tracking.vis` plots comes from
    here."""

    v: list[np.ndarray] = field(default_factory=list)
    f_v: list[np.ndarray] = field(default_factory=list)
    energy: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.energy)


def relax(
    t: np.ndarray,
    n_iterations: int = 60,
    dt_over_tau: float = DEFAULT_DT_OVER_TAU,
    gain: float = DEFAULT_GAIN,
    rng: np.random.Generator | None = None,
    v0: np.ndarray | None = None,
    init_spread: float = DEFAULT_INIT_SPREAD,
    energy_tol: float = 1e-6,
    state_tol: float = 1e-4,
) -> RelaxationHistory:
    """Relax the network from a random start (paper: "random starting
    values are assigned to the neuron outputs") for up to `n_iterations`
    steps, stopping early once the *state* stops changing (paper: iterate
    "until the pattern of output values stops changing"; convergence
    "usually occurred in less than 10 iterations" -- ours typically takes a
    bit longer, 15-40, at the more conservative `gain` this implementation
    needs, see the module docstring).

    Deliberately checks `v` itself (`state_tol`), not just the energy: two
    genuinely different, still-evolving states can briefly have equal
    energy (e.g. by symmetry) without the network having actually settled,
    so an energy-only plateau check can stop too early on a state that's
    still very much in motion. `energy_tol` is kept as an additional (not
    sufficient on its own) condition.
    """
    rng = rng if rng is not None else np.random.default_rng()
    n = t.shape[0]
    v = v0 if v0 is not None else rng.uniform(0.5 - init_spread, 0.5 + init_spread, size=n)

    history = RelaxationHistory()
    f_v = sigmoid(v, gain)
    history.v.append(v.copy())
    history.f_v.append(f_v.copy())
    history.energy.append(energy(t, f_v))

    for _ in range(n_iterations):
        v_prev = v
        v = step(v, t, f_v, dt_over_tau)
        f_v = sigmoid(v, gain)
        history.v.append(v.copy())
        history.f_v.append(f_v.copy())
        history.energy.append(energy(t, f_v))
        state_converged = np.max(np.abs(v - v_prev)) < state_tol
        energy_converged = abs(history.energy[-1] - history.energy[-2]) < energy_tol
        if state_converged and energy_converged:
            break

    return history
