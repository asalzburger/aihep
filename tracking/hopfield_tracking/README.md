# hopfield_tracking

A from-scratch, didactic reimplementation of the Hopfield-network track
finder from B. Denby, *"Neural networks and cellular automata in
experimental high energy physics"*, Computer Physics Communications 49
(1988) 429-448 (section 8, `tracking/denby/resources/denby-paper-1988.pdf`)
-- one of the first applications of a neural network to particle-track
reconstruction. Works on plain `(x, y)` hit points; it doesn't know about
`detector2d` layers at all, so it's reusable on any 2D hit set.

## The paper, in brief

The full paper surveys neural nets and cellular automata across several HEP
pattern-recognition problems (a Traveling-Salesman-style combinatorics
network, calorimeter clustering, a "contiguity trigger"), but the
centerpiece -- and everything this package implements -- is section 8's
continuous Hopfield network for **track finding**:

- **Neurons** are candidate track *segments*: `(i, j)` represents a directed
  segment from measured point `i` to point `j`. A valid track is a
  non-bifurcating chain of "on" segments.
- **Dynamics**: `tau dv/dt = sum_j T_ij f(v_j) - v_i`, with a piecewise-
  linear sigmoid `f` (their fig. 7). **Energy** `E = -1/2 sum_ij T_ij f(v_i)
  f(v_j)` decreases as the network relaxes -- this is exactly the number
  printed in the paper's fig. 8 panels.
- **Coefficients** `T_ij`, the actual physics:
  - **type 1** (segments sharing an endpoint): `T_ij ~ cos(theta)^n / r_ij`
    -- reward a smooth, short continuation.
  - **type 2** (segments that don't share a point but lie nearby): `T_ij ~
    1 - chi^2` of a circle fit through their 4 endpoints -- added later in
    the paper specifically to help resolve close-together tracks.
  - **inhibition**: a constant `-B` between segments that compete for the
    same start or end point.
- Random initial state, `n=5`, `Delta t = 0.5 tau`, `R_c = 4.5<r>`
  (candidate-neighbor cutoff), converges in <10 iterations. Fig. 8(a) (a
  4-track event) reconstructs perfectly; fig. 8(b) (5 tracks) converges but
  with confusion where tracks pass close together.

`tracking/denby/resources/denby_detector_event.svg` turns out to be a
traced vector copy of fig. 8(a)'s own *last frame* -- its "Energy
-1158.3411 / Iteration 6 / T=3.0tau" annotation matches the paper exactly.
`tracking/denby/` already reverse-engineered that event's geometry; this
package is what actually *runs the algorithm* that produced it.

## Project layout (same order the paper presents section 8)

| module | contents |
|---|---|
| `network.py` | `Segment` (one neuron), `build_segments(x, y, r_c, layer_ids=None)` (candidate generation: R_c cutoff, plus excluding same-layer pairs when `layer_ids` is given -- see below), `mean_consecutive_hit_distance` (the paper's `<r>`, from ground truth). |
| `coefficients.py` | The `T_ij` partition (`classify_pair`): **reverse**/**compete** -> inhibit, **chain** -> type 1, else type 2 if close. `build_weight_matrix` assembles the full matrix. |
| `dynamics.py` | `sigmoid`, `energy`, `step`, `relax` (the Euler relaxation loop + history, for plotting). |
| `extract.py` | Threshold "on" segments and chain them into tracks; `score_against_truth` for the quantitative perfect/confused check. |
| `vis.py` | `plot_iterations`: the fig.-8-style multi-panel figure. |
| `cli.py` | `run`: hits CSV -> relaxation -> figure + score, end to end. |

## Two things the paper leaves implicit that turned out to matter

Getting this to actually converge to the right answer (rather than to the
trivial "everything off" solution, or a symmetric standoff) took real
debugging, and the paper's own admission -- *"the choice of coefficients
seems to be rather delicate, and a number of attempts were made before
arriving at forms which gave reasonable results"* -- is not an
exaggeration. Two structural findings, not just knob-turning:

1. **`type1_scale` must exceed ~1.** The paper's own type-1 formula gives
   an interior segment on a straight chain a coefficient of exactly 0.5 to
   each neighbor. That's only *neutrally* stable: a finite chain's end
   segments (one neighbor instead of two) can't sustain themselves at that
   level, and the weakness cascades inward until the whole chain decays
   toward zero. Linearizing the dynamics around `v=0` shows the correct
   solution only *grows* (rather than decaying back to trivial) once the
   largest eigenvalue of `T` exceeds 1 -- which needs the type-1 coefficient
   scaled up by roughly 2-3x in our normalized units
   (`coefficients.DEFAULT_TYPE1_SCALE`).
2. **The initial `v` should be narrow, not spread across `[0, 1]`.** A wide
   random start combined with a steep sigmoid gain just saturates every
   neuron to 0 or 1 based on its *initial random sign*, before the
   network's actual structure gets a chance to matter. Starting instead in
   a narrow band around the sigmoid's unstable center (`dynamics.
   DEFAULT_INIT_SPREAD`) and letting the now-genuinely-amplifying dynamics
   differentiate real signal from noise over several iterations reproduces
   the paper's own "start near the middle, let structure win" description
   of its TSP network.

## Same-layer segments are unphysical, and excluded

A candidate segment built purely from an `R_c` distance cutoff can connect
two hits that happen to be close together *on the same detector layer* --
easy to hit right where several tracks fan out from one vertex, since
different tracks' hits on the same layer can end up closer to each other
than consecutive hits along a single track. That's never a valid track
segment: a real particle produces at most one hit per plane, so two hits on
one layer can't be the two ends of one physical segment, however close they
are in `(x, y)`. On the recovered Denby event this was 44 of 188 candidate
segments (23%) -- pure noise for the dynamics to (usually, not reliably)
out-inhibit on its own, rather than something excluded from consideration
up front the way the paper's own figures show.

`build_segments(x, y, r_c, layer_ids=...)` excludes any pair sharing a
layer id; `cli.run` picks this up automatically from a `layer_id` column if
the hits table has one (see `tracking/denby/python/run_hopfield_on_denby_event.py`,
which now carries `layer_id` through from `tracksim2d`'s hits table for
exactly this reason). It's optional (`layer_ids=None` keeps the old pure-
distance behavior) since this package otherwise only ever needs `(x, y)`.

See `dynamics.py`'s module docstring for the full derivation.

## How well does it actually reproduce the paper?

**Perfectly, on a clean case.** Three straight tracks fanning from a common
vertex at well-separated angles reconstruct exactly, matching ground truth
segment-for-segment in under 15 iterations (`tests/test_extract.py::test_end_to_end_perfect_reconstruction_of_a_well_separated_fan`)
-- proof the mechanism itself (coefficients + dynamics + extraction) is
correct, independent of the harder case below.

**Mostly, on the paper's own event.** Running on `tracking/denby`'s
recovered 4-track event (`../denby/python/run_hopfield_on_denby_event.py`)
reliably reconstructs 2 of the 4 tracks exactly and the other 2 correctly
along their outer ~8 of 13 hits, with confusion in their innermost ~5 hits
-- precisely where all 4 tracks pass closest together, right after the
shared vertex. This is a smaller-scale echo of what the paper itself says
about fig. 8(b): *"confusion in regions where tracks are very close
together, leading to incorrect or illegal ... solutions in these regions."*
Here that region is the vertex itself (since, unlike a typical fig. 8(a)/(b)
crossing, all 4 tracks in our recovered event originate from *one* shared
point), which is a harder disambiguation problem than two tracks crossing
mid-flight.

**Type 2 doesn't help yet, as currently gated.** The paper adds type-2
coefficients specifically to resolve close-together tracks -- exactly our
remaining problem -- so it was a natural thing to try. As implemented
(`use_type2=True`), it currently makes things *worse*: its locality/radius
gates admit too many spuriously "co-circular" 4-point combinations near the
crowded vertex, overwhelming the real signal (`tests/test_coefficients.py`
covers the mechanism itself, which is correct in isolation -- e.g. it
correctly favors true circle fits and correctly zeroes out far-apart pairs
-- the remaining issue is calibrating the gate tightly enough for this
geometry). It's implemented, tested, and available via `--type2`/
`use_type2=True`, off by default, and left as a documented, honest
opportunity for further tuning rather than something force-fit to look
better than it currently is.

## Setup

```bash
cd tracking/hopfield_tracking
python3 -m venv .venv
.venv/bin/pip install -e . -r requirements.txt
```

No dependency on `detector2d`/`tracksim2d` -- just `numpy`/`pandas`/
`matplotlib` -- since this package only ever sees a plain hits table.

## Running it

```bash
.venv/bin/python -m hopfield_tracking.cli run --hits hits.csv --save fig8.png
```

`hits.csv` needs `x, y` columns; a `particle_id` column, if present, is
used both to auto-calibrate `R_c`/`r_scale` from `<r>` and to score the
result against ground truth. `--type2` turns on type-2 coefficients (off by
default, see above); `--seed` fixes the random initial state.

The saved figure defaults to a 2x2-ish grid (`--layout grid`, or `row` for
a single row) with the y-axis flipped so the vertex renders at the bottom
and tracks fan upward (`--no-invert-y` to disable) -- matching the paper's
own fig. 8 layout and orientation; see `vis.plot_iterations` if you're
calling it as a library rather than through the CLI.

See `tracking/denby/python/run_hopfield_on_denby_event.py` for reproducing
the paper's own event end to end, and `run_hopfield_sweep.py` for the
1-6-track multiplicity study described below.

## Using it as a library

```python
import pandas as pd
from hopfield_tracking.network import build_segments, mean_consecutive_hit_distance
from hopfield_tracking.coefficients import build_weight_matrix
from hopfield_tracking.dynamics import relax
from hopfield_tracking.extract import on_segments, chain_tracks, score_against_truth

hits = pd.read_csv("hits.csv")  # x, y, and optionally particle_id, layer_id
mean_r = mean_consecutive_hit_distance(hits)
r_c = 1.5 * mean_r

layer_ids = hits["layer_id"].to_numpy() if "layer_id" in hits else None
segments = build_segments(hits["x"].to_numpy(), hits["y"].to_numpy(), r_c, layer_ids=layer_ids)
t = build_weight_matrix(segments, r_c=r_c, r_scale=mean_r, inhibition=-0.8, use_type2=False)
history = relax(t, n_iterations=60)

active = on_segments(segments, history.f_v[-1])
chains = chain_tracks(active)
```

## Tests

```bash
.venv/bin/python -m pytest tests/
```

Covers segment generation, the `R_c` cutoff, and same-layer exclusion; the
`classify_pair` partition (all four cases, plus symmetry); the sign/scale
behavior of type-1 (rewards smooth continuation, inhibits sharp reversal);
the circle fit and type-2 gate; the sigmoid shape, energy formula, and one
Euler step against hand-computed values; `cli.run`'s auto-detection of
`layer_id`; and, as an end-to-end regression guard for the calibrated
defaults, perfect reconstruction of a clean 3-track fan.
