"""Run the Hopfield track finder on a hits CSV and report/plot the result.

    python -m hopfield_tracking.cli run --hits hits.csv --save fig8.png
    python -m hopfield_tracking.cli run --hits hits.csv --type2

`hits.csv` needs at least `x, y` columns; a `particle_id` column, if
present, is used both to auto-calibrate R_c (see
`network.mean_consecutive_hit_distance`) and to score the result against
ground truth; a `layer_id` column, if present, excludes same-layer
candidate segments (see `network.build_segments`) -- none of this is
required to run the network itself.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .coefficients import DEFAULT_TYPE1_SCALE, build_weight_matrix
from .dynamics import DEFAULT_GAIN, DEFAULT_INIT_SPREAD, relax
from .extract import chain_tracks, on_segments, score_against_truth
from .network import build_segments, mean_consecutive_hit_distance
from .vis import plot_iterations

#: See coefficients.py/dynamics.py for why these aren't the paper's own
#: literal values (4.5, 1, "wide-random-init") -- R_c=4.5<r> and an
#: unscaled type-1 coefficient are calibrated for the paper's own chamber's
#: hit density, and are both far too permissive/weak for this geometry.
DEFAULT_R_C_FACTOR = 1.5


def run(
    hits: pd.DataFrame,
    n: int = 5,
    r_c: float | None = None,
    r_c_factor: float = DEFAULT_R_C_FACTOR,
    r_scale: float | None = None,
    use_type2: bool = False,
    inhibition: float = -0.8,
    type1_scale: float = DEFAULT_TYPE1_SCALE,
    type2_scale: float = 1.0,
    gain: float = DEFAULT_GAIN,
    dt_over_tau: float = 0.5,
    n_iterations: int = 60,
    init_spread: float = DEFAULT_INIT_SPREAD,
    seed: int | None = None,
    layer_ids: np.ndarray | None = None,
):
    """Run the full pipeline on one event's hits. Returns (segments, history, chains, score).

    `r_scale` (a characteristic length used to non-dimensionalize the type-1
    coefficient, see `coefficients.type1_coefficient`) defaults to the same
    calibrated `<r>` used for `r_c` when ground truth is available.

    `use_type2` defaults to `False`: as currently gated (locality window,
    circle-fit chi^2), type-2 coefficients measurably *hurt* reconstruction
    on our reference event rather than helping resolve the close-together
    tracks they're meant to (see the README) -- included and tested for its
    own sake, and as a documented, working starting point for further
    tuning, not because it's recommended out of the box yet.

    `layer_ids` defaults to `hits["layer_id"]` when that column exists, and
    is used to exclude same-layer candidate segments (see
    `network.build_segments`).
    """
    hits = hits.reset_index(drop=True)
    x, y = hits["x"].to_numpy(), hits["y"].to_numpy()

    if layer_ids is None and "layer_id" in hits:
        layer_ids = hits["layer_id"].to_numpy()

    if r_c is None or r_scale is None:
        if "particle_id" not in hits:
            raise ValueError("no `particle_id` column to auto-calibrate from; pass r_c and r_scale explicitly")
        mean_r = mean_consecutive_hit_distance(hits)
        r_c = r_c if r_c is not None else r_c_factor * mean_r
        r_scale = r_scale if r_scale is not None else mean_r

    segments = build_segments(x, y, r_c, layer_ids=layer_ids)
    t = build_weight_matrix(
        segments,
        n=n,
        r_c=r_c,
        r_scale=r_scale,
        inhibition=inhibition,
        type1_scale=type1_scale,
        type2_scale=type2_scale,
        use_type2=use_type2,
    )
    history = relax(
        t,
        n_iterations=n_iterations,
        dt_over_tau=dt_over_tau,
        gain=gain,
        init_spread=init_spread,
        rng=np.random.default_rng(seed),
    )

    active = on_segments(segments, history.f_v[-1])
    chains = chain_tracks(active)

    score = None
    if "particle_id" in hits:
        score = score_against_truth(chains, dict(enumerate(hits["particle_id"])))

    return segments, history, chains, score


def _cmd_run(args: argparse.Namespace) -> None:
    hits = pd.read_csv(args.hits)
    segments, history, chains, score = run(
        hits,
        use_type2=args.type2,
        n_iterations=args.n_iterations,
        seed=args.seed,
    )

    print(f"{len(segments)} candidate segments (R_c-limited)")
    print(f"converged in {len(history) - 1} iteration(s), final energy {history.energy[-1]:.4f}")
    print(f"found {len(chains)} chain(s): {chains}")
    if score is not None:
        print(f"score: {score}")

    if args.save:
        xy = hits[["x", "y"]].to_numpy()
        fig = plot_iterations(xy, segments, history, layout=args.layout, invert_y=not args.no_invert_y)
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"saved {args.save}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hopfield_tracking", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_p = subparsers.add_parser("run", help="Run the network on one event's hits")
    run_p.add_argument("--hits", required=True, help="CSV with at least x, y (and optionally particle_id) columns")
    run_p.add_argument(
        "--type2", action="store_true", help="Also enable type-2 (circle-fit) coefficients (off by default, see README)"
    )
    run_p.add_argument("--n-iterations", type=int, default=60)
    run_p.add_argument("--seed", type=int, default=None)
    run_p.add_argument("--save", default=None, help="Save the fig.-8-style panel figure to this path")
    run_p.add_argument(
        "--layout",
        choices=["grid", "row"],
        default="grid",
        help="Panel layout for --save: 2x2-ish grid like the paper's fig. 8 (default), or a single row",
    )
    run_p.add_argument(
        "--no-invert-y",
        action="store_true",
        help="Don't flip the y-axis (by default it's flipped so the vertex renders at the bottom, matching fig. 8)",
    )
    run_p.set_defaults(func=_cmd_run)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
