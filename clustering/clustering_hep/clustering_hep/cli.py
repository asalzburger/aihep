"""CLI entry point: run the simulation from a config file and write output,
or visualize a previously written run.

    python -m clustering_hep.cli run --config configs/default.yaml --n-events 100 \\
        --output-dir out/ --format arrow --seed 42

    python -m clustering_hep.cli visualize --output-dir out/ --format arrow --event-id 0
"""

from __future__ import annotations

import argparse

import numpy as np

from .io import read_run, write_run
from .sim import SimConfig, cluster_hits, digitize_events, load_config, simulate_events


def run_simulation(config: SimConfig) -> tuple:
    """Run the full pipeline. Returns (hits, clusters, truth) DataFrames."""
    seed_seq = np.random.SeedSequence(config.seed)
    rng_sim, rng_dig = (np.random.default_rng(s) for s in seed_seq.spawn(2))

    grids, truth = simulate_events(config, rng=rng_sim)
    hits = digitize_events(grids, config, rng=rng_dig)
    hits, clusters = cluster_hits(hits, config.detector, config.cluster_connectivity)
    return hits, clusters, truth


def _cmd_run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.n_events is not None:
        config.n_events = args.n_events
    if args.seed is not None:
        config.seed = args.seed

    hits, clusters, truth = run_simulation(config)
    paths = write_run(args.output_dir, args.format, hits, clusters, truth)

    print(f"Simulated {config.n_events} event(s), seed={config.seed}")
    print(f"  hits:     {len(hits):>8} rows -> {paths['hits']}")
    print(f"  clusters: {len(clusters):>8} rows -> {paths['clusters']}")
    print(f"  truth:    {len(truth):>8} rows -> {paths['truth']}")


def _cmd_visualize(args: argparse.Namespace) -> None:
    from .vis import plot_event  # deferred: matplotlib import only needed here

    hits, clusters, truth = read_run(args.output_dir, args.format)
    config = load_config(args.config)
    zoom = tuple(args.zoom) if args.zoom else None
    fig = plot_event(
        hits,
        clusters,
        truth,
        config.detector,
        args.event_id,
        zoom=zoom,
        grid=args.grid,
        readout_threshold=args.readout_threshold,
        digital=args.digital,
    )

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved {args.save}")
    else:
        import matplotlib.pyplot as plt

        plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clustering_hep", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_p = subparsers.add_parser("run", help="Run the simulation and write output tables")
    run_p.add_argument("--config", default=None, help="YAML config file (defaults if omitted)")
    run_p.add_argument("--n-events", type=int, default=None, help="Override n_events from config")
    run_p.add_argument("--seed", type=int, default=None, help="Override seed from config")
    run_p.add_argument("--output-dir", default="out", help="Directory to write hits/clusters/truth into")
    run_p.add_argument("--format", choices=["csv", "arrow"], default="csv")
    run_p.set_defaults(func=_cmd_run)

    viz_p = subparsers.add_parser("visualize", help="Plot a single event from a written run")
    viz_p.add_argument("--config", default=None, help="YAML config file used for the run (for detector geometry)")
    viz_p.add_argument("--output-dir", default="out", help="Directory previously written by `run`")
    viz_p.add_argument("--format", choices=["csv", "arrow"], default="csv")
    viz_p.add_argument("--event-id", type=int, default=0)
    viz_p.add_argument(
        "--zoom",
        type=int,
        nargs=2,
        metavar=("NX", "NY"),
        default=None,
        help="Show only an NX x NY pixel window centered on the event's largest cluster",
    )
    viz_p.add_argument("--grid", action="store_true", help="Overlay light pixel-boundary gridlines")
    viz_p.add_argument(
        "--readout-threshold",
        type=float,
        default=0.15,  # keep in sync with vis.DEFAULT_READOUT_THRESHOLD
        help="Pixels with charge at or below this are not read out (default: 0.15)",
    )
    viz_p.add_argument(
        "--digital",
        action="store_true",
        help="Show pixels above the readout threshold as flat on/off instead of charge-graded color",
    )
    viz_p.add_argument("--save", default=None, help="Save the figure to this path instead of showing it")
    viz_p.set_defaults(func=_cmd_visualize)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
