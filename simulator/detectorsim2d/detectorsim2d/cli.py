"""CLI entry point: run the simulation from a config file and write output,
or visualize a previously written run.

    python -m detectorsim2d.cli run --config configs/default.yaml --n-events 100 \\
        --output-dir out/ --format arrow --seed 42

    python -m detectorsim2d.cli visualize --config configs/default.yaml --output-dir out/ --format arrow --event-id 0
"""

from __future__ import annotations

import argparse

from .config import GUN_MODES, load_config
from .io import read_deposits, read_run, write_run
from .simulate import simulate_events


def _cmd_run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.n_events is not None:
        config.n_events = args.n_events
    if args.seed is not None:
        config.seed = args.seed
    if args.mode is not None:
        config.gun.mode = args.mode

    particles, hits, deposits = simulate_events(config)
    # a tracker-only config has no calorimeter, so write no deposits file
    paths = write_run(
        args.output_dir, args.format, particles, hits, deposits if len(deposits) else None
    )

    print(f"Simulated {config.n_events} event(s), mode={config.gun.mode}, seed={config.seed}")
    print(f"  particles: {len(particles):>8} rows -> {paths['particles']}")
    print(f"  hits:      {len(hits):>8} rows -> {paths['hits']}")
    if "deposits" in paths:
        print(f"  deposits:  {len(deposits):>8} rows -> {paths['deposits']}")


def _cmd_visualize(args: argparse.Namespace) -> None:
    from viz_style import PRESENT, PRINT

    from .vis import plot_event, plot_lego  # deferred: matplotlib import only needed here

    particles, hits = read_run(args.output_dir, args.format)
    deposits = read_deposits(args.output_dir, args.format)
    config = load_config(args.config)
    tracker_boundary = args.tracker_boundary if args.tracker_boundary is not None else config.tracker_boundary
    theme = PRESENT if args.style == "present" else PRINT

    if args.view == "lego":
        fig = plot_lego(
            deposits, config.layers, args.event_id, theme=theme,
            phi_range=tuple(args.phi_range) if args.phi_range else None,
        )
    else:
        fig = plot_event(
            particles,
            hits,
            config.layers,
            args.event_id,
            track_length=args.track_length,
            tracker_boundary=tracker_boundary,
            theme=theme,
            deposits=deposits if len(deposits) else None,
            field=config.field_regions,
            world_radius=config.world_radius,
            max_path_length=config.max_path_length,
        )

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved {args.save}")
    else:
        import matplotlib.pyplot as plt

        plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="detectorsim2d", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_p = subparsers.add_parser("run", help="Run the simulation and write output tables")
    run_p.add_argument("--config", default=None, help="YAML config file (defaults if omitted)")
    run_p.add_argument("--n-events", type=int, default=None, help="Override n_events from config")
    run_p.add_argument("--seed", type=int, default=None, help="Override seed from config")
    run_p.add_argument(
        "--mode",
        choices=GUN_MODES,
        default=None,
        help="Override gun.mode from config: standard (independent particles, the default), "
        "jets (same multiplicity, grouped into 2-4 collimated sprays), "
        "anomaly (standard, plus an injected back-to-back calo+dimuon cluster in some events)",
    )
    run_p.add_argument("--output-dir", default="out", help="Directory to write particles/hits into")
    run_p.add_argument("--format", choices=["csv", "arrow"], default="csv")
    run_p.set_defaults(func=_cmd_run)

    viz_p = subparsers.add_parser("visualize", help="Plot a single event from a written run")
    viz_p.add_argument("--config", default=None, help="YAML config file used for the run (for detector layout)")
    viz_p.add_argument("--output-dir", default="out", help="Directory previously written by `run`")
    viz_p.add_argument("--format", choices=["csv", "arrow"], default="csv")
    viz_p.add_argument("--event-id", type=int, default=0)
    viz_p.add_argument(
        "--view",
        choices=["xy", "lego"],
        default="xy",
        help="xy (default): the detector in the plane. lego: the calorimeter unrolled in azimuth, "
        "one row per sampling layer.",
    )
    viz_p.add_argument(
        "--phi-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
        help="lego view only: zoom to this azimuth range in degrees. Zoom in far enough and "
        "individual cell edges are drawn, which is how the ECAL's half-cell stagger becomes visible.",
    )
    viz_p.add_argument(
        "--track-length", type=float, default=100.0, help="Draw length for particles with no hits"
    )
    viz_p.add_argument(
        "--tracker-boundary",
        type=float,
        default=None,
        help="Outer tracker radius; caps how far a drawn arc extends instead of looping back inward "
        "(overrides the config's tracker_boundary if both are given)",
    )
    viz_p.add_argument("--save", default=None, help="Save the figure to this path instead of showing it")
    viz_p.add_argument(
        "--style",
        choices=["print", "present"],
        default="print",
        help="print (default): full titles/axes/labels. present: no title, no axes -- for a slide.",
    )
    viz_p.set_defaults(func=_cmd_visualize)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
