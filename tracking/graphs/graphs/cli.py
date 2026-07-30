"""CLI entry point: build a candidate track graph from a `tracksim2d` run's
hits, or visualize one on top of that run's event display.

    python -m graphs.cli build --run-dir ../../simulator/tracksim2d/out \\
        --format arrow --config configs/connection_rules.yaml --output-dir out/

    python -m graphs.cli visualize --run-dir ../../simulator/tracksim2d/out \\
        --format arrow --sim-config ../../simulator/tracksim2d/configs/barrel6.yaml \\
        --graph-config configs/connection_rules.yaml --event-id 0 --save graph0.png
"""

from __future__ import annotations

import argparse

from tracksim2d.config import load_config as load_sim_config
from tracksim2d.io import read_run

from .build import build_edges
from .config import load_config as load_graph_config
from .edm import TrackGraph
from .io import write_graph
from .truth import label_edges, purity


def _cmd_build(args: argparse.Namespace) -> None:
    _particles, hits = read_run(args.run_dir, args.format)
    graph_config = load_graph_config(args.config)

    edges = build_edges(hits, graph_config.prescription)
    if args.label_truth:
        edges = label_edges(hits, edges)
    paths = write_graph(args.output_dir, args.format, TrackGraph(nodes=hits, edges=edges))

    print(f"Built {len(edges)} edge(s) from {len(hits)} hit(s), prescription={graph_config.prescription!r}")
    if args.label_truth:
        print(f"  purity: {purity(edges):.3f} (fraction of edges connecting the same particle)")
    print(f"  edges: {len(edges):>8} rows -> {paths['edges']}")


def _cmd_visualize(args: argparse.Namespace) -> None:
    from .vis import plot_event_with_graph  # deferred: matplotlib import only needed here

    particles, hits = read_run(args.run_dir, args.format)
    sim_config = load_sim_config(args.sim_config)
    graph_config = load_graph_config(args.graph_config)
    edges = build_edges(hits, graph_config.prescription)
    if args.label_truth:
        edges = label_edges(hits, edges)

    fig = plot_event_with_graph(
        particles,
        hits,
        edges,
        sim_config.layers,
        args.event_id,
        track_length=args.track_length,
        tracker_boundary=args.tracker_boundary if args.tracker_boundary is not None else sim_config.tracker_boundary,
    )

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved {args.save}")
    else:
        import matplotlib.pyplot as plt

        plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="graphs", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_p = subparsers.add_parser("build", help="Build a candidate track graph from a tracksim2d run's hits")
    build_p.add_argument("--run-dir", required=True, help="Directory written by tracksim2d's `run` (particles/hits)")
    build_p.add_argument("--format", choices=["csv", "arrow"], default="csv")
    build_p.add_argument("--config", default=None, help="YAML prescription config (defaults to fully_connected)")
    build_p.add_argument("--output-dir", default="out", help="Directory to write edges.<format> into")
    build_p.add_argument(
        "--label-truth",
        action="store_true",
        help="Add the is_true_edge ground-truth column (same particle_id) and print purity",
    )
    build_p.set_defaults(func=_cmd_build)

    viz_p = subparsers.add_parser("visualize", help="Plot a single event's graph on top of its simulation display")
    viz_p.add_argument("--run-dir", required=True, help="Directory written by tracksim2d's `run` (particles/hits)")
    viz_p.add_argument("--format", choices=["csv", "arrow"], default="csv")
    viz_p.add_argument("--sim-config", default=None, help="tracksim2d YAML config used for the run (for the layout)")
    viz_p.add_argument("--graph-config", default=None, help="YAML prescription config (defaults to fully_connected)")
    viz_p.add_argument("--event-id", type=int, default=0)
    viz_p.add_argument(
        "--track-length", type=float, default=100.0, help="Draw length for particles with no hits"
    )
    viz_p.add_argument(
        "--tracker-boundary",
        type=float,
        default=None,
        help="Outer tracker radius; overrides --sim-config's tracker_boundary if both are given",
    )
    viz_p.add_argument(
        "--label-truth",
        action="store_true",
        help="Label edges with is_true_edge and color true edges distinctly (see graphs.truth)",
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
