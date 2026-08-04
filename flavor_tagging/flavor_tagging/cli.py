"""CLI entry point: simulate jets+b-jets, reconstruct them, and make the
validation plots -- three subcommands mirroring the pipeline's own three
stages.

    python -m flavor_tagging.cli simulate --config configs/jets_bjets.yaml \\
        --output-dir out/sim --format arrow --seed 42

    python -m flavor_tagging.cli reconstruct --sim-dir out/sim --config configs/reco.yaml \\
        --output-dir out/reco --format arrow

    python -m flavor_tagging.cli validate --reco-dir out/reco --format arrow --output-dir out/plots
"""

from __future__ import annotations

import argparse

import numpy as np
from detectorreco2d.config import load_config as load_reco_config
from detectorreco2d.io import read_run as read_reco_run
from detectorreco2d.io import write_run as write_reco_run
from detectorreco2d.reconstruct import reconstruct
from detectorsim2d.config import load_config as load_sim_config
from detectorsim2d.io import read_deposits
from detectorsim2d.io import read_run as read_sim_run
from detectorsim2d.io import write_run as write_sim_run
from detectorsim2d.simulate import simulate_events

from .pipeline import RECO_CONFIG_PATH, SIM_CONFIG_PATH
from .vis import make_validation_plots


def _cmd_simulate(args: argparse.Namespace) -> None:
    config = load_sim_config(args.config or SIM_CONFIG_PATH)
    if args.n_events is not None:
        config.n_events = args.n_events
    if args.seed is not None:
        config.seed = args.seed

    particles, hits, deposits = simulate_events(config)
    paths = write_sim_run(args.output_dir, args.format, particles, hits, deposits if len(deposits) else None)

    in_a_jet = particles[particles["jet_id"] != -1]
    per_jet_flavor = in_a_jet.groupby(["event_id", "jet_id"])["is_b_jet"].first()
    print(
        f"Simulated {config.n_events} event(s), seed={config.seed}: "
        f"{per_jet_flavor.size} jets, {int(per_jet_flavor.sum())} of them b-jets"
    )
    print(f"  particles: {len(particles):>8} rows -> {paths['particles']}")
    print(f"  hits:      {len(hits):>8} rows -> {paths['hits']}")
    if "deposits" in paths:
        print(f"  deposits:  {len(deposits):>8} rows -> {paths['deposits']}")


def _cmd_reconstruct(args: argparse.Namespace) -> None:
    config = load_reco_config(args.config or RECO_CONFIG_PATH)
    if args.seed is not None:
        config.seed = args.seed

    particles, _hits = read_sim_run(args.sim_dir, args.sim_format)
    deposits = read_deposits(args.sim_dir, args.sim_format)

    rng = np.random.default_rng(config.seed)
    tracks, clusters = reconstruct(particles, deposits, config, rng)
    paths = write_reco_run(args.output_dir, args.format, tracks, clusters if len(clusters) else None)

    print(f"Reconstructed {len(particles)} particle(s) from {args.sim_dir}, seed={config.seed}")
    print(f"  tracks:   {len(tracks):>8} rows -> {paths['tracks']}")
    if "clusters" in paths:
        print(f"  clusters: {len(clusters):>8} rows -> {paths['clusters']}")


def _cmd_validate(args: argparse.Namespace) -> None:
    from viz_style import PRESENT, PRINT

    tracks, clusters = read_reco_run(args.reco_dir, args.format)
    theme = PRESENT if args.style == "present" else PRINT
    paths = make_validation_plots(tracks, clusters, args.output_dir, theme=theme)

    print(f"Wrote {len(paths)} validation plot(s) to {args.output_dir}:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flavor_tagging", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sim_p = subparsers.add_parser("simulate", help="Simulate a run of jets/b-jets")
    sim_p.add_argument("--config", default=None, help=f"YAML SimConfig file (default: {SIM_CONFIG_PATH})")
    sim_p.add_argument("--n-events", type=int, default=None, help="Override n_events from config")
    sim_p.add_argument("--seed", type=int, default=None, help="Override seed from config")
    sim_p.add_argument("--output-dir", default="out/sim", help="Directory to write particles/hits/deposits into")
    sim_p.add_argument("--format", choices=["csv", "arrow"], default="arrow")
    sim_p.set_defaults(func=_cmd_simulate)

    reco_p = subparsers.add_parser("reconstruct", help="Reconstruct tracks/clusters from a simulated run")
    reco_p.add_argument("--sim-dir", default="out/sim", help="Directory previously written by `simulate`")
    reco_p.add_argument("--sim-format", choices=["csv", "arrow"], default="arrow", help="Format of --sim-dir")
    reco_p.add_argument("--config", default=None, help=f"YAML RecoConfig file (default: {RECO_CONFIG_PATH})")
    reco_p.add_argument("--seed", type=int, default=None, help="Override seed from config")
    reco_p.add_argument("--output-dir", default="out/reco", help="Directory to write tracks/clusters into")
    reco_p.add_argument("--format", choices=["csv", "arrow"], default="arrow")
    reco_p.set_defaults(func=_cmd_reconstruct)

    val_p = subparsers.add_parser("validate", help="Make the validation plots from a reconstructed run")
    val_p.add_argument("--reco-dir", default="out/reco", help="Directory previously written by `reconstruct`")
    val_p.add_argument("--format", choices=["csv", "arrow"], default="arrow", help="Format of --reco-dir")
    val_p.add_argument("--output-dir", default="out/plots", help="Directory to write PNGs into")
    val_p.add_argument(
        "--style",
        choices=["print", "present"],
        default="print",
        help="print (default): full titles/axes/labels. present: no title -- for a slide.",
    )
    val_p.set_defaults(func=_cmd_validate)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
