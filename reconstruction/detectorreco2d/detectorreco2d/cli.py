"""CLI entry point: reconstruct tracks/clusters from a previously written
`detectorsim2d` run.

    python -m detectorreco2d.cli run --sim-dir out/sim --config configs/default.yaml \\
        --output-dir out/reco --format arrow --seed 42
"""

from __future__ import annotations

import argparse

import numpy as np
from detectorsim2d.io import read_deposits
from detectorsim2d.io import read_run as read_sim_run

from .config import load_config
from .io import write_run
from .reconstruct import reconstruct


def _cmd_run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.seed is not None:
        config.seed = args.seed

    particles, _hits = read_sim_run(args.sim_dir, args.sim_format)
    deposits = read_deposits(args.sim_dir, args.sim_format)

    rng = np.random.default_rng(config.seed)
    tracks, clusters = reconstruct(particles, deposits, config, rng)

    paths = write_run(args.output_dir, args.format, tracks, clusters if len(clusters) else None)

    print(f"Reconstructed {len(particles)} particle(s) from {args.sim_dir}, seed={config.seed}")
    print(f"  tracks:   {len(tracks):>8} rows -> {paths['tracks']}")
    if "clusters" in paths:
        print(f"  clusters: {len(clusters):>8} rows -> {paths['clusters']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="detectorreco2d", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_p = subparsers.add_parser("run", help="Reconstruct tracks/clusters from a detectorsim2d run")
    run_p.add_argument("--sim-dir", required=True, help="Directory previously written by `detectorsim2d.cli run`")
    run_p.add_argument("--sim-format", choices=["csv", "arrow"], default="csv", help="Format of --sim-dir")
    run_p.add_argument(
        "--config", default=None, help="YAML RecoConfig file (defaults -- no smearing -- if omitted)"
    )
    run_p.add_argument("--seed", type=int, default=None, help="Override seed from config")
    run_p.add_argument("--output-dir", default="out", help="Directory to write tracks/clusters into")
    run_p.add_argument("--format", choices=["csv", "arrow"], default="csv")
    run_p.set_defaults(func=_cmd_run)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
