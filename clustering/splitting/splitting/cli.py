"""CLI entry point: apply a cluster splitter to a previously-written
`sensor` run (hits/clusters/truth/contributions), writing a new, complete
run directory with the split hits/clusters (truth/contributions pass
through unchanged, since splitting never changes ground truth).

    python -m splitting.cli run --splitter truth \\
        --input-dir resources/p123 --output-dir out/p123_truth_split --format arrow
"""

from __future__ import annotations

import argparse

import pandas as pd

from .io import read_run, write_run
from .pipeline import apply_splitter
from .registry import SPLITTERS


def run_split(
    splitter_name: str, hits: pd.DataFrame, clusters: pd.DataFrame, truth: pd.DataFrame, contributions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply the named splitter. Returns (hits, clusters, truth,
    contributions) ready to write out as a complete run."""
    splitter = SPLITTERS[splitter_name]()
    new_hits, new_clusters = apply_splitter(splitter, hits, clusters, contributions)
    return new_hits, new_clusters, truth, contributions


def _cmd_run(args: argparse.Namespace) -> None:
    hits, clusters, truth, contributions = read_run(args.input_dir, args.format)
    new_hits, new_clusters, truth, contributions = run_split(args.splitter, hits, clusters, truth, contributions)
    paths = write_run(args.output_dir, args.format, new_hits, new_clusters, truth, contributions)

    print(f"Split with {args.splitter!r}: {len(clusters)} cluster(s) -> {len(new_clusters)} cluster(s)")
    print(f"  hits:          {len(new_hits):>8} rows -> {paths['hits']}")
    print(f"  clusters:      {len(new_clusters):>8} rows -> {paths['clusters']}")
    print(f"  truth:         {len(truth):>8} rows -> {paths['truth']}")
    print(f"  contributions: {len(contributions):>8} rows -> {paths['contributions']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="splitting", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_p = subparsers.add_parser("run", help="Apply a splitter to a run directory, writing a new split run")
    run_p.add_argument(
        "--splitter", choices=sorted(SPLITTERS), default="truth", help="Which splitting algorithm to use"
    )
    run_p.add_argument(
        "--input-dir", required=True, help="Directory containing hits/clusters/truth/contributions to split"
    )
    run_p.add_argument("--output-dir", required=True, help="Directory to write the split run into")
    run_p.add_argument("--format", choices=["csv", "arrow"], default="arrow")
    run_p.set_defaults(func=_cmd_run)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
