"""CLI entry point: simulate jets+b-jets, reconstruct them, make the
validation plots, and train/evaluate the b-tagger -- five subcommands
mirroring the pipeline's own stages. `train` and `evaluate` are meant to be
pointed at two *separate* reconstructed runs (different seeds -- e.g.
`out/reco_train` and `out/reco_test`), so `evaluate` measures generalization
to an independent dataset rather than replaying the training set.

    python -m flavor_tagging.cli simulate --config configs/jets_bjets.yaml \\
        --output-dir out/sim_train --format arrow --seed 42
    python -m flavor_tagging.cli simulate --config configs/jets_bjets.yaml \\
        --output-dir out/sim_test --format arrow --seed 4242

    python -m flavor_tagging.cli reconstruct --sim-dir out/sim_train --config configs/reco.yaml \\
        --output-dir out/reco_train --format arrow
    python -m flavor_tagging.cli reconstruct --sim-dir out/sim_test --config configs/reco.yaml \\
        --output-dir out/reco_test --format arrow

    python -m flavor_tagging.cli validate --reco-dir out/reco_train --format arrow --output-dir out/plots

    python -m flavor_tagging.cli train --reco-dir out/reco_train --format arrow --output model.pt
    python -m flavor_tagging.cli evaluate --model model.pt --reco-dir out/reco_test --format arrow \\
        --save-dir out/tagger_plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

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

from .evaluate import evaluate_model, plot_confusion_matrix, plot_roc, plot_score_distribution
from .pipeline import RECO_CONFIG_PATH, SIM_CONFIG_PATH
from .train import load_checkpoint, save_checkpoint, train_model
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


def _cmd_train(args: argparse.Namespace) -> None:
    tracks, clusters = read_reco_run(args.reco_dir, args.format)
    model, preprocessing, history = train_model(
        tracks,
        clusters,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
    )
    save_checkpoint(args.output, model, preprocessing)
    print(
        f"\nSaved model to {args.output} "
        f"({preprocessing['n_track_slots']} track slots, {len(preprocessing['feature_names'])} features)"
    )
    print(f"Final validation accuracy: {history.val_accuracy[-1]:.3f}")


def _cmd_evaluate(args: argparse.Namespace) -> None:
    model, preprocessing = load_checkpoint(args.model, device=args.device)
    tracks, clusters = read_reco_run(args.reco_dir, args.format)
    result = evaluate_model(model, preprocessing, tracks, clusters)

    print(f"Evaluated on {len(result['dataset'].is_b_jet)} jet(s) from {args.reco_dir}")
    print(f"  accuracy: {result['accuracy']:.3f}")
    print(f"  AUC:      {result['roc_auc']:.3f}")
    print("  confusion matrix (rows=true [light, b-jet], cols=predicted):")
    print(result["confusion_matrix"])

    if args.save_dir:
        from viz_style import PRESENT, PRINT

        theme = PRESENT if args.style == "present" else PRINT
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_roc(result["fpr"], result["tpr"], result["roc_auc"], save_path=save_dir / "roc.png", theme=theme)
        plot_confusion_matrix(result["confusion_matrix"], save_path=save_dir / "confusion_matrix.png", theme=theme)
        plot_score_distribution(
            result["score"], result["dataset"].is_b_jet, save_path=save_dir / "score.png", theme=theme
        )
        print(f"Saved plots to {save_dir}")


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

    train_p = subparsers.add_parser("train", help="Train the b-tagger MLP on a reconstructed run")
    train_p.add_argument("--reco-dir", default="out/reco_train", help="Directory previously written by `reconstruct`")
    train_p.add_argument("--format", choices=["csv", "arrow"], default="arrow", help="Format of --reco-dir")
    train_p.add_argument("--epochs", type=int, default=50)
    train_p.add_argument("--batch-size", type=int, default=64)
    train_p.add_argument("--val-fraction", type=float, default=0.15, help="Held out for in-training validation")
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    train_p.add_argument("--seed", type=int, default=0)
    train_p.add_argument("--output", default="model.pt", help="Path to write the trained checkpoint to")
    train_p.set_defaults(func=_cmd_train)

    eval_p = subparsers.add_parser(
        "evaluate", help="Evaluate a trained b-tagger on an independent reconstructed run (ROC, confusion matrix)"
    )
    eval_p.add_argument("--model", required=True, help="Path to a checkpoint written by `train`")
    eval_p.add_argument(
        "--reco-dir", default="out/reco_test", help="Independent reconstructed run -- NOT the one used for `train`"
    )
    eval_p.add_argument("--format", choices=["csv", "arrow"], default="arrow", help="Format of --reco-dir")
    eval_p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    eval_p.add_argument("--save-dir", default=None, help="Directory to save ROC/confusion-matrix/score plots into")
    eval_p.add_argument(
        "--style",
        choices=["print", "present"],
        default="print",
        help="print (default): full titles. present: no title -- for a slide. Only affects --save-dir plots.",
    )
    eval_p.set_defaults(func=_cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
