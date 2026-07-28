"""CLI entry point: train and evaluate a small MLP that predicts a
cluster's particle multiplicity (1, 2, or 3) from its fixed-size pixel
matrix.

    python -m multiplicity.cli train \\
        --input-dir ../sensor/p1 ../sensor/p2 ../sensor/p3 \\
        --format arrow --epochs 50 --output model.pt

    python -m multiplicity.cli evaluate \\
        --model model.pt --input-dir ../sensor/p123 --format arrow --save-dir plots/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .evaluate import evaluate_model, plot_confusion_matrix, plot_roc
from .train import load_checkpoint, save_checkpoint, train_model


def _cmd_train(args: argparse.Namespace) -> None:
    model, matrix_shape, history = train_model(
        args.input_dir,
        fmt=args.format,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
    )
    save_checkpoint(args.output, model, matrix_shape)
    print(f"\nSaved model to {args.output} (input matrix shape {matrix_shape})")
    print(f"Final validation accuracy: {history.val_accuracy[-1]:.3f}")


def _cmd_evaluate(args: argparse.Namespace) -> None:
    model, matrix_shape = load_checkpoint(args.model, device=args.device)
    result = evaluate_model(model, matrix_shape, args.input_dir, fmt=args.format)

    print(f"Accuracy: {result['accuracy']:.3f}")
    print("Confusion matrix (rows=true, cols=predicted):")
    print(result["confusion_matrix"])
    for n_class, (_, _, roc_auc) in result["roc"].items():
        print(f"  AUC (n={n_class} vs rest): {roc_auc:.3f}")

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_roc(result["roc"], save_path=save_dir / "roc.png")
        plot_confusion_matrix(result["confusion_matrix"], result["classes"], save_path=save_dir / "confusion_matrix.png")
        print(f"Saved plots to {save_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="multiplicity", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_p = subparsers.add_parser("train", help="Train the multiplicity MLP")
    train_p.add_argument("--input-dir", nargs="+", required=True, help="One or more sensor-shaped run directories")
    train_p.add_argument("--format", choices=["csv", "arrow"], default="arrow")
    train_p.add_argument("--epochs", type=int, default=50)
    train_p.add_argument("--batch-size", type=int, default=256)
    train_p.add_argument("--val-fraction", type=float, default=0.15, help="Held out for in-training validation")
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    train_p.add_argument("--seed", type=int, default=0)
    train_p.add_argument("--output", default="model.pt", help="Path to write the trained checkpoint to")
    train_p.set_defaults(func=_cmd_train)

    eval_p = subparsers.add_parser(
        "evaluate", help="Evaluate a trained model on an independent run directory (ROC, confusion matrix)"
    )
    eval_p.add_argument("--model", required=True, help="Path to a checkpoint written by `train`")
    eval_p.add_argument("--input-dir", nargs="+", required=True, help="Independent run directory/directories")
    eval_p.add_argument("--format", choices=["csv", "arrow"], default="arrow")
    eval_p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    eval_p.add_argument("--save-dir", default=None, help="Directory to save ROC/confusion-matrix plots into")
    eval_p.set_defaults(func=_cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
