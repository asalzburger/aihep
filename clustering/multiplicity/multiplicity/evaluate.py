"""Evaluate a trained multiplicity model on an independent run directory:
confusion matrix, accuracy, and one-vs-rest ROC curves per class.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import auc, confusion_matrix, roc_curve
from viz_style import Theme, palette

from .dataset import build_dataset
from .io import Format
from .model import MAX_PARTICLES, MIN_PARTICLES, MultiplicityMLP, decode_score

# Fixed categorical hues (blue/orange/aqua), one per class, in the same
# order used elsewhere in this repo -- never the default matplotlib cycle.
CLASS_COLORS = palette.CLASS_COLORS


def _class_score(score: np.ndarray, n_class: int) -> np.ndarray:
    """One-vs-rest ranking score for `n_class`, derived from the raw [0, 1]
    ordinal output: the highest class ranks directly on the score, the
    lowest ranks on the inverted score, and any class in between (bounded
    on both sides) ranks on proximity to its own target value."""
    target = (n_class - MIN_PARTICLES) / (MAX_PARTICLES - MIN_PARTICLES)
    if n_class == MIN_PARTICLES:
        return 1.0 - score
    if n_class == MAX_PARTICLES:
        return score
    return -np.abs(score - target)


def evaluate_model(
    model: MultiplicityMLP,
    matrix_shape: tuple[int, int],
    input_dirs: list[str | Path],
    fmt: Format = "arrow",
) -> dict:
    dataset = build_dataset(input_dirs, fmt=fmt, matrix_shape=matrix_shape)
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        score = model(torch.from_numpy(dataset.matrices).to(device)).cpu().numpy()

    predicted = decode_score(score)
    classes = list(range(MIN_PARTICLES, MAX_PARTICLES + 1))

    cm = confusion_matrix(dataset.n_particles, predicted, labels=classes)
    accuracy = float((predicted == dataset.n_particles).mean())

    roc = {}
    for n_class in classes:
        y_true = (dataset.n_particles == n_class).astype(int)
        y_score = _class_score(score, n_class)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc[n_class] = (fpr, tpr, float(auc(fpr, tpr)))

    return dict(
        dataset=dataset,
        score=score,
        predicted=predicted,
        classes=classes,
        confusion_matrix=cm,
        accuracy=accuracy,
        roc=roc,
    )


def plot_roc(roc: dict, save_path: str | Path | None = None, theme: Theme | None = None):
    import matplotlib.pyplot as plt
    from viz_style.mpl import style_axes

    fig, ax = plt.subplots(figsize=(5, 5))
    for color, (n_class, (fpr, tpr, roc_auc)) in zip(CLASS_COLORS, roc.items()):
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{n_class} particle(s) (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="0.7", linestyle="--", linewidth=1, zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    style_axes(
        ax, theme, spatial=False, title="Multiplicity classifier: one-vs-rest ROC",
        xlabel="false positive rate", ylabel="true positive rate", legend=True, legend_loc="lower right",
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    cm: np.ndarray, classes: list[int], save_path: str | Path | None = None, theme: Theme | None = None,
    normalize: bool = False,
):
    """`normalize=True` shows each row (true class) as a fraction of its own
    row total -- i.e. per-class recall -- instead of raw counts, which is
    usually the more readable view whenever classes are imbalanced. A true
    class with zero examples gets an all-zero row rather than dividing by
    zero."""
    import matplotlib.pyplot as plt
    from viz_style.mpl import style_axes

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        display = np.divide(cm, row_sums, out=np.zeros(cm.shape, dtype=float), where=row_sums > 0)
    else:
        display = cm

    fig, ax = plt.subplots(figsize=(4.5, 4))
    if normalize:
        im = ax.imshow(display, cmap=palette.SEQUENTIAL_CONFUSION_CMAP, vmin=0.0, vmax=1.0)
    else:
        im = ax.imshow(display, cmap=palette.SEQUENTIAL_CONFUSION_CMAP)
    ax.set_xticks(range(len(classes)), labels=[str(c) for c in classes])
    ax.set_yticks(range(len(classes)), labels=[str(c) for c in classes])
    style_axes(
        ax, theme, spatial=False, title="Confusion matrix",
        xlabel="predicted n particles", ylabel="true n particles",
    )
    threshold = display.max() / 2 if display.max() > 0 else 0
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = f"{display[i, j]:.2f}" if normalize else str(cm[i, j])
            ax.text(
                j, i, text, ha="center", va="center", color="white" if display[i, j] > threshold else "black"
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
