"""Evaluate a trained b-tagger model on an *independent* reconstructed run
-- one simulated with a different seed than training, so this measures
generalization rather than memorization: accuracy, confusion matrix, ROC
curve, and the score distribution itself (light jets vs. b-jets, how well
separated the two actually are)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, confusion_matrix, roc_curve
from viz_style import Theme, palette

from .dataset import build_dataset, standardize
from .model import BTaggerMLP

LIGHT_COLOR = palette.CATEGORICAL_OKABE_ITO[0]
B_JET_COLOR = palette.CATEGORICAL_OKABE_ITO[1]

CLASSES = ["light", "b-jet"]


def evaluate_model(
    model: BTaggerMLP, preprocessing: dict, tracks: pd.DataFrame, clusters: pd.DataFrame
) -> dict:
    dataset = build_dataset(tracks, clusters, n_track_slots=preprocessing["n_track_slots"])
    features = standardize(dataset.features, preprocessing["mean"], preprocessing["std"])
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        score = model(torch.from_numpy(features).to(device)).cpu().numpy()

    predicted = (score >= 0.5).astype(int)
    cm = confusion_matrix(dataset.is_b_jet, predicted, labels=[0, 1])
    accuracy = float((predicted == dataset.is_b_jet).mean())
    fpr, tpr, _ = roc_curve(dataset.is_b_jet, score)
    roc_auc = float(auc(fpr, tpr))

    return dict(
        dataset=dataset,
        score=score,
        predicted=predicted,
        confusion_matrix=cm,
        accuracy=accuracy,
        fpr=fpr,
        tpr=tpr,
        roc_auc=roc_auc,
    )


def plot_roc(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, save_path=None, theme: Theme | None = None):
    import matplotlib.pyplot as plt
    from viz_style.mpl import style_axes

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color=B_JET_COLOR, linewidth=2, label=f"b-tagger (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="0.7", linestyle="--", linewidth=1, zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    style_axes(
        ax, theme, spatial=False, title="b-tagger ROC",
        xlabel="false positive rate (light-jet mistag rate)", ylabel="true positive rate (b-jet efficiency)",
        legend=True, legend_loc="lower right",
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrix(cm: np.ndarray, save_path=None, theme: Theme | None = None, normalize: bool = True):
    """`normalize=True` (default) shows each row (true class) as a fraction
    of its own row total -- i.e. per-class recall, the b-tagging efficiency
    and light-jet mistag rate directly -- rather than raw (imbalanced,
    since b-jets are the minority class) counts."""
    import matplotlib.pyplot as plt
    from viz_style.mpl import style_axes

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        display = np.divide(cm, row_sums, out=np.zeros(cm.shape, dtype=float), where=row_sums > 0)
    else:
        display = cm

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(
        display, cmap=palette.SEQUENTIAL_CONFUSION_CMAP, vmin=0.0 if normalize else None, vmax=1.0 if normalize else None
    )
    ax.set_xticks(range(len(CLASSES)), labels=CLASSES)
    ax.set_yticks(range(len(CLASSES)), labels=CLASSES)
    style_axes(ax, theme, spatial=False, title="b-tagger confusion matrix", xlabel="predicted", ylabel="true")
    threshold = display.max() / 2 if display.max() > 0 else 0
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            text = f"{display[i, j]:.2f}" if normalize else str(cm[i, j])
            ax.text(j, i, text, ha="center", va="center", color="white" if display[i, j] > threshold else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_score_distribution(score: np.ndarray, is_b_jet: np.ndarray, bins: int = 40, save_path=None, theme=None):
    """Classifier output score, light jets vs. b-jets, overlaid -- how
    cleanly the two classes actually separate, which the accuracy/AUC
    summarize into single numbers but this shows directly."""
    import matplotlib.pyplot as plt
    from viz_style.mpl import style_axes

    fig, ax = plt.subplots(figsize=(6, 4))
    edges = np.linspace(0.0, 1.0, bins + 1)
    ax.hist(score[is_b_jet == 0], bins=edges, color=LIGHT_COLOR, alpha=0.6, density=True, label="light jets")
    ax.hist(score[is_b_jet == 1], bins=edges, color=B_JET_COLOR, alpha=0.6, density=True, label="b-jets")
    style_axes(
        ax, theme, spatial=False, title="b-tagger score", xlabel="P(b-jet)", ylabel="density", legend=True
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
