import numpy as np
from viz_style import PRESENT, PRINT

from multiplicity.evaluate import plot_confusion_matrix, plot_roc


def _fake_roc():
    fpr = np.array([0.0, 0.5, 1.0])
    tpr = np.array([0.0, 0.8, 1.0])
    return {1: (fpr, tpr, 0.9), 2: (fpr, tpr, 0.8), 3: (fpr, tpr, 0.85)}


def test_plot_roc_present_drops_title_but_keeps_axes_and_legend():
    # statistical plots (spatial=False) never lose axes/labels/legend --
    # only ROC's own title is theme-gated.
    fig_print = plot_roc(_fake_roc(), theme=PRINT)
    ax_print = fig_print.axes[0]
    assert ax_print.get_title() != ""
    assert ax_print.get_xlabel() == "false positive rate"
    assert ax_print.get_ylabel() == "true positive rate"
    assert ax_print.get_legend() is not None

    fig_present = plot_roc(_fake_roc(), theme=PRESENT)
    ax_present = fig_present.axes[0]
    assert ax_present.get_title() == ""
    assert ax_present.get_xlabel() == "false positive rate"
    assert ax_present.get_ylabel() == "true positive rate"
    assert ax_present.get_legend() is not None


def test_plot_confusion_matrix_present_drops_title_but_keeps_axes():
    cm = np.array([[5, 1, 0], [0, 6, 1], [0, 0, 7]])
    classes = [1, 2, 3]

    fig_print = plot_confusion_matrix(cm, classes, theme=PRINT)
    ax_print = fig_print.axes[0]
    assert ax_print.get_title() != ""
    assert ax_print.get_xlabel() == "predicted n particles"

    fig_present = plot_confusion_matrix(cm, classes, theme=PRESENT)
    ax_present = fig_present.axes[0]
    assert ax_present.get_title() == ""
    assert ax_present.get_xlabel() == "predicted n particles"


def test_plot_roc_default_theme_matches_print():
    fig = plot_roc(_fake_roc())
    assert fig.axes[0].get_title() != ""
