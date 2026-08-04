"""Validation plots for the flavor-tagging pipeline: does the simulation +
reconstruction actually produce jets whose light/b-jet split looks the way
`configs/jets_bjets.yaml` says it should?

Every plot compares light jets against b-jets side by side or overlaid,
using the truth `is_b_jet` label carried through the whole pipeline (see
`detectorsim2d.edm.PARTICLES_COLUMNS` and `detectorreco2d.edm`) -- these are
validation plots checking the pipeline against its own knobs, not a tagging
algorithm's output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from viz_style import Theme, palette
from viz_style.mpl import style_axes

from .pipeline import summarize_jets

LIGHT_COLOR = palette.CATEGORICAL_OKABE_ITO[0]
B_JET_COLOR = palette.CATEGORICAL_OKABE_ITO[1]


def _save_or_show(fig, save_path: str | Path | None):
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return Path(save_path)
    return None


def plot_track_d0(tracks: pd.DataFrame, bins: int = 60, theme: Theme | None = None, save_path=None):
    """Smeared track impact parameter (`d0`), light jets vs b-jets: a
    b-jet's tracks come from a vertex displaced along the jet axis, so their
    `d0` distribution is visibly wider than light jets' (whose tracks
    genuinely originate at the primary vertex -- `d0 == 0` up to detector
    resolution alone)."""
    import matplotlib.pyplot as plt

    jetted = tracks[tracks["jet_id"] != -1]
    light, b_jet = jetted[~jetted["is_b_jet"]], jetted[jetted["is_b_jet"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    edges = np.histogram_bin_edges(jetted["d0"], bins=bins)
    ax.hist(
        light["d0"], bins=edges, color=LIGHT_COLOR, alpha=0.6, density=True, label=f"light jets (n={len(light)})"
    )
    ax.hist(b_jet["d0"], bins=edges, color=B_JET_COLOR, alpha=0.6, density=True, label=f"b-jets (n={len(b_jet)})")
    style_axes(
        ax,
        theme,
        spatial=False,
        title="Reconstructed track d0",
        xlabel="d0 (impact parameter)",
        ylabel="density",
        legend=True,
    )
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_cluster_energy(clusters: pd.DataFrame, bins: int = 60, theme: Theme | None = None, save_path=None):
    """Truth vs. smeared cluster energy, overlaid -- shows the resolution
    smearing `detectorreco2d.reconstruct_clusters` applies, independent of
    jet flavor."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    edges = np.histogram_bin_edges(clusters["energy_true"], bins=bins)
    ax.hist(clusters["energy_true"], bins=edges, color=palette.TRUTH, alpha=0.5, density=True, label="truth")
    ax.hist(clusters["energy"], bins=edges, color=palette.CLUSTER, alpha=0.5, density=True, label="reconstructed")
    style_axes(
        ax, theme, spatial=False, title="Calorimeter cluster energy", xlabel="energy", ylabel="density", legend=True
    )
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_track_multiplicity(tracks: pd.DataFrame, theme: Theme | None = None, save_path=None):
    """Number of reconstructed tracks per jet, light vs b-jet -- the
    `b_jet_track_boost` effect."""
    import matplotlib.pyplot as plt

    jets = summarize_jets(tracks)
    light, b_jet = jets[~jets["is_b_jet"]], jets[jets["is_b_jet"]]
    max_n = int(jets["n_tracks"].max()) if len(jets) else 0
    edges = np.arange(-0.5, max_n + 1.5, 1.0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        light["n_tracks"],
        bins=edges,
        color=LIGHT_COLOR,
        alpha=0.6,
        density=True,
        label=f"light jets (mean={light['n_tracks'].mean():.1f})",
    )
    ax.hist(
        b_jet["n_tracks"],
        bins=edges,
        color=B_JET_COLOR,
        alpha=0.6,
        density=True,
        label=f"b-jets (mean={b_jet['n_tracks'].mean():.1f})",
    )
    style_axes(
        ax, theme, spatial=False, title="Tracks per jet", xlabel="number of tracks", ylabel="fraction of jets",
        legend=True,
    )
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_muon_multiplicity(tracks: pd.DataFrame, theme: Theme | None = None, save_path=None):
    """Number of muons per jet, light vs b-jet -- the `jet_muon_fraction`/
    `b_jet_muon_fraction` effect (a semileptonic B decay makes a b-jet far
    more likely to contain a muon)."""
    import matplotlib.pyplot as plt

    jets = summarize_jets(tracks)
    light, b_jet = jets[~jets["is_b_jet"]], jets[jets["is_b_jet"]]
    counts = sorted(jets["n_muons"].unique()) if len(jets) else [0]
    x = np.arange(len(counts))
    width = 0.35

    def _fractions(subset: pd.DataFrame) -> list[float]:
        return [float((subset["n_muons"] == n).mean()) if len(subset) else 0.0 for n in counts]

    light_with_muon = (light["n_muons"] > 0).mean() if len(light) else 0.0
    b_jet_with_muon = (b_jet["n_muons"] > 0).mean() if len(b_jet) else 0.0

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, _fractions(light), width, color=LIGHT_COLOR, label=f"light jets ({light_with_muon:.1%} with a muon)")
    ax.bar(x + width / 2, _fractions(b_jet), width, color=B_JET_COLOR, label=f"b-jets ({b_jet_with_muon:.1%} with a muon)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in counts])
    style_axes(
        ax, theme, spatial=False, title="Muons per jet", xlabel="number of muons", ylabel="fraction of jets",
        legend=True,
    )
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def make_validation_plots(
    tracks: pd.DataFrame, clusters: pd.DataFrame, output_dir: str | Path, theme: Theme | None = None
) -> dict[str, Path]:
    """All four validation plots at once, written to `output_dir` -- the
    CLI's `validate` command."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "track_d0": output_dir / "track_d0.png",
        "cluster_energy": output_dir / "cluster_energy.png",
        "track_multiplicity": output_dir / "track_multiplicity.png",
        "muon_multiplicity": output_dir / "muon_multiplicity.png",
    }
    plot_track_d0(tracks, theme=theme, save_path=paths["track_d0"])
    plot_cluster_energy(clusters, theme=theme, save_path=paths["cluster_energy"])
    plot_track_multiplicity(tracks, theme=theme, save_path=paths["track_multiplicity"])
    plot_muon_multiplicity(tracks, theme=theme, save_path=paths["muon_multiplicity"])
    return paths
