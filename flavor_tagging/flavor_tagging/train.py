"""Train the b-tagger MLP on one reconstructed run's `tracks`/`clusters`,
with a stratified train/validation split for monitoring during training.
Final validation (ROC, confusion matrix, score separation) against a
genuinely independent simulated-and-reconstructed run belongs in
`evaluate.py`, not here -- see the module docstring there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .dataset import build_dataset, compute_standardization, standardize
from .device import select_device
from .model import BTaggerMLP


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)


def _tensors(features: np.ndarray, is_b_jet: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(features).to(device)
    y = torch.from_numpy(is_b_jet.astype(np.float32)).to(device)
    return x, y


def train_model(
    tracks: pd.DataFrame,
    clusters: pd.DataFrame,
    epochs: int = 50,
    batch_size: int = 64,
    val_fraction: float = 0.15,
    lr: float = 1e-3,
    device: str = "auto",
    seed: int = 0,
) -> tuple[BTaggerMLP, dict, TrainHistory]:
    """Returns `(model, preprocessing, history)`. `preprocessing`
    (`n_track_slots`, `mean`, `std`, `feature_names`) must be passed to
    `evaluate.evaluate_model`/`dataset.build_dataset` for any later dataset
    so it's embedded and scaled identically -- see `save_checkpoint`, which
    bundles it with the model for exactly that reason.
    """
    dataset = build_dataset(tracks, clusters)
    mean, std = compute_standardization(dataset.features)
    features = standardize(dataset.features, mean, std)

    train_idx, val_idx = train_test_split(
        np.arange(len(dataset.is_b_jet)), test_size=val_fraction, random_state=seed, stratify=dataset.is_b_jet
    )

    torch_device = select_device(device)
    model = BTaggerMLP(features.shape[1]).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    x_train, y_train = _tensors(features[train_idx], dataset.is_b_jet[train_idx], torch_device)
    x_val, y_val = _tensors(features[val_idx], dataset.is_b_jet[val_idx], torch_device)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

    history = TrainHistory()
    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_loss = loss_fn(val_pred, y_val).item()
            val_predicted = (val_pred.cpu().numpy() >= 0.5).astype(int)
            val_accuracy = float((val_predicted == dataset.is_b_jet[val_idx]).mean())

        history.train_loss.append(float(np.mean(batch_losses)))
        history.val_loss.append(val_loss)
        history.val_accuracy.append(val_accuracy)
        print(
            f"epoch {epoch + 1:>3}/{epochs}  train_loss={history.train_loss[-1]:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_accuracy:.3f}"
        )

    preprocessing = dict(
        n_track_slots=dataset.n_track_slots, mean=mean, std=std, feature_names=dataset.feature_names
    )
    return model, preprocessing, history


def save_checkpoint(path: str | Path, model: BTaggerMLP, preprocessing: dict) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_features": model.n_features,
            "n_track_slots": preprocessing["n_track_slots"],
            "mean": preprocessing["mean"].tolist(),
            "std": preprocessing["std"].tolist(),
            "feature_names": preprocessing["feature_names"],
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str = "auto") -> tuple[BTaggerMLP, dict]:
    torch_device = select_device(device)
    checkpoint = torch.load(path, map_location=torch_device, weights_only=True)

    model = BTaggerMLP(checkpoint["n_features"]).to(torch_device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    preprocessing = dict(
        n_track_slots=checkpoint["n_track_slots"],
        mean=np.array(checkpoint["mean"], dtype=np.float32),
        std=np.array(checkpoint["std"], dtype=np.float32),
        feature_names=checkpoint["feature_names"],
    )
    return model, preprocessing
