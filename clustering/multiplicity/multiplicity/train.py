"""Train the multiplicity MLP on one or more sensor-shaped run directories,
with a stratified train/validation split for monitoring during training.
Final validation (ROC curves, confusion matrix) against a genuinely
independent dataset belongs in `evaluate.py`, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .dataset import Dataset, build_dataset
from .device import select_device
from .io import Format
from .model import MultiplicityMLP, decode_score, encode_label


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)


def _tensors(matrices: np.ndarray, n_particles: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(matrices).to(device)
    y = torch.from_numpy(encode_label(n_particles)).to(device)
    return x, y


def train_model(
    input_dirs: list[str | Path],
    fmt: Format = "arrow",
    epochs: int = 50,
    batch_size: int = 256,
    val_fraction: float = 0.15,
    lr: float = 1e-3,
    device: str = "auto",
    seed: int = 0,
) -> tuple[MultiplicityMLP, tuple[int, int], TrainHistory]:
    """Returns (model, matrix_shape, history). `matrix_shape` must be
    passed to `evaluate.evaluate_model`/`dataset.build_dataset` for any
    later dataset so it's embedded into the exact same fixed input size."""
    dataset = build_dataset(input_dirs, fmt=fmt)
    n_x, n_y = dataset.matrix_shape

    train_idx, val_idx = train_test_split(
        np.arange(len(dataset.n_particles)),
        test_size=val_fraction,
        random_state=seed,
        stratify=dataset.n_particles,
    )

    torch_device = select_device(device)
    model = MultiplicityMLP(n_x, n_y).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    x_train, y_train = _tensors(dataset.matrices[train_idx], dataset.n_particles[train_idx], torch_device)
    x_val, y_val = _tensors(dataset.matrices[val_idx], dataset.n_particles[val_idx], torch_device)
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
            val_classes = decode_score(val_pred.cpu().numpy())
            val_accuracy = float((val_classes == dataset.n_particles[val_idx]).mean())

        history.train_loss.append(float(np.mean(batch_losses)))
        history.val_loss.append(val_loss)
        history.val_accuracy.append(val_accuracy)
        print(
            f"epoch {epoch + 1:>3}/{epochs}  "
            f"train_loss={history.train_loss[-1]:.4f}  val_loss={val_loss:.4f}  val_acc={val_accuracy:.3f}"
        )

    return model, dataset.matrix_shape, history


def save_checkpoint(path: str | Path, model: MultiplicityMLP, matrix_shape: tuple[int, int]) -> None:
    torch.save({"state_dict": model.state_dict(), "matrix_shape": matrix_shape}, path)


def load_checkpoint(path: str | Path, device: str = "auto") -> tuple[MultiplicityMLP, tuple[int, int]]:
    torch_device = select_device(device)
    checkpoint = torch.load(path, map_location=torch_device, weights_only=True)
    n_x, n_y = checkpoint["matrix_shape"]
    model = MultiplicityMLP(n_x, n_y).to(torch_device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, (n_x, n_y)
