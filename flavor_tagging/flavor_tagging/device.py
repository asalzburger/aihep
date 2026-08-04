"""Device selection: prefer Apple Silicon's MPS backend when available,
falling back to CPU. This network is tiny, so either device trains it in
seconds -- MPS is just the better default when it's there."""

from __future__ import annotations

import torch


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
