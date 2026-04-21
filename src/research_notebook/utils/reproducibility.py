"""Reproducibility utilities for setting random seeds."""

from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Sets seeds for Python's random module, NumPy, and optionally PyTorch
    if it is installed.

    Args:
        seed: The random seed to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
