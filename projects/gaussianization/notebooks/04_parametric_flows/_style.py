"""Shared plot styling for the Gaussianization *foundations* notebooks.

Kept deliberately tiny so each foundations notebook can ``from _style import
style_ax, GAUSS_KW`` and get a consistent look without re-deriving it. Mirrors
the spirit of ``notebooks/fair_gauss/_style.py`` so the two collections sit
next to each other without visual jarring.
"""

from __future__ import annotations

import numpy as np

# Standard-normal reference curve colour, used wherever we overlay phi(z).
REF_COLOR = "k"
DATA_COLOR = "tab:blue"
LATENT_COLOR = "tab:orange"

SCATTER_KW = dict(s=10, alpha=0.4, edgecolors="none")
GAUSS_KW = dict(color=REF_COLOR, lw=1.5, ls="--")


def style_ax(ax) -> None:
    """Apply a light grid + minor ticks to a matplotlib axis."""
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)
    ax.minorticks_on()


def standard_normal_pdf(z):
    """phi(z) = (2 pi)^{-1/2} exp(-z^2 / 2), for overlay curves."""
    z = np.asarray(z)
    return np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
