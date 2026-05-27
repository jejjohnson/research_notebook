"""Shared plot styling for the fair-Gaussianization notebooks.

Mirrors the look of ``keras-fairkl/docs/notebooks/_style.py`` so the
two notebook collections sit next to each other without visual jarring.
"""

from __future__ import annotations


def style_ax(ax) -> None:
    """Apply a light grid + minor ticks to a matplotlib axis."""
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)
    ax.minorticks_on()


SCATTER_KW = dict(s=22, edgecolors="k", linewidths=0.4, alpha=0.55, zorder=5)
GROUP_COLORS = {
    "male": "tab:blue",
    "female": "tab:orange",
    0: "tab:blue",
    1: "tab:orange",
}
G_COLOR = "tab:purple"  # G-XCOV everywhere
CKA_COLOR = "tab:red"  # CKA baseline everywhere
