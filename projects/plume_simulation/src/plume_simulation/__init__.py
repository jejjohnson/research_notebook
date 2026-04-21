"""plume_simulation — forward models for atmospheric plume dispersion.

Sub-packages
------------
- ``gauss_plume``    : steady-state Gaussian plume (JAX + NumPyro)

Additional dispersion models (puff, LES-based, etc.) may be added as
sibling sub-packages in future ports.
"""

from __future__ import annotations

from plume_simulation import gauss_plume

__all__ = ["gauss_plume"]
