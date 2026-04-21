"""plume_simulation — forward models for atmospheric plume dispersion.

Sub-packages
------------
- ``gauss_plume``    : steady-state Gaussian plume (JAX + NumPyro)
- ``gauss_puff``     : time-resolved Gaussian puff (JAX + diffrax + NumPyro)

Additional dispersion models (LES-based, etc.) may be added as sibling
sub-packages in future ports.
"""

from __future__ import annotations

from plume_simulation import gauss_plume, gauss_puff

__all__ = ["gauss_plume", "gauss_puff"]
