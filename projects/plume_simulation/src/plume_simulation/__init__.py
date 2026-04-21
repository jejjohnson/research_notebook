"""plume_simulation — forward models for atmospheric plume dispersion.

Sub-packages
------------
- ``gauss_plume``  : steady-state Gaussian plume (JAX + NumPyro).
- ``gauss_puff``   : time-resolved Gaussian puff (JAX + diffrax + NumPyro).
- ``les_fvm``      : Eulerian 3-D advection-diffusion on an Arakawa C-grid
                     (JAX + diffrax + finitevolX) for spatially-varying
                     wind fields and K-theory eddy diffusivity.

Additional dispersion models (resolved-flow LES, etc.) may be added as
sibling sub-packages in future ports.
"""

from __future__ import annotations

from plume_simulation import gauss_plume, gauss_puff, les_fvm

__all__ = ["gauss_plume", "gauss_puff", "les_fvm"]
