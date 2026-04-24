"""plume_simulation — forward models for atmospheric plume dispersion.

Sub-packages
------------
- ``gauss_plume``  : steady-state Gaussian plume (JAX + NumPyro).
- ``gauss_puff``   : time-resolved Gaussian puff (JAX + diffrax + NumPyro),
                     with optional Ornstein-Uhlenbeck sub-grid turbulence.
- ``les_fvm``      : Eulerian 3-D advection-diffusion on an Arakawa C-grid
                     (JAX + diffrax + finitevolX) for spatially-varying
                     wind fields and K-theory eddy diffusivity.
- ``hapi_lut``     : HITRAN line-by-line absorption cross-section LUTs
                     (HAPI + xarray) plus a Beer-Lambert forward model and
                     the differential-ratio form for plume-enhancement
                     retrievals. ``hitran-api`` is imported lazily.
- ``radtran``      : Band-integrated Beer-Lambert forward model, normalised
                     -brightness LUT, and matched-filter retrieval
                     (multispectral / hyperspectral).

Additional dispersion models (resolved-flow LES, etc.) may be added as
sibling sub-packages in future ports.
"""

from __future__ import annotations

from plume_simulation import gauss_plume, gauss_puff, hapi_lut, les_fvm, radtran

__all__ = ["gauss_plume", "gauss_puff", "hapi_lut", "les_fvm", "radtran"]
