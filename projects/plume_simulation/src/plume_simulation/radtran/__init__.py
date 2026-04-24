"""Band-integrated radiative transfer + matched-filter retrieval.

Bridges the line-by-line :mod:`plume_simulation.hapi_lut` absorption-LUT
family to the multispectral / hyperspectral retrieval world by providing:

- :mod:`.config`         — :class:`ObservationGeometry`, :class:`InstrumentSpec`.
- :mod:`.srf`            — :class:`SpectralResponseFunction` (gaussian /
  rectangular / triangular / custom) for band integration.
- :mod:`.forward`        — exact / Maclaurin / Taylor Beer-Lambert forward
  models and their normalised (differential) variants, each returning
  ``(radiance, Jacobian, transmittance)``.
- :mod:`.nb_lut`         — pre-tabulated normalised-brightness LUT
  ``nB_b(ΔX)`` plus :func:`inject_plume` for scene-injection.
- :mod:`.target`         — matched-filter target-spectrum constructors
  (nonlinear + linearised).
- :mod:`.background`     — robust trimmed-mean background and low-rank
  covariance estimators.
- :mod:`.matched_filter` — the core ``ε̂ = (x−μ)ᵀΣ⁻¹t / (tᵀΣ⁻¹t)``
  at pixel and image scope, plus an SNR helper.
- :mod:`.gaussx_solve` — structured-operator matched filter using
  :mod:`gaussx`'s ``LowRankUpdate`` + Woodbury-backed ``solve``, for
  hyperspectral retrievals where the ``O(n_bands³)`` dense inverse in
  :mod:`.matched_filter` becomes the bottleneck.

The input LUT is expected to be the :class:`xarray.Dataset` produced by
:func:`plume_simulation.hapi_lut.build_lut_dataset`.

Notes
-----
Heavy imports (scipy / sklearn / gaussx) are deferred so that
``import plume_simulation.radtran`` stays cheap — :mod:`.background` pulls
in scipy and :mod:`.gaussx_solve` pulls in gaussx + jax, both only at first
use of the relevant functions.
"""

from __future__ import annotations

# background + matched_filter pull in scipy / sklearn; import eagerly since
# they are small and the cost is paid once at first `radtran` import.
from plume_simulation.radtran.background import (
    robust_lowrank_covariance,
    trimmed_mean_spectrum,
)
from plume_simulation.radtran.config import (
    ATM_TO_PA,
    BOLTZMANN_J_PER_K,
    CM3_PER_M3,
    InstrumentSpec,
    ObservationGeometry,
    number_density_cm3,
)
from plume_simulation.radtran.forward import (
    ForwardResult,
    forward_maclaurin,
    forward_maclaurin_normalized,
    forward_nonlinear,
    forward_nonlinear_normalized,
    forward_taylor,
    forward_taylor_normalized,
)
from plume_simulation.radtran.gaussx_solve import (
    build_lowrank_covariance_operator,
    matched_filter_image_op,
    matched_filter_pixel_op,
    matched_filter_snr_op,
)
from plume_simulation.radtran.matched_filter import (
    matched_filter_image,
    matched_filter_pixel,
    matched_filter_snr,
)
from plume_simulation.radtran.nb_lut import (
    NBLookup,
    build_nb_lut,
    inject_plume,
    lookup_nb,
)
from plume_simulation.radtran.srf import SpectralResponseFunction
from plume_simulation.radtran.target import (
    target_bands,
    target_spectrum_normalized_linear,
    target_spectrum_normalized_nonlinear,
)


__all__ = [
    # config
    "ATM_TO_PA",
    "BOLTZMANN_J_PER_K",
    "CM3_PER_M3",
    "InstrumentSpec",
    "ObservationGeometry",
    "number_density_cm3",
    # forward models
    "ForwardResult",
    "forward_nonlinear",
    "forward_nonlinear_normalized",
    "forward_maclaurin",
    "forward_maclaurin_normalized",
    "forward_taylor",
    "forward_taylor_normalized",
    # SRF
    "SpectralResponseFunction",
    # nB-LUT / injection
    "NBLookup",
    "build_nb_lut",
    "lookup_nb",
    "inject_plume",
    # matched-filter pipeline
    "target_spectrum_normalized_nonlinear",
    "target_spectrum_normalized_linear",
    "target_bands",
    "trimmed_mean_spectrum",
    "robust_lowrank_covariance",
    "matched_filter_pixel",
    "matched_filter_image",
    "matched_filter_snr",
    # gaussx operator-backed matched filter
    "build_lowrank_covariance_operator",
    "matched_filter_pixel_op",
    "matched_filter_image_op",
    "matched_filter_snr_op",
]
