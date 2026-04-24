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
  covariance estimators (depends on :mod:`scipy.stats`).
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
``radtran`` is an aggregator: it re-exports symbols from its submodules but
imports them *lazily* via :pep:`562`. This means

- ``import plume_simulation`` (which itself imports ``radtran``) does not
  pull in :mod:`scipy` or :mod:`gaussx` until you actually touch
  :mod:`.background` / :mod:`.gaussx_solve`.
- ``from plume_simulation.radtran import build_nb_lut`` triggers exactly
  one submodule import (``radtran.nb_lut``) and stops there.
- The ``__all__`` list still drives star-imports and IDE auto-complete,
  so the lazy loading is invisible to consumers.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING


# Map exported name → (submodule, attr-name). Used by ``__getattr__`` to
# route lookups to the right submodule on first access.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # config
    "ATM_TO_PA": ("config", "ATM_TO_PA"),
    "BOLTZMANN_J_PER_K": ("config", "BOLTZMANN_J_PER_K"),
    "CM3_PER_M3": ("config", "CM3_PER_M3"),
    "InstrumentSpec": ("config", "InstrumentSpec"),
    "ObservationGeometry": ("config", "ObservationGeometry"),
    "number_density_cm3": ("config", "number_density_cm3"),
    # forward
    "ForwardResult": ("forward", "ForwardResult"),
    "forward_nonlinear": ("forward", "forward_nonlinear"),
    "forward_nonlinear_normalized": ("forward", "forward_nonlinear_normalized"),
    "forward_maclaurin": ("forward", "forward_maclaurin"),
    "forward_maclaurin_normalized": ("forward", "forward_maclaurin_normalized"),
    "forward_taylor": ("forward", "forward_taylor"),
    "forward_taylor_normalized": ("forward", "forward_taylor_normalized"),
    # SRF
    "SpectralResponseFunction": ("srf", "SpectralResponseFunction"),
    # nB-LUT / injection
    "NBLookup": ("nb_lut", "NBLookup"),
    "build_nb_lut": ("nb_lut", "build_nb_lut"),
    "lookup_nb": ("nb_lut", "lookup_nb"),
    "inject_plume": ("nb_lut", "inject_plume"),
    # target-spectrum constructors
    "target_spectrum_normalized_nonlinear": ("target", "target_spectrum_normalized_nonlinear"),
    "target_spectrum_normalized_linear": ("target", "target_spectrum_normalized_linear"),
    "target_bands": ("target", "target_bands"),
    # background — pulls scipy on first use
    "trimmed_mean_spectrum": ("background", "trimmed_mean_spectrum"),
    "robust_lowrank_covariance": ("background", "robust_lowrank_covariance"),
    # matched filter (numpy)
    "matched_filter_pixel": ("matched_filter", "matched_filter_pixel"),
    "matched_filter_image": ("matched_filter", "matched_filter_image"),
    "matched_filter_snr": ("matched_filter", "matched_filter_snr"),
    # gaussx-backed matched filter — pulls gaussx + jax on first use
    "build_lowrank_covariance_operator": ("gaussx_solve", "build_lowrank_covariance_operator"),
    "matched_filter_pixel_op": ("gaussx_solve", "matched_filter_pixel_op"),
    "matched_filter_image_op": ("gaussx_solve", "matched_filter_image_op"),
    "matched_filter_snr_op": ("gaussx_solve", "matched_filter_snr_op"),
}


def __getattr__(name: str):  # PEP 562 — lazy module attribute access
    try:
        submodule, attr = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module 'plume_simulation.radtran' has no attribute {name!r}"
        ) from None
    module = importlib.import_module(f"plume_simulation.radtran.{submodule}")
    value = getattr(module, attr)
    globals()[name] = value  # cache on first access so subsequent lookups are O(1)
    return value


def __dir__() -> list[str]:  # IDE / dir() introspection sees the full surface
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


if TYPE_CHECKING:  # static type checkers: see the symbols as if eagerly imported
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


__all__ = sorted(_LAZY_EXPORTS)
