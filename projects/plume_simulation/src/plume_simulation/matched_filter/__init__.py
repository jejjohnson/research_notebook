"""Matched filter for hyperspectral methane retrieval.

The matched filter (MF) is the maximum-SNR linear detector for a known target
signature ``t(ν)`` in additive Gaussian noise with covariance ``Σ``. It is
also the **one-CG-iteration limit** of the dual / PSAS 3D-Var solver with a
flat (identity) prior — see ``notebooks/matched_filter/00_mf_derivation.md``.

Score formula (per pixel)::

    y(x) = (x − μ)ᵀ Σ⁻¹ t / (tᵀ Σ⁻¹ t)

where ``x`` is the observed spectrum, ``μ`` is the background mean, and ``t``
is the target signature — typically the radiance Jacobian ``∂L/∂VMR`` from
the Beer–Lambert forward model at the background state.

Architecture
------------
- **Estimation** (``background``, ``cluster``, ``streaming``) — statistical
  estimation uses scikit-learn (``EmpiricalCovariance``, ``LedoitWolf``,
  ``TruncatedSVD``, ``GaussianMixture``). Not differentiable — that's fine,
  MF doesn't need it.
- **Application** (``core``) — the per-pixel score evaluation runs in JAX
  and dispatches on gaussx operator structure. Covariance estimators return
  a :class:`lineax.AbstractLinearOperator` (dense, low-rank + Tikhonov via
  :class:`gaussx.LowRankUpdate`, etc.) so the kernel is single-implementation.
- **Target signatures** (``target``) — derived from an existing
  :class:`plume_simulation.assimilation.RadianceObservationModel` via
  :func:`jax.jvp` (directional derivatives) or a finite-amplitude secant.
  No duplicate Beer–Lambert code.
- **I/O** (``io``) — xarray wrappers for multi-scene workflows.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # core
    "apply_pixel": ("core", "apply_pixel"),
    "apply_image": ("core", "apply_image"),
    "matched_filter_snr": ("core", "matched_filter_snr"),
    "detection_threshold": ("core", "detection_threshold"),
    "validate_mf_inputs": ("core", "validate_mf_inputs"),
    # target
    "linear_target_from_obs": ("target", "linear_target_from_obs"),
    "nonlinear_target_from_obs": ("target", "nonlinear_target_from_obs"),
    # background
    "estimate_mean": ("background", "estimate_mean"),
    "estimate_cov_empirical": ("background", "estimate_cov_empirical"),
    "estimate_cov_shrunk": ("background", "estimate_cov_shrunk"),
    "estimate_cov_lowrank": ("background", "estimate_cov_lowrank"),
    # cluster
    "gmm_cluster_background": ("cluster", "gmm_cluster_background"),
    "adaptive_window_background": ("cluster", "adaptive_window_background"),
    # streaming
    "WelfordAccumulator": ("streaming", "WelfordAccumulator"),
    "streaming_background": ("streaming", "streaming_background"),
    # io
    "apply_image_xarray": ("io", "apply_image_xarray"),
    "open_multi_scene": ("io", "open_multi_scene"),
}


def __getattr__(name: str):  # PEP 562 — lazy module attribute access
    try:
        submodule, attr = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module 'plume_simulation.matched_filter' has no attribute {name!r}"
        ) from None
    module = importlib.import_module(f"plume_simulation.matched_filter.{submodule}")
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


if TYPE_CHECKING:  # static type checkers see eagerly-imported symbols
    from plume_simulation.matched_filter.background import (
        estimate_cov_empirical,
        estimate_cov_lowrank,
        estimate_cov_shrunk,
        estimate_mean,
    )
    from plume_simulation.matched_filter.cluster import (
        adaptive_window_background,
        gmm_cluster_background,
    )
    from plume_simulation.matched_filter.core import (
        apply_image,
        apply_pixel,
        detection_threshold,
        matched_filter_snr,
        validate_mf_inputs,
    )
    from plume_simulation.matched_filter.io import (
        apply_image_xarray,
        open_multi_scene,
    )
    from plume_simulation.matched_filter.streaming import (
        WelfordAccumulator,
        streaming_background,
    )
    from plume_simulation.matched_filter.target import (
        linear_target_from_obs,
        nonlinear_target_from_obs,
    )


__all__ = sorted(_LAZY_EXPORTS)
