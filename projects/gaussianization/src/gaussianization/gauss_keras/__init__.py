"""gauss_keras — pure-Keras Gaussianization flow building blocks.

The library is organised around the hybrid bijector pattern: each
``Bijector`` layer exposes two explicit methods,

    forward_and_log_det(x)  -> (z, log_det_jacobian)
    inverse_and_log_det(z)  -> (x, log_det_jacobian)

and a unified ``call(x, inverse=False)`` entry point that dispatches to
one of them and contributes ``-mean(ldj)`` to the layer loss via
``self.add_loss``. This keeps ``keras.Sequential``-style composition for
training while leaving per-sample log-densities accessible through
``GaussianizationFlow.log_prob``.

Sub-modules
-----------
- ``_math``               : standard-normal helpers (CDF, quantile, log pdf).
- ``mixtures.gaussian``   : mixture-of-Gaussians CDF / pdf / log-pdf.
- ``bijectors.base``      : ``Bijector`` abstract layer.
- ``bijectors.rotation``  : ``Householder`` and ``FixedOrtho`` layers.
- ``bijectors.marginal``  : ``MixtureCDFGaussianization`` (PR B).
- ``bijectors.flow``      : ``GaussianizationFlow`` model (PR B).
"""

from __future__ import annotations

from gaussianization.gauss_keras import bijectors, ig_init, mixtures, training
from gaussianization.gauss_keras.bijectors import (
    Bijector,
    FixedOrtho,
    GaussianizationFlow,
    Householder,
    MixtureCDFCoupling,
    MixtureCDFGaussianization,
    default_half_mask,
    make_mlp_conditioner,
    make_shared_mlp_conditioner,
    sigmoid_log_scale_clamp,
    tanh_log_scale_clamp,
)
from gaussianization.gauss_keras.mixtures import MixtureOfGaussians
from gaussianization.gauss_keras.ig_init import initialize_flow_from_ig
from gaussianization.gauss_keras.training import (
    base_nll_loss,
    make_coupling_flow,
    make_gaussianization_flow,
)

__all__ = [
    "Bijector",
    "FixedOrtho",
    "GaussianizationFlow",
    "Householder",
    "MixtureCDFCoupling",
    "MixtureCDFGaussianization",
    "MixtureOfGaussians",
    "base_nll_loss",
    "bijectors",
    "default_half_mask",
    "ig_init",
    "initialize_flow_from_ig",
    "make_coupling_flow",
    "make_gaussianization_flow",
    "make_mlp_conditioner",
    "make_shared_mlp_conditioner",
    "mixtures",
    "sigmoid_log_scale_clamp",
    "tanh_log_scale_clamp",
    "training",
]
