"""Training helpers: flow factories and a base-distribution NLL loss."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from keras import ops

from gaussianization.gauss_keras._math import norm_log_pdf
from gaussianization.gauss_keras.bijectors.coupling import (
    MixtureCDFCoupling,
    default_half_mask,
    make_mlp_conditioner,
    make_shared_mlp_conditioner,
)
from gaussianization.gauss_keras.bijectors.flow import GaussianizationFlow
from gaussianization.gauss_keras.bijectors.marginal import MixtureCDFGaussianization
from gaussianization.gauss_keras.bijectors.rotation import FixedOrtho, Householder


def base_nll_loss(_y_true: Any, y_pred: Any) -> Any:
    """Negative log-likelihood under a ``N(0, I)`` base.

    The per-layer log-det contributions flow in via ``layer.add_loss``;
    this function only reports ``-E[log p_Z(z)]`` for the final latent.
    Accepts and returns whatever tensor type the active Keras backend
    produces (TF / JAX / torch).
    """
    return -ops.mean(ops.sum(norm_log_pdf(y_pred), axis=-1))


def make_gaussianization_flow(
    input_dim: int,
    num_blocks: int = 6,
    num_reflectors: int | None = None,
    num_components: int = 8,
    pca_init_data: np.ndarray | None = None,
    mixture_init_data: np.ndarray | None = None,
) -> GaussianizationFlow:
    """Build a ``FixedOrtho? → (Householder → MixtureCDF)×N`` flow.

    Args:
        input_dim: data dimensionality ``d``.
        num_blocks: number of ``(rotation, marginal)`` blocks.
        num_reflectors: Householder vectors per rotation. Defaults to
            ``d`` (full-rank coverage of ``O(d)``).
        num_components: mixture components per marginal layer.
        pca_init_data: if given, prepend a :class:`FixedOrtho` layer
            initialised from the eigenvectors of ``cov(pca_init_data)``.
        mixture_init_data: if given, initialise each mixture layer's
            means from the data quantiles via
            :meth:`MixtureCDFGaussianization.adapt_means_from_quantiles`.
    """
    if num_reflectors is None:
        num_reflectors = input_dim

    bijectors = []
    if pca_init_data is not None:
        bijectors.append(FixedOrtho.from_pca(pca_init_data))
    for _ in range(num_blocks):
        bijectors.append(Householder(num_reflectors=num_reflectors))
        marginal = MixtureCDFGaussianization(num_components=num_components)
        if mixture_init_data is not None:
            marginal.adapt_means_from_quantiles(mixture_init_data)
        bijectors.append(marginal)

    return GaussianizationFlow(bijectors, input_dim=input_dim)


def make_coupling_flow(
    input_dim: int,
    num_blocks: int = 4,
    num_components: int = 8,
    hidden: Sequence[int] = (64, 64),
    activation: str = "relu",
    mask: np.ndarray | Sequence[bool] | None = None,
    shared_mixture: bool = False,
    include_rotation: bool = True,
    num_reflectors: int | None = None,
) -> GaussianizationFlow:
    """Build a coupling flow with alternating-mask ``MixtureCDFCoupling`` pairs.

    Each block is ``[Householder?, MixtureCDFCoupling(m), MixtureCDFCoupling(~m)]``
    so that both halves of the mask are transformed within the same
    block. Rotations sit between blocks to mix information across all
    dims; they are optional for ``d == 2`` but strongly recommended for
    higher dims.

    Args:
        input_dim: data dimensionality ``d``.
        num_blocks: number of mask-pair blocks.
        num_components: mixture components ``K`` per b-dim.
        hidden: hidden widths of each conditioner MLP.
        activation: activation used inside conditioner MLPs.
        mask: length-``d`` bool array. Defaults to the first ``d // 2``
            dims being the identity half.
        shared_mixture: if True, the conditioner outputs a single
            ``(3 · K)`` block broadcast to all b-dims; else it outputs
            a per-dim ``(3 · d_b · K)`` block.
        include_rotation: if True, prepend a :class:`Householder` to
            each block.
        num_reflectors: Householder reflector count; defaults to
            ``input_dim``.
    """
    if mask is None:
        mask = default_half_mask(input_dim)
    mask = np.asarray(mask, dtype=bool)
    if mask.size != input_dim:
        raise ValueError(f"mask length {mask.size} != input_dim {input_dim}")
    mask_alt = ~mask
    d_a = int(mask.sum())
    d_b = input_dim - d_a
    if num_reflectors is None:
        num_reflectors = input_dim

    def _cond(d_out):
        factory = (
            make_shared_mlp_conditioner if shared_mixture else make_mlp_conditioner
        )
        return factory(
            d_b=d_out, num_components=num_components, hidden=hidden, activation=activation
        )

    bijectors = []
    for _ in range(num_blocks):
        if include_rotation:
            bijectors.append(Householder(num_reflectors=num_reflectors))
        bijectors.append(
            MixtureCDFCoupling(
                mask=mask,
                conditioner=_cond(d_b),
                num_components=num_components,
            )
        )
        bijectors.append(
            MixtureCDFCoupling(
                mask=mask_alt,
                conditioner=_cond(d_a),
                num_components=num_components,
            )
        )
    return GaussianizationFlow(bijectors, input_dim=input_dim)
