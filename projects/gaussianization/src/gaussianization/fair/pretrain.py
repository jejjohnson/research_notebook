"""Pretrain a Gaussianization flow on a 1-D / k-D dataset and freeze it."""

from __future__ import annotations

import keras
import numpy as np

from gaussianization.fair.freeze import freeze_flow
from gaussianization.gauss_keras.bijectors.flow import GaussianizationFlow
from gaussianization.gauss_keras.training import (
    base_nll_loss,
    make_gaussianization_flow,
)


def fit_and_freeze(
    data: np.ndarray,
    num_blocks: int = 6,
    num_components: int = 8,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    validation_split: float = 0.1,
    patience: int = 20,
    seed: int | None = None,
    pca_init: bool = True,
    quantile_init: bool = True,
    verbose: int = 0,
) -> tuple[GaussianizationFlow, keras.callbacks.History]:
    """Train a flow by maximum likelihood and freeze its parameters.

    Args:
        data: Array of shape ``(n,)`` or ``(n, d)``.
        num_blocks: Number of ``(Householder, MixtureCDFGaussianization)``
            blocks.
        num_components: Mixture components per marginal layer.
        epochs: Maximum training epochs.
        batch_size: SGD batch size.
        lr: Adam learning rate.
        validation_split: Fraction of ``data`` reserved for early stopping.
        patience: Early-stopping patience (epochs of no val-loss improvement).
        seed: If given, sets ``keras.utils.set_random_seed`` before
            constructing the flow for reproducible initialisation.
        pca_init: Initialise a prepended ``FixedOrtho`` from the PCA of
            ``data``.
        quantile_init: Initialise each mixture-CDF layer's means from the
            data quantiles.
        verbose: Verbosity passed to ``keras.Model.fit``.

    Returns:
        A tuple ``(flow, history)`` where ``flow`` is frozen
        (no ``trainable_weights``) and ``history`` is the
        :class:`keras.callbacks.History` returned by ``fit``.
    """
    arr = np.asarray(data, dtype="float32")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    d = arr.shape[1]

    if seed is not None:
        keras.utils.set_random_seed(int(seed))

    flow = make_gaussianization_flow(
        input_dim=d,
        num_blocks=num_blocks,
        num_components=num_components,
        pca_init_data=arr if pca_init else None,
        mixture_init_data=arr if quantile_init else None,
    )
    flow.compile(optimizer=keras.optimizers.Adam(lr), loss=base_nll_loss)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )
    ]
    history = flow.fit(
        arr,
        arr,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=True,
    )
    freeze_flow(flow)
    return flow, history


def fit_and_freeze_joint(
    z: np.ndarray,
    q: np.ndarray,
    num_blocks: int = 6,
    num_components: int = 8,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    validation_split: float = 0.1,
    patience: int = 20,
    seed: int | None = None,
    n_shuffles: int = 1,
    verbose: int = 0,
) -> tuple[GaussianizationFlow, keras.callbacks.History]:
    """Pretrain a joint flow over the *product distribution* of (z, q).

    Constructs the empirical product distribution by **random-permuting**
    ``q`` so that the resulting ``(z, q_pi)`` pairs are independent by
    construction, then fits a single joint Gaussianization flow over the
    concatenated ``(z, q)`` space. The flow's job is to Gaussianise
    independent draws to :math:`\\mathcal{N}(0, I)`. Used by
    :class:`gaussianization.fair.losses.GaussianizedTotalCorrelationLoss`.

    Args:
        z: Baseline predictor-output samples, shape ``(n,)`` or ``(n, d_z)``.
        q: Sensitive-attribute samples, shape ``(n,)`` or ``(n, d_q)``.
        num_blocks: Number of ``(Householder, MixtureCDFGaussianization)``
            blocks. The joint space is small but the dependence in raw
            ``(z, q)`` can be complex; default ``6``.
        num_components: Mixture components per marginal layer.
        epochs: Maximum training epochs.
        batch_size: SGD batch size.
        lr: Adam learning rate.
        validation_split: Fraction reserved for early stopping.
        patience: Early-stopping patience.
        seed: If given, sets ``keras.utils.set_random_seed`` before
            constructing the flow and seeds the permutation RNG.
        n_shuffles: Number of independent permutations of ``q`` to
            concatenate into the training set (effective dataset size
            multiplier). Default ``1``.
        verbose: Verbosity passed to ``keras.Model.fit``.

    Returns:
        ``(flow, history)`` where ``flow`` is frozen and trained on the
        empirical product distribution of ``(z, q)``.
    """
    z_arr = np.asarray(z, dtype="float32")
    q_arr = np.asarray(q, dtype="float32")
    if z_arr.ndim == 1:
        z_arr = z_arr.reshape(-1, 1)
    if q_arr.ndim == 1:
        q_arr = q_arr.reshape(-1, 1)
    if z_arr.shape[0] != q_arr.shape[0]:
        raise ValueError(
            f"z and q must have the same length; got "
            f"{z_arr.shape[0]} vs {q_arr.shape[0]}"
        )

    rng = np.random.default_rng(seed)
    blocks = []
    for _ in range(int(n_shuffles)):
        perm = rng.permutation(z_arr.shape[0])
        blocks.append(np.concatenate([z_arr, q_arr[perm]], axis=1))
    joint_data = np.concatenate(blocks, axis=0).astype("float32")

    return fit_and_freeze(
        joint_data,
        num_blocks=num_blocks,
        num_components=num_components,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        validation_split=validation_split,
        patience=patience,
        seed=seed,
        pca_init=True,
        quantile_init=True,
        verbose=verbose,
    )
