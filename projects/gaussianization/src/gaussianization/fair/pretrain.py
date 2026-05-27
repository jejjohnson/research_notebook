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
