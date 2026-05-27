"""Freeze a Keras model so it acts as a fixed differentiable function."""

from __future__ import annotations

import keras


def freeze_flow(model: keras.Model) -> keras.Model:
    """Mark every weight of ``model`` (and its sub-layers) as non-trainable.

    Setting ``model.trainable = False`` is necessary but not sufficient
    for nested sub-models under some Keras versions, so this also walks
    the layer tree and flips ``layer.trainable``.

    Args:
        model: The Keras model to freeze (typically a pretrained
            :class:`~gaussianization.gauss_keras.bijectors.flow.GaussianizationFlow`).

    Returns:
        The same model object, modified in place.
    """
    model.trainable = False
    for layer in _iter_layers(model):
        layer.trainable = False
    return model


def is_fully_frozen(model: keras.Model) -> bool:
    """Return ``True`` if ``model`` has no trainable weights."""
    return len(model.trainable_weights) == 0


def _iter_layers(model: keras.Model):
    """Depth-first walk of every Keras layer below ``model``."""
    stack = list(getattr(model, "_layers", []) or model.layers)
    while stack:
        layer = stack.pop()
        yield layer
        children = getattr(layer, "_layers", None) or getattr(layer, "layers", None)
        if children:
            stack.extend(children)
