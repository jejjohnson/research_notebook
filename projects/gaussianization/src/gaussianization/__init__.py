"""gaussianization — Gaussianization flows for density estimation.

Sub-packages
------------
- ``gauss_keras`` : pure-Keras 3 implementation (TF / JAX / torch backend
                    chosen by ``KERAS_BACKEND``). Provides rotation and
                    marginal-Gaussianization bijectors plus a flow model.
- ``fair`` : fairness penalties built on top of frozen Gaussianization
              flows. Designed to drop into ``fairkl.models.FairModelWrapper``.

Additional backends (``gauss_jax``, ``gauss_flax``, …) may be added as
sibling sub-packages.
"""

from __future__ import annotations

from gaussianization import fair, gauss_keras

__all__ = ["fair", "gauss_keras"]
