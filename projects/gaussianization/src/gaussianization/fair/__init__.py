"""Fair learning with Gaussianization flows.

A small library that composes:

* :mod:`gaussianization.gauss_keras` -- Keras 3 Gaussianization flows.
* `fairkl <https://github.com/jejjohnson/keras-fairkl>`_ -- the
  ``FairModelWrapper`` and a CKA baseline.

The flow is pretrained on a dataset, frozen, and used as a differentiable
"Gaussian-space probe" for measuring dependence between a predictor's
output and a sensitive attribute. See
``docs/fair_gaussianization_experiment.md`` for the full design.
"""

from __future__ import annotations

from gaussianization.fair.freeze import freeze_flow, is_fully_frozen
from gaussianization.fair.losses import (
    GaussianizedMutualInfoLoss,
    GaussianizedTotalCorrelationLoss,
    GaussianizedXCovLoss,
)
from gaussianization.fair.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    pearson_corr,
)
from gaussianization.fair.pretrain import (
    fit_and_freeze,
    fit_and_freeze_joint,
)


__all__ = [
    "GaussianizedMutualInfoLoss",
    "GaussianizedTotalCorrelationLoss",
    "GaussianizedXCovLoss",
    "demographic_parity_difference",
    "equalized_odds_difference",
    "fit_and_freeze",
    "fit_and_freeze_joint",
    "freeze_flow",
    "is_fully_frozen",
    "pearson_corr",
]
