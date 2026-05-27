"""Lorenz-63 data-assimilation benchmark harness.

A single problem definition (truth trajectory, partial observations,
prior, noise) shared across seven `pipekit_cycle.AnalysisStep` methods
from `vardax`. Each per-method notebook builds the same `LorenzProblem`
and runs it through one analysis method; the comparison notebook stacks
them all side-by-side.
"""

from __future__ import annotations

from assimilation.benchmark import MethodResult, compare, run_method
from assimilation.lorenz63 import (
    Lorenz63Forward,
    LorenzProblem,
    generate_problem,
)
from assimilation.lorenz96 import (
    Lorenz96Forward,
    LorenzL96Problem,
    generate_l96_problem,
)
from assimilation.metrics import nll_gaussian, rmse, sigma_coverage


__all__ = [
    "Lorenz63Forward",
    "Lorenz96Forward",
    "LorenzL96Problem",
    "LorenzProblem",
    "MethodResult",
    "compare",
    "generate_l96_problem",
    "generate_problem",
    "nll_gaussian",
    "rmse",
    "run_method",
    "sigma_coverage",
]
