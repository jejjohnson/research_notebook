"""Per-method benchmark harness.

Each notebook calls

    result = run_method(name, run_fn, problem)

which times one analysis evaluation, computes the canonical metrics,
and returns a `MethodResult`. The comparison notebook collects a list
of these and turns them into a pandas DataFrame via `compare`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from assimilation.lorenz63 import LorenzProblem
from assimilation.metrics import rmse


@dataclass
class MethodResult:
    """One method's output on the canonical Lorenz-63 problem.

    Attributes
    ----------
    name: short string id (e.g. ``"oi"``, ``"strong_4dvar"``).
    mean: ``(T+1, 3)`` analysis trajectory (the MAP / posterior mean).
    runtime_ms: wall-clock for a single analysis call, in milliseconds.
        Includes the first compile; pass ``warmup=True`` to subtract it.
    train_time_s: training wall-clock for learned methods, in seconds.
        ``None`` for closed-form / iterative methods that don't train.
    rmse_total: scalar RMSE over the whole trajectory.
    rmse_per_component: ``(3,)`` per-component RMSE for the three Lorenz
        states.
    extras: free-form dict — used by individual notebooks for method-
        specific diagnostics (e.g. number of GN outer iterations).
    """

    name: str
    mean: Float[Array, "T_plus_1 3"]
    runtime_ms: float
    rmse_total: float
    rmse_per_component: Float[Array, 3]
    train_time_s: float | None = None
    extras: dict[str, Any] | None = None


def run_method(
    name: str,
    run_fn: Callable[[], Float[Array, "T_plus_1 3"]],
    problem: LorenzProblem,
    *,
    warmup: bool = True,
    train_time_s: float | None = None,
    extras: dict[str, Any] | None = None,
) -> MethodResult:
    """Time ``run_fn`` and compute RMSE against the problem's truth.

    ``run_fn`` is a zero-arg closure that returns the analysis trajectory
    of shape ``(T+1, 3)``. It is called twice when ``warmup=True``: the
    first call's compile time is discarded.
    """
    if warmup:
        _warm = run_fn()
        jax.block_until_ready(_warm)

    t0 = time.perf_counter()
    mean = run_fn()
    jax.block_until_ready(mean)
    runtime_ms = 1e3 * (time.perf_counter() - t0)

    return MethodResult(
        name=name,
        mean=mean,
        runtime_ms=float(runtime_ms),
        rmse_total=float(rmse(mean, problem.truth)),
        rmse_per_component=rmse(mean, problem.truth, axis=0),
        train_time_s=train_time_s,
        extras=extras,
    )


def compare(*results: MethodResult):
    """Stack a list of `MethodResult` into a comparison ``pandas.DataFrame``.

    For 3-D problems (e.g. Lorenz-63) the per-component columns are
    labelled ``rmse_x``, ``rmse_y``, ``rmse_z``. For higher-
    dimensional problems (Lorenz-96 with ``K = 40``) the per-component
    breakdown is collapsed to ``rmse_min / rmse_median / rmse_max``
    over grid points, which is more informative than reporting the
    first three components.

    Returned columns always include ``rmse_total``, ``runtime_ms``,
    ``train_time_s``. Index: ``name``.
    """
    import pandas as pd  # lazy import — harness has no hard pandas dep

    rows = []
    for r in results:
        per_comp = r.rmse_per_component
        n = per_comp.shape[0]
        row = {"rmse_total": r.rmse_total}
        if n == 3:
            row["rmse_x"] = float(per_comp[0])
            row["rmse_y"] = float(per_comp[1])
            row["rmse_z"] = float(per_comp[2])
        else:
            row["rmse_min"] = float(per_comp.min())
            row["rmse_median"] = float(jnp.median(per_comp))
            row["rmse_max"] = float(per_comp.max())
        row["runtime_ms"] = r.runtime_ms
        row["train_time_s"] = r.train_time_s
        rows.append(row)
    return pd.DataFrame(rows, index=[r.name for r in results])
