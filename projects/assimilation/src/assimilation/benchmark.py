"""Per-method benchmark harness — assim window + free forecast.

Following Ahmed et al. 2020 (PyDA), every method runs on the same
assim-window batch and is then extended into the forecast window by
free-running the forward integrator. This harness provides the small
set of helpers that make the analysis-then-forecast composition
uniform across the seven `AnalysisStep` methods.

Workflow inside each notebook:

    batch = assim_batch(problem)
    analysis_or_x0 = method(batch)
    full = assemble_full_trajectory(analysis_or_x0, problem, forward)
    result = run_method(name, lambda: full, problem)

`run_method` times the call and computes RMSE over three regions:
the full trajectory, the assim window only, and the forecast window
only — so the table can show analysis quality and forecast skill
side by side.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import vardax as vdx
from jaxtyping import Array, Float

from assimilation.metrics import rmse


class _ForecastProblem(Protocol):
    """Minimal duck-type for the three Lorenz problem dataclasses.

    The harness reads only ``truth``, ``obs``, ``mask``, ``dt``,
    ``T_assim``, ``T_total``.
    """

    truth: Float[Array, "T_total_plus_1 D"]
    obs: Float[Array, "T_assim_plus_1 D"]
    mask: Float[Array, "T_assim_plus_1 D"]
    dt: float
    T_assim: int
    T_total: int


def assim_batch(problem: _ForecastProblem) -> vdx.Batch1D:
    """Build the per-method input batch from the problem's assim window."""
    return vdx.Batch1D(
        input=problem.obs[None],
        mask=problem.mask[None],
        target=None,
    )


def free_forecast(
    x0: Float[Array, " D"],
    n_steps: int,
    forward: Any,
) -> Float[Array, "n_steps_plus_1 D"]:
    """Roll the forward model forward ``n_steps`` from ``x0``.

    Returns the trajectory **including** ``x0`` at index 0, so the
    output has length ``n_steps + 1``.
    """
    if n_steps == 0:
        return x0[None, :]

    def step(s, _):
        new = forward.step(s, forward.dt)
        return new, new

    _, traj = jax.lax.scan(step, x0, None, length=n_steps)
    return jnp.concatenate([x0[None, :], traj], axis=0)


def assemble_full_trajectory(
    analysis: Float[Array, "T_assim_plus_1 D"] | Float[Array, " D"],
    problem: _ForecastProblem,
    forward: Any,
    *,
    etas: Float[Array, "T_assim D"] | None = None,
) -> Float[Array, "T_total_plus_1 D"]:
    """Build the full ``(T_total+1, D)`` trajectory from a method's output.

    The harness handles three shapes of analysis output:

    - **1-D state** ``(D,)`` — 4DVar-family methods that return only
      :math:`x_0^*`. Free-forecast all ``T_total`` steps from it.
    - **2-D trajectory** ``(T_assim+1, D)`` — OI / 3DVar / FourDVarNet
      / AmortizedPosterior, which estimate the entire assim window.
      Take the last state and free-forecast ``T_forecast`` more steps.
    - **2-D trajectory + ``etas``** — weak-4DVar. Roll the perturbed
      forward through the assim window using ``etas``, then free-
      forecast (no model error) for the rest.
    """
    if analysis.ndim == 1:
        # 4DVar-family: x_0 only.
        return free_forecast(analysis, problem.T_total, forward)

    if etas is not None:
        # Weak-4DVar: re-do the perturbed rollout, then free-forecast.
        x0 = analysis
        T_assim = problem.T_assim

        def perturbed_step(s, eta):
            new = forward.step(s, forward.dt) + eta
            return new, new

        _, perturbed_traj = jax.lax.scan(perturbed_step, x0, etas)
        assim_traj = jnp.concatenate([x0[None, :], perturbed_traj], axis=0)
        if T_assim + 1 != assim_traj.shape[0]:
            raise ValueError(
                f"weak-4DVar etas length {etas.shape[0]} != T_assim {T_assim}."
            )
    else:
        assim_traj = analysis
    forecast = free_forecast(assim_traj[-1], problem.T_forecast, forward)
    # Skip the duplicated state at t = T_assim.
    return jnp.concatenate([assim_traj, forecast[1:]], axis=0)


@dataclass
class MethodResult:
    """One method's output on a shared benchmark problem.

    Attributes
    ----------
    name: short string id (e.g. ``"oi"``, ``"strong_4dvar"``).
    mean: ``(T_total+1, D)`` analysis-then-forecast trajectory.
    runtime_ms: wall-clock for a single analysis call, in milliseconds
        (excludes the JIT compile when ``warmup=True`` was passed).
    train_time_s: training wall-clock for learned methods, in seconds.
    rmse_total: scalar RMSE over the full ``(T_total+1, D)`` trajectory.
    rmse_assim: scalar RMSE over the assim window only.
    rmse_forecast: scalar RMSE over the free-forecast window only.
    rmse_per_component: ``(D,)`` per-component RMSE over the full window.
    rmse_trace: ``(T_total+1,)`` instantaneous spatial RMSE per
        time step. ``rmse_trace[t]`` quantifies how far the analysis-
        plus-forecast is from truth at time ``t``.
    extras: free-form dict for method-specific diagnostics.
    """

    name: str
    mean: Float[Array, "T_total_plus_1 D"]
    runtime_ms: float
    rmse_total: float
    rmse_assim: float
    rmse_forecast: float
    rmse_per_component: Float[Array, " D"]
    rmse_trace: Float[Array, " T_total_plus_1"]
    train_time_s: float | None = None
    extras: dict[str, Any] | None = None


def run_method(
    name: str,
    run_fn: Callable[[], Float[Array, "T_total_plus_1 D"]],
    problem: _ForecastProblem,
    *,
    warmup: bool = True,
    train_time_s: float | None = None,
    extras: dict[str, Any] | None = None,
) -> MethodResult:
    """Time ``run_fn`` and compute the per-region RMSE metrics.

    ``run_fn`` is a zero-arg closure that returns the full
    ``(T_total+1, D)`` trajectory (analysis + free forecast). The
    `assemble_full_trajectory` helper exists to make this composition
    one line per method.
    """
    if warmup:
        _warm = run_fn()
        jax.block_until_ready(_warm)

    t0 = time.perf_counter()
    mean = run_fn()
    jax.block_until_ready(mean)
    runtime_ms = 1e3 * (time.perf_counter() - t0)

    # Per-time-step spatial RMSE; per-region averages collapse this.
    per_step = jnp.sqrt(jnp.mean((mean - problem.truth) ** 2, axis=-1))
    T_assim_plus_1 = problem.T_assim + 1
    return MethodResult(
        name=name,
        mean=mean,
        runtime_ms=float(runtime_ms),
        rmse_total=float(rmse(mean, problem.truth)),
        rmse_assim=float(rmse(mean[:T_assim_plus_1], problem.truth[:T_assim_plus_1])),
        rmse_forecast=float(
            rmse(mean[T_assim_plus_1:], problem.truth[T_assim_plus_1:])
        ),
        rmse_per_component=rmse(mean, problem.truth, axis=0),
        rmse_trace=per_step,
        train_time_s=train_time_s,
        extras=extras,
    )


def compare(*results: MethodResult):
    """Stack `MethodResult`\\s into a comparison DataFrame.

    Columns: ``rmse_total``, ``rmse_assim``, ``rmse_forecast``,
    plus either ``(rmse_x, rmse_y, rmse_z)`` for 3-D problems or
    ``(rmse_min, rmse_median, rmse_max)`` for higher-dim. Plus
    ``runtime_ms`` and ``train_time_s``.
    """
    import pandas as pd  # lazy import — harness has no hard pandas dep

    rows = []
    for r in results:
        per_comp = r.rmse_per_component
        n = per_comp.shape[0]
        row = {
            "rmse_total": r.rmse_total,
            "rmse_assim": r.rmse_assim,
            "rmse_forecast": r.rmse_forecast,
        }
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
