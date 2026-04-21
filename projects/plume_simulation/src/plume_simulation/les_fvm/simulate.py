"""High-level diffrax runner returning an xarray ``Dataset``.

This module wires every other ``les_fvm`` piece together behind a single
user-facing entry point, :func:`simulate_eulerian_dispersion`, modelled
after ``gauss_puff.simulate_puff`` for cross-model consistency.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import diffrax
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jaxtyping import Array, Float

from plume_simulation.gauss_puff.wind import WindSchedule
from plume_simulation.les_fvm.boundary import build_default_concentration_bc
from plume_simulation.les_fvm.diffusion import EddyDiffusivity, make_eddy_diffusivity
from plume_simulation.les_fvm.dynamics import EulerianDispersionRHS
from plume_simulation.les_fvm.grid import PlumeGrid3D, coord_to_numpy, make_grid
from plume_simulation.les_fvm.source import make_gaussian_source
from plume_simulation.les_fvm.wind import (
    PrescribedWindField,
    uniform_wind_field,
    wind_field_from_schedule,
)


SolverName = Literal["tsit5", "dopri5", "ssprk3", "heun"]
AdvectionScheme = Literal[
    "upwind1", "weno3", "weno5", "weno7", "weno9",
    "minmod", "van_leer", "superbee", "mc", "naive",
]


def simulate_eulerian_dispersion(
    *,
    # Domain and resolution
    domain_x: tuple[float, float, int],
    domain_y: tuple[float, float, int],
    domain_z: tuple[float, float, int],
    # Time
    t_start: float,
    t_end: float,
    save_interval: float,
    # Source
    emission_rate: float | Callable[[Float[Array, ""]], Float[Array, ""]],
    source_location: tuple[float, float, float],
    source_radius: float | None = None,
    # Flow
    wind_schedule: WindSchedule | None = None,
    uniform_wind: tuple[float, float, float] | None = None,
    wind_field: PrescribedWindField | None = None,
    # SGS / eddy diffusivity
    eddy_diffusivity: (
        float | tuple[float, float] | str | EddyDiffusivity
    ) = "pg",
    stability_class: str | None = None,
    pg_reference_distance: float = 200.0,
    # Numerics
    advection_scheme: AdvectionScheme = "weno5",
    bc_x: (
        str
        | tuple[str, str]
        | tuple[tuple[str, float], tuple[str, float]]
    ) = ("dirichlet", "outflow"),
    bc_y: (
        str
        | tuple[str, str]
        | tuple[tuple[str, float], tuple[str, float]]
    ) = "periodic",
    bc_z: (
        str
        | tuple[str, str]
        | tuple[tuple[str, float], tuple[str, float]]
    ) = ("neumann", "neumann"),
    solver: SolverName = "tsit5",
    dt0: float = 1.0,
    rtol: float = 1e-3,
    atol: float = 1e-8,
    max_steps: int = 100_000,
    # Initial condition
    initial_concentration: Float[Array, "Nz Ny Nx"] | None = None,
    seed: int = 0,
) -> xr.Dataset:
    """Simulate 3-D Eulerian tracer transport under a prescribed wind.

    Parameters
    ----------
    domain_x, domain_y, domain_z : tuple (min, max, n_interior)
        Physical extent + number of interior cells per axis.
    t_start, t_end : float
        Simulation window [s].
    save_interval : float
        Save cadence [s].  Saved times are
        ``np.arange(t_start, t_end + save_interval, save_interval)``.
    emission_rate : float or callable
        Methane source rate [kg/s] (constant or ``t -> q(t)``).
    source_location : tuple (x, y, z)
        Release point [m].
    source_radius : float, optional
        Gaussian source radius [m].  Defaults to ``2 · max(dx, dy, dz)``.
    wind_schedule : WindSchedule, optional
    uniform_wind : (u, v, w), optional
    wind_field : PrescribedWindField, optional
        One of these three must be provided — they are mutually exclusive.
    eddy_diffusivity : float | (K_h, K_z) | "pg" | EddyDiffusivity
        K-theory eddy diffusivity spec.  The string ``"pg"`` pulls a rough
        calibration from the PG σ curves — requires ``stability_class`` and
        a non-zero mean wind speed.
    stability_class : str, optional
        Required when ``eddy_diffusivity="pg"``.
    pg_reference_distance : float, default 200.0
        Reference downwind distance for PG → K-theory conversion.
    advection_scheme : {"upwind1", "weno3", "weno5", ...}, default "weno5"
        Horizontal reconstruction scheme.
    bc_x, bc_y, bc_z : BC specs
        See :func:`boundary.build_default_concentration_bc`.
    solver : {"tsit5", "dopri5", "ssprk3", "heun"}, default "tsit5"
        diffrax time integrator.  ``"ssprk3"`` and ``"heun"`` are
        explicit-fixed-step and ignore ``rtol``/``atol``.
    dt0 : float
        Initial / fixed step size [s].
    rtol, atol : float
        Adaptive-step controller tolerances (ignored for fixed-step solvers).
    max_steps : int
        Upper bound on diffrax internal steps.
    initial_concentration : ndarray, optional
        Interior-shaped ``(nz, ny, nx)`` initial tracer field.  Defaults
        to zero.
    seed : int
        Currently unused (reserved for future stochastic variants).

    Returns
    -------
    xr.Dataset
        ``concentration(time, z, y, x)`` [kg/m³] and the diagnostic
        ``column_concentration(time, y, x)`` [kg/m²].  The dataset
        carries model metadata in ``ds.attrs``.

    Raises
    ------
    ValueError
        For inconsistent wind/source/emission specs.
    """
    del seed  # Reserved for future stochastic branch.

    plume_grid = make_grid(domain_x, domain_y, domain_z)

    wf = _resolve_wind_field(
        plume_grid=plume_grid,
        wind_schedule=wind_schedule,
        uniform_wind=uniform_wind,
        wind_field=wind_field,
    )

    source = make_gaussian_source(
        plume_grid=plume_grid,
        emission_rate=emission_rate,
        source_location=source_location,
        source_radius=source_radius,
    )

    eddy = _resolve_eddy_diffusivity(
        eddy_diffusivity=eddy_diffusivity,
        stability_class=stability_class,
        wind_schedule=wind_schedule,
        uniform_wind=uniform_wind,
        pg_reference_distance=pg_reference_distance,
        t_start=t_start,
        t_end=t_end,
    )

    horizontal_bc, vertical_bc = build_default_concentration_bc(
        bc_x=bc_x, bc_y=bc_y, bc_z=bc_z
    )

    rhs = EulerianDispersionRHS(
        plume_grid=plume_grid,
        wind_field=wf,
        eddy_diffusivity=eddy,
        source=source,
        horizontal_bc=horizontal_bc,
        vertical_bc=vertical_bc,
        advection_scheme=advection_scheme,
    )

    concentration0 = _build_initial_concentration(
        plume_grid=plume_grid, initial_concentration=initial_concentration
    )
    save_times = _build_save_times(
        t_start=t_start, t_end=t_end, save_interval=save_interval
    )
    solution = _solve(
        rhs=rhs,
        t_start=t_start,
        t_end=t_end,
        dt0=dt0,
        y0=concentration0,
        save_times=save_times,
        solver_name=solver,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
    )
    return _to_dataset(
        plume_grid=plume_grid,
        times=np.asarray(solution.ts),
        concentration_history=np.asarray(solution.ys),
        attrs={
            "title": "Eulerian 3-D methane dispersion (finitevolX)",
            "model": "L2: prescribed wind + K-theory diffusivity",
            "advection_scheme": advection_scheme,
            "solver": solver,
        },
    )


def _resolve_wind_field(
    *,
    plume_grid: PlumeGrid3D,
    wind_schedule: WindSchedule | None,
    uniform_wind: tuple[float, float, float] | None,
    wind_field: PrescribedWindField | None,
) -> PrescribedWindField:
    provided = sum(
        arg is not None
        for arg in (wind_schedule, uniform_wind, wind_field)
    )
    if provided != 1:
        raise ValueError(
            "simulate_eulerian_dispersion: exactly one of "
            "`wind_schedule`, `uniform_wind`, `wind_field` must be provided"
        )
    if wind_field is not None:
        return wind_field
    if wind_schedule is not None:
        return wind_field_from_schedule(plume_grid=plume_grid, schedule=wind_schedule)
    u_mean, v_mean, w_mean = uniform_wind
    return uniform_wind_field(
        plume_grid=plume_grid, u=u_mean, v=v_mean, w=w_mean
    )


def _resolve_eddy_diffusivity(
    *,
    eddy_diffusivity,
    stability_class: str | None,
    wind_schedule: WindSchedule | None,
    uniform_wind: tuple[float, float, float] | None,
    pg_reference_distance: float,
    t_start: float,
    t_end: float,
) -> EddyDiffusivity:
    wind_speed = None
    if uniform_wind is not None:
        u_mean, v_mean, _ = uniform_wind
        wind_speed = float(np.hypot(u_mean, v_mean))
    elif wind_schedule is not None:
        # Time-mean speed over the *simulated* window [t_start, t_end]. We
        # sample the piecewise-linear schedule at densely-spaced points and
        # take the mean of |V(t)| — this respects the simulation window
        # and knot spacing, so adding knots outside the run interval or
        # changing knot density does not move the calibration.
        wind_speed = _mean_schedule_speed(wind_schedule, t_start, t_end)
    if (
        isinstance(eddy_diffusivity, str)
        and eddy_diffusivity.lower() == "pg"
        and wind_speed is not None
        and wind_speed == 0.0
    ):
        raise ValueError(
            "simulate_eulerian_dispersion: eddy_diffusivity='pg' requires a "
            "non-zero mean wind speed to compute the K-theory calibration"
        )
    return make_eddy_diffusivity(
        eddy_diffusivity,
        stability_class=stability_class,
        wind_speed=wind_speed,
        reference_distance=pg_reference_distance,
    )


def _mean_schedule_speed(
    schedule: WindSchedule, t_start: float, t_end: float
) -> float:
    """Time-mean wind speed of a piecewise-linear schedule over ``[t_start, t_end]``.

    Integrates ``|V(t)|`` with np.interp on a dense uniform sample grid
    and returns the window mean.  100 samples is enough — the result is
    only used to calibrate a scalar K-theory diffusivity.
    """
    if t_end <= t_start:
        # Degenerate window — fall back to the instantaneous speed at t_start.
        u_at = float(np.interp(
            t_start, np.asarray(schedule.times), np.asarray(schedule.u_wind)
        ))
        v_at = float(np.interp(
            t_start, np.asarray(schedule.times), np.asarray(schedule.v_wind)
        ))
        return float(np.hypot(u_at, v_at))
    t_sample = np.linspace(t_start, t_end, 100)
    u_sample = np.interp(
        t_sample, np.asarray(schedule.times), np.asarray(schedule.u_wind)
    )
    v_sample = np.interp(
        t_sample, np.asarray(schedule.times), np.asarray(schedule.v_wind)
    )
    return float(np.hypot(u_sample, v_sample).mean())


def _build_save_times(
    *, t_start: float, t_end: float, save_interval: float
) -> Float[Array, "M"]:
    """Return save-time knots clipped to ``[t_start, t_end]``.

    ``jnp.arange(t_start, t_end + 0.5 * save_interval, save_interval)`` can
    step past ``t_end`` when the interval does not evenly divide the run
    window (e.g. ``0..10`` with ``save_interval = 6`` gives ``[0, 6, 12]``).
    diffrax's ``SaveAt(ts=...)`` rejects save points outside the integration
    interval, so we clip explicitly and dedupe the trailing value.
    """
    if save_interval <= 0.0:
        raise ValueError(
            f"save_interval must be > 0 (got {save_interval!r})"
        )
    if t_end < t_start:
        raise ValueError(
            f"t_end must be >= t_start (got t_start={t_start}, t_end={t_end})"
        )
    raw = np.arange(t_start, t_end + 0.5 * save_interval, save_interval)
    raw = raw[raw <= t_end + 1e-9]  # guard against fp round-up
    # Snap the last knot to exactly t_end when it falls short, so the final
    # save point matches the user-requested simulation horizon.
    if raw.size > 0 and raw[-1] < t_end - 1e-9:
        raw = np.concatenate([raw, np.asarray([t_end])])
    return jnp.asarray(raw, dtype=jnp.float32)


def _build_initial_concentration(
    *,
    plume_grid: PlumeGrid3D,
    initial_concentration: Float[Array, "nz ny nx"] | None,
) -> Float[Array, "Nz Ny Nx"]:
    Nz, Ny, Nx = plume_grid.shape
    if initial_concentration is None:
        return jnp.zeros((Nz, Ny, Nx), dtype=plume_grid.x.dtype)
    c0 = jnp.asarray(initial_concentration, dtype=plume_grid.x.dtype)
    expected = plume_grid.interior_shape
    if c0.shape != expected:
        raise ValueError(
            f"simulate_eulerian_dispersion: initial_concentration shape "
            f"{c0.shape!r} does not match interior shape {expected!r}"
        )
    return jnp.pad(c0, ((1, 1), (1, 1), (1, 1)), mode="constant")


def _pick_solver(name: SolverName):
    if name == "tsit5":
        return diffrax.Tsit5(), True
    if name == "dopri5":
        return diffrax.Dopri5(), True
    if name == "ssprk3":
        from finitevolx import RK3SSP

        return RK3SSP(), False
    if name == "heun":
        return diffrax.Heun(), False
    raise ValueError(f"simulate_eulerian_dispersion: unknown solver {name!r}")


@eqx.filter_jit
def _solve_jit(
    rhs: EulerianDispersionRHS,
    t_start: float,
    t_end: float,
    dt0: float,
    y0: Float[Array, "Nz Ny Nx"],
    save_times: Float[Array, "M"],
    solver,
    stepsize_controller,
    max_steps: int,
):
    term = diffrax.ODETerm(rhs)
    return diffrax.diffeqsolve(
        term,
        solver,
        t0=t_start,
        t1=t_end,
        dt0=dt0,
        y0=y0,
        saveat=diffrax.SaveAt(ts=save_times),
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
    )


def _solve(
    *,
    rhs: EulerianDispersionRHS,
    t_start: float,
    t_end: float,
    dt0: float,
    y0: Float[Array, "Nz Ny Nx"],
    save_times: Float[Array, "M"],
    solver_name: SolverName,
    rtol: float,
    atol: float,
    max_steps: int,
):
    solver, adaptive = _pick_solver(solver_name)
    if adaptive:
        stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
    else:
        stepsize_controller = diffrax.ConstantStepSize()
    return _solve_jit(
        rhs=rhs,
        t_start=float(t_start),
        t_end=float(t_end),
        dt0=float(dt0),
        y0=y0,
        save_times=save_times,
        solver=solver,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
    )


def _to_dataset(
    *,
    plume_grid: PlumeGrid3D,
    times: np.ndarray,
    concentration_history: np.ndarray,
    attrs: dict,
) -> xr.Dataset:
    """Package solver output as an xarray ``Dataset`` on the interior grid."""
    # concentration_history has shape (n_saves, Nz, Ny, Nx) — drop ghost rings.
    interior = concentration_history[:, 1:-1, 1:-1, 1:-1]
    x_np, y_np, z_np = coord_to_numpy(plume_grid)
    ds = xr.Dataset(
        {"concentration": (["time", "z", "y", "x"], interior)},
        coords={
            "time": (["time"], times),
            "z": (["z"], z_np),
            "y": (["y"], y_np),
            "x": (["x"], x_np),
        },
        attrs=attrs,
    )
    ds["concentration"].attrs = {"long_name": "Methane concentration", "units": "kg/m³"}
    ds["column_concentration"] = ds["concentration"].sum(dim="z") * plume_grid.dz
    ds["column_concentration"].attrs = {
        "long_name": "Column-integrated methane",
        "units": "kg/m²",
    }
    return ds


