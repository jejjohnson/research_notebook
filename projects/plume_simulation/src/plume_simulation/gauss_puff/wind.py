"""Time-varying wind schedules with diffrax-based cumulative integrals.

A Gaussian puff released at time ``t_r`` under a time-varying wind field
``V(t) = (u(t), v(t))`` satisfies the trivial ODE ``dX/dt = V(t)``, whose
integral is

    X(t) − X(t_r) = ∫_{t_r}^{t} V(τ) dτ .

Because every puff sees the same wind, the per-puff offsets are differences
of a single *cumulative* integral evaluated at puff release times and the
current simulation time:

    I_u(t) = ∫_{t_0}^{t} u(τ) dτ                          [m]
    I_v(t) = ∫_{t_0}^{t} v(τ) dτ                          [m]
    S(t)   = ∫_{t_0}^{t} √(u(τ)² + v(τ)²) dτ              [m]   (travel distance)

We compute these with :func:`diffrax.diffeqsolve` driven by a piecewise
-linear wind interpolation, so the result is exact for piecewise-linear
winds up to the solver tolerance, and handles arbitrary continuous winds
with adaptive stepping.

Public surface
--------------
- ``WindSchedule``              : equinox.Module wrapping (t, u, v) arrays
- ``cumulative_wind_integrals`` : diffrax solve returning ``(I_u, I_v, S)``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import diffrax
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

if TYPE_CHECKING:  # NumPy arrays are accepted as factory inputs only.
    import numpy as np


class WindSchedule(eqx.Module):
    """Piecewise-linear time-varying wind field.

    Attributes
    ----------
    times : Float[Array, "T"]
        Monotone increasing time knots [s].
    u_wind : Float[Array, "T"]
        x-component of wind velocity at the knots [m/s].
    v_wind : Float[Array, "T"]
        y-component of wind velocity at the knots [m/s].

    Notes
    -----
    The schedule behaves as a piecewise-linear interpolant between knots.
    Queries outside ``[times[0], times[-1]]`` linearly extrapolate — consumers
    that need strict clamping should range-check times before calling.
    """

    times: Float[Array, "T"]
    u_wind: Float[Array, "T"]
    v_wind: Float[Array, "T"]

    @classmethod
    def from_speed_direction(
        cls,
        times: np.ndarray | jnp.ndarray,
        wind_speed: np.ndarray | jnp.ndarray,
        wind_direction: np.ndarray | jnp.ndarray,
    ) -> WindSchedule:
        """Build a schedule from (speed, direction) in meteorological "from" convention.

        ``wind_direction = 270°`` is a wind *from* the west (u > 0);
        ``wind_direction = 0°`` is a wind *from* the north (v < 0).

            u = −|V| · sin(θ),   v = −|V| · cos(θ),   θ = direction (radians).

        Parameters
        ----------
        times : array-like, shape (T,)
            Monotone time knots [s].
        wind_speed : array-like, shape (T,)
            Wind speed at each knot [m/s]. Must be ≥ 0.
        wind_direction : array-like, shape (T,)
            Wind direction at each knot [degrees from North].

        Returns
        -------
        WindSchedule
        """
        times_j = jnp.asarray(times, dtype=jnp.float32)
        speed_j = jnp.asarray(wind_speed, dtype=jnp.float32)
        theta = jnp.deg2rad(jnp.asarray(wind_direction, dtype=jnp.float32))
        u = -speed_j * jnp.sin(theta)
        v = -speed_j * jnp.cos(theta)
        return cls(times=times_j, u_wind=u, v_wind=v)

    def _interp_u(self) -> diffrax.LinearInterpolation:
        return diffrax.LinearInterpolation(ts=self.times, ys=self.u_wind)

    def _interp_v(self) -> diffrax.LinearInterpolation:
        return diffrax.LinearInterpolation(ts=self.times, ys=self.v_wind)

    def wind_at(
        self, t: Float[Array, ""]
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        """Interpolate the wind velocity components at time ``t``."""
        return self._interp_u().evaluate(t), self._interp_v().evaluate(t)

    def speed_at(self, t: Float[Array, ""]) -> Float[Array, ""]:
        """Interpolate the wind speed |V(t)| at time ``t``."""
        u, v = self.wind_at(t)
        return jnp.sqrt(u**2 + v**2)


@eqx.filter_jit
def cumulative_wind_integrals(
    schedule: WindSchedule,
    save_at: Float[Array, "M"],
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> tuple[Float[Array, "M"], Float[Array, "M"], Float[Array, "M"]]:
    """Compute cumulative wind integrals via a single diffrax ODE solve.

    Integrates ``d(I_u, I_v, S)/dt = (u(t), v(t), √(u² + v²))`` from
    ``schedule.times[0]`` to ``max(save_at[-1], schedule.times[-1])`` with
    initial condition ``(0, 0, 0)`` using :class:`diffrax.Tsit5` with an
    adaptive :class:`diffrax.PIDController`, saving at the requested times.

    Parameters
    ----------
    schedule : WindSchedule
        Time-varying wind field.
    save_at : Float[Array, "M"]
        **Monotone non-decreasing** times [s] at which to return the integrals.
        Save times outside ``[schedule.times[0], schedule.times[-1]]`` are
        evaluated using the linear extrapolation supported by
        :attr:`WindSchedule` — use the schedule knots to bound the physically
        meaningful range when that matters.
    rtol, atol : float
        Relative and absolute tolerances for the adaptive step controller.

    Returns
    -------
    I_u, I_v, S : Float[Array, "M"]
        Cumulative integrals of u, v, and wind speed, each evaluated at
        ``save_at``.
    """
    interp_u = schedule._interp_u()
    interp_v = schedule._interp_v()

    def rhs(t: Float[Array, ""], y: Float[Array, "3"], args) -> Float[Array, "3"]:
        u = interp_u.evaluate(t)
        v = interp_v.evaluate(t)
        speed = jnp.sqrt(u**2 + v**2)
        return jnp.stack([u, v, speed])

    t0 = schedule.times[0]
    t1 = jnp.maximum(save_at[-1], schedule.times[-1])

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs),
        diffrax.Tsit5(),
        t0=t0,
        t1=t1,
        dt0=None,
        y0=jnp.zeros(3, dtype=schedule.u_wind.dtype),
        saveat=diffrax.SaveAt(ts=save_at),
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        max_steps=None,
    )
    ys = sol.ys
    return ys[:, 0], ys[:, 1], ys[:, 2]
