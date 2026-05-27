"""Lorenz-63 forward model + canonical forecast-mode benchmark.

The whole benchmark hangs off `generate_problem(key)`. Following
Ahmed et al. 2020 (PyDA), each problem carries **two** time scales:

1. **Assimilation window** ``t \\in [0, T_assim]`` — the methods see
   observations here.
2. **Free-forecast window** ``t \\in (T_assim, T_total]`` — the
   analysis from step 1 is rolled forward; methods receive no obs
   here. This is where the chaotic divergence becomes visible.

Defaults follow PyDA: total run of 10 time units (``T_total = 1000`` at
``dt = 0.01``, roughly 9 L63 Lyapunov times) with a 2-time-unit assim
window. Observations live on the x-component every 0.2 time units —
partial observations on a chaotic system, the regime where 4DVar
beats OI by an order of magnitude.

Every method consumes the same assim-window `Batch1D`; their analyses
are then extended into the forecast window by free-running the
forward integrator. The harness in `assimilation.benchmark` does this
extension uniformly via `free_forecast` / `assemble_full_trajectory`.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float, PRNGKeyArray
from vardax._src.utils.dynamical_systems import Lorenz63


@dataclass(frozen=True)
class LorenzProblem:
    """Forecast-mode L63 benchmark problem.

    Attributes
    ----------
    truth: ``(T_total+1, 3)`` ground-truth trajectory across the full
        analysis + forecast window.
    obs: ``(T_assim+1, 3)`` observations on the assim window only
        (zero at masked entries).
    mask: ``(T_assim+1, 3)`` binary mask.
    prior_mean: ``(T_assim+1, 3)`` background for OI/3DVar/learned
        methods.
    prior_mean_state: ``(3,)`` background for the 4DVar-family x_0
        control.
    B_op, R_op: covariances over ``(T_assim+1, 3)``.
    B_op_state, R_op_state: covariances over the ``(3,)`` state.
    dt: integration time-step.
    T_assim: number of forecast steps inside the assim window
        (assim trajectory has length ``T_assim + 1``).
    T_total: number of forecast steps across the full run.
    obs_every: temporal obs stride inside the assim window.
    obs_noise: std of the Gaussian observation noise.
    """

    truth: Float[Array, "T_total_plus_1 3"]
    obs: Float[Array, "T_assim_plus_1 3"]
    mask: Float[Array, "T_assim_plus_1 3"]
    prior_mean: Float[Array, "T_assim_plus_1 3"]
    prior_mean_state: Float[Array, 3]
    B_op: lx.AbstractLinearOperator
    R_op: lx.AbstractLinearOperator
    B_op_state: lx.AbstractLinearOperator
    R_op_state: lx.AbstractLinearOperator
    dt: float
    T_assim: int
    T_total: int
    obs_every: int
    obs_noise: float

    @property
    def T_assim_plus_1(self) -> int:
        return self.T_assim + 1

    @property
    def T_total_plus_1(self) -> int:
        return self.T_total + 1

    @property
    def T_forecast(self) -> int:
        return self.T_total - self.T_assim


class Lorenz63Forward(eqx.Module):
    """Lorenz-63 forward model — `pipekit_cycle.ForwardModel` compatible."""

    dt: float
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0

    @property
    def state_signature(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((3,), jnp.float32)

    def _vector_field(self, x: Float[Array, 3]) -> Float[Array, 3]:
        l63 = Lorenz63(sigma=self.sigma, rho=self.rho, beta=self.beta)
        return l63(0.0, x, None)

    def step(self, state: Float[Array, 3], dt: float) -> Float[Array, 3]:
        """Single RK4 step. Signature matches ``pipekit_cycle.ForwardModel.step``."""
        k1 = self._vector_field(state)
        k2 = self._vector_field(state + 0.5 * dt * k1)
        k3 = self._vector_field(state + 0.5 * dt * k2)
        k4 = self._vector_field(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _simulate_truth(
    key: PRNGKeyArray,
    *,
    T: int,
    dt: float,
    sigma: float,
    rho: float,
    beta: float,
    burn_in: int = 1000,
) -> Float[Array, "T_plus_1 3"]:
    """Integrate Lorenz-63 from a slightly-perturbed fixed point.

    We integrate ``burn_in`` extra steps and discard them so the
    starting state sits on the attractor.
    """
    fwd = Lorenz63Forward(dt=dt, sigma=sigma, rho=rho, beta=beta)
    fp = jnp.array(
        [
            jnp.sqrt(beta * (rho - 1.0)),
            jnp.sqrt(beta * (rho - 1.0)),
            rho - 1.0,
        ]
    )
    x0 = fp + 0.01 * jax.random.normal(key, (3,))

    def _scan_step(state, _):
        new = fwd.step(state, dt)
        return new, new

    state, _ = jax.lax.scan(_scan_step, x0, None, length=burn_in)
    _, traj = jax.lax.scan(_scan_step, state, None, length=T)
    return jnp.concatenate([state[None, :], traj], axis=0)


def generate_problem(
    *,
    key: PRNGKeyArray,
    T_assim: int = 50,
    T_total: int = 1000,
    dt: float = 0.01,
    obs_every: int = 5,
    obs_noise: float = 0.5,
    prior_std: float = 5.0,
    observe_components: tuple[int, ...] = (0, 1, 2),
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> LorenzProblem:
    """Generate the forecast-mode Lorenz-63 benchmark (PyDA-style).

    Defaults: 0.5-time-unit assim window (half a Lyapunov time) inside
    a 10-time-unit total run (~9 Lyapunov times); full state
    ``(x, y, z)`` observed every 0.05 time units with Gaussian noise
    std 0.5. With these settings strong-4DVar's BFGS inner solver
    converges reliably on the short window (chaotic-rollout gradients
    become unstable past ~1 Lyapunov time), and the analysis quality
    becomes a clean function of how good a single ``x_0`` each method
    recovers — making the 9-Lyapunov-time free-forecast story the
    headline.

    Parameters
    ----------
    key: PRNG key for the truth's perturbation and obs noise.
    T_assim: forecast steps in the assim window
        (trajectory length there is ``T_assim + 1``).
    T_total: total forecast steps including the free-forecast phase.
    dt: integration time-step.
    obs_every: observe one in every ``obs_every`` time steps within
        the assim window. PyDA uses 20 (every 0.2 time units).
    obs_noise: std of Gaussian observation noise. PyDA-equivalent
        for full-state obs.
    prior_std: std for the diagonal background covariance.
    observe_components: which Lorenz components carry observations.
        Default is full state ``(0, 1, 2)`` (PyDA convention). Pass
        ``(0,)`` to recover the classic "x-only" partial-obs regime.
    sigma, rho, beta: Lorenz-63 parameters.
    """
    if T_total < T_assim:
        raise ValueError(f"T_total ({T_total}) must be >= T_assim ({T_assim}).")

    k_truth, k_obs = jax.random.split(key)
    truth = _simulate_truth(
        k_truth, T=T_total, dt=dt, sigma=sigma, rho=rho, beta=beta
    ).astype(jnp.float32)

    # Mask + obs over the assim window only.
    T_assim_plus_1 = T_assim + 1
    mask = jnp.zeros((T_assim_plus_1, 3), dtype=jnp.float32)
    obs_times = jnp.arange(0, T_assim_plus_1, obs_every)
    comp_idx = jnp.asarray(observe_components, dtype=jnp.int32)
    mask = mask.at[obs_times[:, None], comp_idx[None, :]].set(1.0)

    noise = obs_noise * jax.random.normal(k_obs, (T_assim_plus_1, 3))
    obs = (truth[:T_assim_plus_1] + noise) * mask

    prior_mean = jnp.zeros((T_assim_plus_1, 3), dtype=jnp.float32)
    prior_mean_state = jnp.zeros(3, dtype=jnp.float32)

    B_diag = jnp.full((T_assim_plus_1, 3), prior_std**2, dtype=jnp.float32)
    R_diag = jnp.full((T_assim_plus_1, 3), obs_noise**2, dtype=jnp.float32)
    state_diag = jnp.full((3,), prior_std**2, dtype=jnp.float32)
    state_R_diag = jnp.full((3,), obs_noise**2, dtype=jnp.float32)

    def _pd(op: lx.AbstractLinearOperator) -> lx.AbstractLinearOperator:
        return lx.TaggedLinearOperator(op, lx.positive_semidefinite_tag)

    return LorenzProblem(
        truth=truth,
        obs=obs,
        mask=mask,
        prior_mean=prior_mean,
        prior_mean_state=prior_mean_state,
        B_op=_pd(lx.DiagonalLinearOperator(B_diag)),
        R_op=_pd(lx.DiagonalLinearOperator(R_diag)),
        B_op_state=_pd(lx.DiagonalLinearOperator(state_diag)),
        R_op_state=_pd(lx.DiagonalLinearOperator(state_R_diag)),
        dt=dt,
        T_assim=T_assim,
        T_total=T_total,
        obs_every=obs_every,
        obs_noise=obs_noise,
    )
