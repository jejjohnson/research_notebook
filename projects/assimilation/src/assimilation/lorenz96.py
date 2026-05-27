"""Lorenz-96 (single-level) forward model + canonical benchmark problem.

The Lorenz-96 system is the higher-dimensional sibling of L63: a ring
of ``K`` coupled scalar variables with periodic boundary conditions,
$$
\\dot{x}_k = (x_{k+1} - x_{k-2}) x_{k-1} - x_k + F.
$$

With ``K = 40`` and forcing ``F = 8`` the system is fully chaotic.
This module mirrors `assimilation.lorenz63`: a `Lorenz96Forward`
satisfying `pipekit_cycle.ForwardModel` and a `generate_l96_problem`
factory returning the shared problem object used by every method
notebook.

Observation design (the **partial-and-sparse** setup):

- *Spatial mask*: observe every `obs_every_space`-th grid point
  (default every 4th → 10 of 40 sites).
- *Temporal mask*: observe every `obs_every_time`-th step
  (default every 4th → 6 of 21 time slices).
- Gaussian noise with std `obs_noise` at every observed entry.

Total observed scalars (default): ``10 * 6 = 60``. Total state
unknowns: ``21 * 40 = 840`` — heavily under-determined without
dynamics or structured priors.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float, PRNGKeyArray
from vardax._src.utils.dynamical_systems import Lorenz96


@dataclass(frozen=True)
class LorenzL96Problem:
    """Forecast-mode L96 benchmark problem.

    Following PyDA, every method assimilates obs on the
    ``[0, T_assim]`` window and is then free-forecast through to
    ``T_total``. The truth covers the full window for plotting and
    RMSE; obs/mask/prior are restricted to the assim window.

    Attributes
    ----------
    truth: ``(T_total+1, K)`` full ground-truth trajectory.
    obs: ``(T_assim+1, K)`` observations on the assim window only.
    mask: ``(T_assim+1, K)`` binary mask.
    prior_mean: ``(T_assim+1, K)`` background for OI / 3DVar /
        FourDVarNet / Amortized.
    prior_mean_state: ``(K,)`` background for the 4DVar-family ``x_0``
        control.
    B_op, R_op: covariances over ``(T_assim+1, K)``.
    B_op_state, R_op_state: covariances over ``(K,)``.
    K, F: Lorenz-96 dimension and forcing.
    dt: integration time-step.
    T_assim, T_total: forecast-step counts for the assim window and
        the full run (length T_assim+1 and T_total+1 respectively).
    obs_every_space, obs_every_time: obs strides in the assim window.
    obs_noise: std of the Gaussian observation noise.
    """

    truth: Float[Array, "T_total_plus_1 K"]
    obs: Float[Array, "T_assim_plus_1 K"]
    mask: Float[Array, "T_assim_plus_1 K"]
    prior_mean: Float[Array, "T_assim_plus_1 K"]
    prior_mean_state: Float[Array, " K"]
    B_op: lx.AbstractLinearOperator
    R_op: lx.AbstractLinearOperator
    B_op_state: lx.AbstractLinearOperator
    R_op_state: lx.AbstractLinearOperator
    K: int
    F: float
    dt: float
    T_assim: int
    T_total: int
    obs_every_space: int
    obs_every_time: int
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


class Lorenz96Forward(eqx.Module):
    """Lorenz-96 forward model — `pipekit_cycle.ForwardModel` compatible.

    Wraps the canonical L96 vector field as a one-step RK4 integrator.
    Periodic boundary conditions are handled inside `vardax`'s
    `Lorenz96` via `jnp.roll`. Pure JAX, no diffrax state to thread.

    Attributes
    ----------
    K: state dimension.
    F: forcing.
    dt: integration time-step.
    """

    K: int
    F: float
    dt: float

    @property
    def state_signature(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self.K,), jnp.float32)

    def _vector_field(self, x: Float[Array, " K"]) -> Float[Array, " K"]:
        l96 = Lorenz96(F=self.F)
        return l96(0.0, x, None)

    def step(
        self,
        state: Float[Array, " K"],
        dt: float,
    ) -> Float[Array, " K"]:
        """Advance the state by ``dt`` via one RK4 step."""
        k1 = self._vector_field(state)
        k2 = self._vector_field(state + 0.5 * dt * k1)
        k3 = self._vector_field(state + 0.5 * dt * k2)
        k4 = self._vector_field(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _simulate_l96_truth(
    key: PRNGKeyArray,
    *,
    K: int,
    F: float,
    T: int,
    dt: float,
    burn_in: int = 1000,
) -> Float[Array, "T_plus_1 K"]:
    """Integrate L96 from a small random perturbation of the constant
    forcing equilibrium ``x_k = F`` for ``burn_in`` steps (to land on
    the attractor), then save ``T+1`` snapshots."""
    fwd = Lorenz96Forward(K=K, F=F, dt=dt)
    x0 = F * jnp.ones(K, dtype=jnp.float32) + 0.01 * jax.random.normal(key, (K,))

    def _scan_step(state, _):
        new = fwd.step(state, dt)
        return new, new

    state, _ = jax.lax.scan(_scan_step, x0, None, length=burn_in)
    _, traj = jax.lax.scan(_scan_step, state, None, length=T)
    return jnp.concatenate([state[None, :], traj], axis=0)


def generate_l96_problem(
    *,
    key: PRNGKeyArray,
    K: int = 40,
    F: float = 8.0,
    T_assim: int = 50,
    T_total: int = 250,
    dt: float = 0.01,
    obs_every_space: int = 4,
    obs_every_time: int = 5,
    obs_noise: float = 1.0,
    prior_std: float = 5.0,
) -> LorenzL96Problem:
    """Generate the forecast-mode Lorenz-96 benchmark.

    Defaults: K=40, F=8 (chaotic), 0.5-time-unit assim window inside
    a 2.5-time-unit total run (~5 L96 Lyapunov times). Observe every
    4th grid point every 0.05 time units inside the assim window;
    free-forecast for 2 time units after. The 5-Lyapunov-time
    horizon is short enough that methods with good $x_0$ recovery
    visibly outperform the noisy-analysis baselines.

    Parameters
    ----------
    key: PRNG key for truth + obs noise.
    K: number of grid points on the periodic ring.
    F: Lorenz-96 forcing constant.
    T_assim: forecast steps inside the assim window.
    T_total: total forecast steps over the full run.
    dt: integration time-step.
    obs_every_space: spatial obs stride. ``1`` for full-state, larger
        values for sparser obs.
    obs_every_time: temporal obs stride inside the assim window.
    obs_noise: std of the Gaussian observation noise.
    prior_std: std for the diagonal background covariance.
    """
    if T_total < T_assim:
        raise ValueError(f"T_total ({T_total}) must be >= T_assim ({T_assim}).")

    k_truth, k_obs = jax.random.split(key)
    truth = _simulate_l96_truth(k_truth, K=K, F=F, T=T_total, dt=dt).astype(jnp.float32)

    T_assim_plus_1 = T_assim + 1
    space_idx = jnp.arange(0, K, obs_every_space)
    time_idx = jnp.arange(0, T_assim_plus_1, obs_every_time)
    mask = jnp.zeros((T_assim_plus_1, K), dtype=jnp.float32)
    mask = mask.at[time_idx[:, None], space_idx[None, :]].set(1.0)

    noise = obs_noise * jax.random.normal(k_obs, (T_assim_plus_1, K))
    obs = (truth[:T_assim_plus_1] + noise) * mask

    prior_mean = jnp.zeros((T_assim_plus_1, K), dtype=jnp.float32)
    prior_mean_state = jnp.zeros(K, dtype=jnp.float32)

    B_diag = jnp.full((T_assim_plus_1, K), prior_std**2, dtype=jnp.float32)
    R_diag = jnp.full((T_assim_plus_1, K), obs_noise**2, dtype=jnp.float32)
    state_B_diag = jnp.full((K,), prior_std**2, dtype=jnp.float32)
    state_R_diag = jnp.full((K,), obs_noise**2, dtype=jnp.float32)

    def _pd(op: lx.AbstractLinearOperator) -> lx.AbstractLinearOperator:
        return lx.TaggedLinearOperator(op, lx.positive_semidefinite_tag)

    return LorenzL96Problem(
        truth=truth,
        obs=obs,
        mask=mask,
        prior_mean=prior_mean,
        prior_mean_state=prior_mean_state,
        B_op=_pd(lx.DiagonalLinearOperator(B_diag)),
        R_op=_pd(lx.DiagonalLinearOperator(R_diag)),
        B_op_state=_pd(lx.DiagonalLinearOperator(state_B_diag)),
        R_op_state=_pd(lx.DiagonalLinearOperator(state_R_diag)),
        K=K,
        F=F,
        dt=dt,
        T_assim=T_assim,
        T_total=T_total,
        obs_every_space=obs_every_space,
        obs_every_time=obs_every_time,
        obs_noise=obs_noise,
    )
