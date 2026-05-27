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
    """Shared L96 benchmark problem.

    Attributes
    ----------
    truth: ``(T+1, K)`` ground-truth trajectory.
    obs: ``(T+1, K)`` observations, zero at masked entries.
    mask: ``(T+1, K)`` binary mask combining spatial and temporal
        sparsity. ``1`` where the entry is observed.
    prior_mean: ``(T+1, K)`` background mean for OI/3DVar/learned
        methods (used as ``y0`` for the iterative solvers).
    prior_mean_state: ``(K,)`` background for the 4DVar-family ``x_0``
        control.
    B_op, R_op: covariances over ``(T+1, K)`` — used by OI / 3DVar /
        learned heads.
    B_op_state, R_op_state: covariances over ``(K,)`` — used by the
        4DVar family.
    K: state dimension (number of grid points on the periodic ring).
    F: Lorenz-96 forcing.
    dt: integration time-step.
    T: number of forecast steps. Trajectories have length ``T + 1``.
    obs_every_space, obs_every_time: stride of the spatial / temporal
        observation grids.
    obs_noise: std of the Gaussian observation noise.
    """

    truth: Float[Array, "T_plus_1 K"]
    obs: Float[Array, "T_plus_1 K"]
    mask: Float[Array, "T_plus_1 K"]
    prior_mean: Float[Array, "T_plus_1 K"]
    prior_mean_state: Float[Array, " K"]
    B_op: lx.AbstractLinearOperator
    R_op: lx.AbstractLinearOperator
    B_op_state: lx.AbstractLinearOperator
    R_op_state: lx.AbstractLinearOperator
    K: int
    F: float
    dt: float
    T: int
    obs_every_space: int
    obs_every_time: int
    obs_noise: float

    @property
    def T_plus_1(self) -> int:
        return self.T + 1


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
    T: int = 20,
    dt: float = 0.01,
    obs_every_space: int = 4,
    obs_every_time: int = 4,
    obs_noise: float = 1.0,
    prior_std: float = 5.0,
) -> LorenzL96Problem:
    """Generate the canonical Lorenz-96 partial-obs assimilation problem.

    Defaults match the textbook partial-obs setup: ``K = 40``,
    ``F = 8`` (chaotic), 21-step window, observe every 4th grid
    point at every 4th time step.

    Parameters
    ----------
    key: PRNG key for truth + obs noise.
    K: number of grid points on the periodic ring.
    F: Lorenz-96 forcing constant.
    T: number of forecast steps. Trajectory has length ``T + 1``.
    dt: integration time-step.
    obs_every_space: spatial obs stride. ``1`` for full-state, larger
        values for sparser obs.
    obs_every_time: temporal obs stride.
    obs_noise: std of the Gaussian observation noise.
    prior_std: std for the diagonal background covariance.

    Returns
    -------
    A `LorenzL96Problem` carrying every operator each method needs.
    """
    k_truth, k_obs = jax.random.split(key)
    truth = _simulate_l96_truth(k_truth, K=K, F=F, T=T, dt=dt).astype(jnp.float32)

    T_plus_1 = T + 1
    # Combined spatial / temporal mask.
    space_idx = jnp.arange(0, K, obs_every_space)
    time_idx = jnp.arange(0, T_plus_1, obs_every_time)
    mask = jnp.zeros((T_plus_1, K), dtype=jnp.float32)
    mask = mask.at[time_idx[:, None], space_idx[None, :]].set(1.0)

    # Noisy observations at observed entries; zero elsewhere.
    noise = obs_noise * jax.random.normal(k_obs, (T_plus_1, K))
    obs = (truth + noise) * mask

    prior_mean = jnp.zeros((T_plus_1, K), dtype=jnp.float32)
    prior_mean_state = jnp.zeros(K, dtype=jnp.float32)

    B_diag = jnp.full((T_plus_1, K), prior_std**2, dtype=jnp.float32)
    R_diag = jnp.full((T_plus_1, K), obs_noise**2, dtype=jnp.float32)
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
        T=T,
        obs_every_space=obs_every_space,
        obs_every_time=obs_every_time,
        obs_noise=obs_noise,
    )
