"""Lorenz-63 forward model + canonical assimilation problem.

The whole benchmark hangs off `generate_problem(key)`: every notebook
loads the *identical* `(truth, obs, mask, prior_mean, B_op, R_op, T, dt)`
tuple, then runs it through one analysis method. Re-using the same
problem across methods is the only way the comparison numbers are
apples-to-apples.

Observation setup (classic Lorenz partial-obs):

- Observe only the **x-component** (the canonical noisy / chaotic
  driver). y and z must be recovered through cross-covariances or
  dynamics.
- Observe every `obs_every` time-steps, not every step.
- Gaussian observation noise with std `obs_noise`.

This is harder than the full-state case and is the regime in which
4DVar-family methods most clearly beat OI / 3DVar.
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
    """Shared benchmark problem.

    Attributes
    ----------
    truth: ``(T+1, 3)`` ground-truth Lorenz-63 trajectory.
    obs: ``(T+1, 3)`` observations (NaN-clean, zeros at masked entries).
    mask: ``(T+1, 3)`` binary mask — 1 at observed (time, component) pairs.
    prior_mean: ``(T+1, 3)`` background mean for OI/3DVar/learned methods.
    prior_mean_state: ``(3,)`` background for the 4DVar-family x_0 control.
    B_op: prior covariance as a lineax operator over ``(T+1, 3)``.
    R_op: observation covariance over ``(T+1, 3)``.
    B_op_state: prior covariance over the ``(3,)`` initial state.
    R_op_state: observation covariance over the ``(3,)`` per-step obs.
    dt: integration time-step (seconds in the canonical scaling).
    T: number of forecast steps (so trajectories have length ``T+1``).
    obs_every: temporal stride between observations.
    obs_noise: std of the Gaussian observation noise.
    """

    truth: Float[Array, "T_plus_1 3"]
    obs: Float[Array, "T_plus_1 3"]
    mask: Float[Array, "T_plus_1 3"]
    prior_mean: Float[Array, "T_plus_1 3"]
    prior_mean_state: Float[Array, 3]
    B_op: lx.AbstractLinearOperator
    R_op: lx.AbstractLinearOperator
    B_op_state: lx.AbstractLinearOperator
    R_op_state: lx.AbstractLinearOperator
    dt: float
    T: int
    obs_every: int
    obs_noise: float

    @property
    def T_plus_1(self) -> int:
        return self.T + 1


class Lorenz63Forward(eqx.Module):
    """Lorenz-63 forward model satisfying ``pipekit_cycle.ForwardModel``.

    Wraps the diffrax-integrated `vardax._src.utils.dynamical_systems.Lorenz63`
    vector field as a one-step pipekit_cycle-compatible operator.
    Integration uses a single RK4 step of size `dt`, which is sufficient
    for the canonical `dt = 0.01` Lorenz-63 setup and keeps the model
    pure-jax (no diffrax solver state to thread).

    Attributes
    ----------
    dt: integration time-step.
    sigma, rho, beta: Lorenz-63 parameters.
    """

    dt: float
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0

    @property
    def state_signature(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((3,), jnp.float32)

    def _vector_field(self, x: Float[Array, 3]) -> Float[Array, 3]:
        # Re-use the canonical L63 vector field; ignore t (autonomous).
        l63 = Lorenz63(sigma=self.sigma, rho=self.rho, beta=self.beta)
        return l63(0.0, x, None)

    def step(
        self,
        state: Float[Array, 3],
        dt: float,
    ) -> Float[Array, 3]:
        """Advance the state by ``dt`` via one RK4 step.

        The signature matches ``pipekit_cycle.ForwardModel.step``.
        """
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

    # Burn-in.
    state, _ = jax.lax.scan(_scan_step, x0, None, length=burn_in)
    # Saved trajectory.
    _, traj = jax.lax.scan(_scan_step, state, None, length=T)
    return jnp.concatenate([state[None, :], traj], axis=0)


def generate_problem(
    *,
    key: PRNGKeyArray,
    T: int = 40,
    dt: float = 0.01,
    obs_every: int = 4,
    obs_noise: float = 1.0,
    prior_std: float = 5.0,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> LorenzProblem:
    """Generate the canonical Lorenz-63 partial-obs assimilation problem.

    Parameters
    ----------
    key: PRNG key for the truth's initial perturbation and obs noise.
    T: number of forecast steps. Trajectory length is ``T + 1``.
    dt: integration time-step. ``0.01`` is the canonical Lorenz-63 value.
    obs_every: observe one in every ``obs_every`` time steps.
    obs_noise: std of Gaussian observation noise on the x-component.
    prior_std: std used for the diagonal background covariance ``B``.
    sigma, rho, beta: Lorenz-63 parameters.

    Returns
    -------
    `LorenzProblem` with all the pieces every method needs.
    """
    k_truth, k_obs = jax.random.split(key)
    truth = _simulate_truth(
        k_truth, T=T, dt=dt, sigma=sigma, rho=rho, beta=beta
    ).astype(jnp.float32)

    # Mask: 1 at (time t multiple of obs_every, component x=0), else 0.
    T_plus_1 = T + 1
    mask = jnp.zeros((T_plus_1, 3), dtype=jnp.float32)
    obs_times = jnp.arange(0, T_plus_1, obs_every)
    mask = mask.at[obs_times, 0].set(1.0)

    # Noisy observations on the x-component; zero elsewhere.
    noise = obs_noise * jax.random.normal(k_obs, (T_plus_1, 3))
    obs = (truth + noise) * mask

    prior_mean = jnp.zeros((T_plus_1, 3), dtype=jnp.float32)
    prior_mean_state = jnp.zeros(3, dtype=jnp.float32)

    # B = prior_std^2 * I over the (T+1, 3) state.
    # R = obs_noise^2 * I over the (T+1, 3) obs.
    # `DiagonalLinearOperator` (rather than `IdentityLinearOperator * scalar`)
    # carries the positive-semidefinite tag through, which `lineax.CG`
    # inside `vardax.ThreeDVar` / `IncrementalFourDVar` requires. Users
    # can swap for structured operators (`gaussx` Matern, etc.) for
    # higher-fidelity comparisons.
    B_diag = jnp.full((T_plus_1, 3), prior_std**2, dtype=jnp.float32)
    R_diag = jnp.full((T_plus_1, 3), obs_noise**2, dtype=jnp.float32)
    state_diag = jnp.full((3,), prior_std**2, dtype=jnp.float32)
    state_R_diag = jnp.full((3,), obs_noise**2, dtype=jnp.float32)

    # Wrap with `TaggedLinearOperator(..., positive_semidefinite_tag)`:
    # `DiagonalLinearOperator` does not auto-carry the tag, but
    # `lineax.CG` (used inside `vardax.ThreeDVar` /
    # `IncrementalFourDVar`) refuses untagged operators.
    def _pd(op: lx.AbstractLinearOperator) -> lx.AbstractLinearOperator:
        return lx.TaggedLinearOperator(op, lx.positive_semidefinite_tag)

    B_op = _pd(lx.DiagonalLinearOperator(B_diag))
    R_op = _pd(lx.DiagonalLinearOperator(R_diag))
    B_op_state = _pd(lx.DiagonalLinearOperator(state_diag))
    R_op_state = _pd(lx.DiagonalLinearOperator(state_R_diag))

    return LorenzProblem(
        truth=truth,
        obs=obs,
        mask=mask,
        prior_mean=prior_mean,
        prior_mean_state=prior_mean_state,
        B_op=B_op,
        R_op=R_op,
        B_op_state=B_op_state,
        R_op_state=R_op_state,
        dt=dt,
        T=T,
        obs_every=obs_every,
        obs_noise=obs_noise,
    )
