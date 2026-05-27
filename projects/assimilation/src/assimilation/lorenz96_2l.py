"""Lorenz-96 two-level (multi-scale) model + benchmark problem.

The two-level Lorenz-96 system (Wilks 2005, Arnold et al. 2013) is
the canonical sub-grid testbed in data assimilation. Each slow
variable $X_k$ couples to $J$ fast variables $Y_{j,k}$:

.. math::

    \\dot{X}_k = X_{k-1}(X_{k+1} - X_{k-2}) - X_k + F -
                \\frac{h c}{b} \\sum_{j=0}^{J-1} Y_{j,k},

    \\dot{Y}_{j,k} = -c b\\, Y_{j+1,k}\\,(Y_{j+2,k} - Y_{j-1,k}) -
                    c\\, Y_{j,k} + \\frac{h c}{b}\\, X_k.

The fast variables live on a single $JK$-long periodic ring; each
slow $X_k$ couples to the contiguous block $Y_{Jk}, \\ldots, Y_{Jk+J-1}$
via a back-reaction term (the mean of its $J$ fast neighbours).
With canonical parameters $h = 1, c = 10, b = 10, F = 20$ the system
is fully chaotic in both regimes — the fast scale is ten times
faster and ten times smaller in amplitude than the slow one.

For the assimilation benchmark we **observe only the slow $X_k$**,
sparsely in space and time, and ask each method to recover the full
$(X, Y)$ state. This is the canonical "unresolved sub-grid"
inversion problem.

State layout
------------
The flat state vector has length ``K + J * K``: the first ``K``
entries are the slow variables, the remaining ``J * K`` are the fast
variables in row-major order with the slow index varying slowest.

.. code-block:: python

    X = state[:K]
    Y = state[K:].reshape(J, K)  # Y[j, k]
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float, PRNGKeyArray


@dataclass(frozen=True)
class LorenzL96TwoLevelProblem:
    """Shared two-level L96 benchmark problem.

    Attributes
    ----------
    truth: ``(T+1, K + J*K)`` flattened ground-truth trajectory.
    obs: ``(T+1, K + J*K)`` observations, zero at masked entries.
        Only the slow-variable block (first ``K`` columns) carries
        observations by default; the fast block is fully masked.
    mask: ``(T+1, K + J*K)`` binary mask.
    prior_mean, prior_mean_state: zero priors over the full flat
        state, in trajectory and ``(K + J*K,)`` shape respectively.
    B_op, R_op, B_op_state, R_op_state: diagonal covariances over the
        trajectory and the flat state.
    K, J: slow and fast dimensions.
    F, h, c, b: Lorenz-96-2L parameters.
    dt, T, obs_every_space, obs_every_time, obs_noise: setup knobs.
    """

    truth: Float[Array, "T_plus_1 D"]
    obs: Float[Array, "T_plus_1 D"]
    mask: Float[Array, "T_plus_1 D"]
    prior_mean: Float[Array, "T_plus_1 D"]
    prior_mean_state: Float[Array, " D"]
    B_op: lx.AbstractLinearOperator
    R_op: lx.AbstractLinearOperator
    B_op_state: lx.AbstractLinearOperator
    R_op_state: lx.AbstractLinearOperator
    K: int
    J: int
    F: float
    h: float
    c: float
    b: float
    dt: float
    T: int
    obs_every_space: int
    obs_every_time: int
    obs_noise: float

    @property
    def D(self) -> int:
        """Flat state dimension ``K + J * K``."""
        return self.K + self.J * self.K

    @property
    def T_plus_1(self) -> int:
        return self.T + 1


class Lorenz96TwoLevelVF(eqx.Module):
    """Two-level Lorenz-96 vector field.

    Pure-jax callable returning ``\\dot{state}`` from ``state``. The
    forward integrator (`Lorenz96TwoLevelForward`) wraps this in an
    RK4 step.
    """

    K: int = eqx.field(static=True)
    J: int = eqx.field(static=True)
    F: float
    h: float
    c: float
    b: float

    def __call__(self, state: Float[Array, " D"]) -> Float[Array, " D"]:
        K, J = self.K, self.J
        X = state[:K]
        Y = state[K:].reshape(J * K)

        # Slow back-reaction: mean of the J fast variables in each
        # slow block.
        Y_blocks = Y.reshape(K, J)  # (K, J)
        Y_mean_per_slow = Y_blocks.mean(axis=1)  # (K,)

        # Slow tendencies: same L96-1L structure, plus back-reaction.
        dX = (
            jnp.roll(X, 1) * (jnp.roll(X, -1) - jnp.roll(X, 2))
            - X
            + self.F
            - self.h * self.c * Y_mean_per_slow  # (h c / b) * J * Y_mean
        )

        # Fast tendencies: cyclic shifts on the full JK-long ring.
        # Coupling source for each Y_{Jk + j} is X_k.
        X_repeat = jnp.repeat(X, J)  # (J*K,) — each X_k repeated J times
        dY = (
            -self.c * self.b * jnp.roll(Y, -1) * (jnp.roll(Y, -2) - jnp.roll(Y, 1))
            - self.c * Y
            + (self.h * self.c / self.b) * X_repeat
        )

        return jnp.concatenate([dX, dY])


class Lorenz96TwoLevelForward(eqx.Module):
    """Two-level Lorenz-96 forward model (pipekit_cycle.ForwardModel).

    One RK4 step of size ``dt`` on the full flat state. The fast scale
    is ``c`` times faster than the slow scale, so ``dt`` needs to be
    significantly smaller than for single-level L96: the default
    ``dt = 0.005`` is stable for the canonical ``c = 10`` regime.

    Attributes
    ----------
    K, J: slow and fast dimensions.
    F, h, c, b: Lorenz-96-2L parameters.
    dt: integration time-step.
    """

    K: int = eqx.field(static=True)
    J: int = eqx.field(static=True)
    F: float
    h: float
    c: float
    b: float
    dt: float

    @property
    def state_signature(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self.K + self.J * self.K,), jnp.float32)

    def _vf(self, x: Float[Array, " D"]) -> Float[Array, " D"]:
        return Lorenz96TwoLevelVF(
            K=self.K, J=self.J, F=self.F, h=self.h, c=self.c, b=self.b
        )(x)

    def step(
        self,
        state: Float[Array, " D"],
        dt: float,
    ) -> Float[Array, " D"]:
        """Advance the flat state by ``dt`` via one RK4 step."""
        k1 = self._vf(state)
        k2 = self._vf(state + 0.5 * dt * k1)
        k3 = self._vf(state + 0.5 * dt * k2)
        k4 = self._vf(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _simulate_l96_2l_truth(
    key: PRNGKeyArray,
    *,
    K: int,
    J: int,
    F: float,
    h: float,
    c: float,
    b: float,
    T: int,
    dt: float,
    burn_in: int = 2000,
) -> Float[Array, "T_plus_1 D"]:
    """Integrate the two-level L96 system from a perturbed equilibrium.

    Slow variables start at ``F``, fast variables at zero; a small
    Gaussian perturbation kicks off the chaotic regime. We integrate
    ``burn_in`` extra steps so the saved trajectory sits on the
    attractor.
    """
    fwd = Lorenz96TwoLevelForward(K=K, J=J, F=F, h=h, c=c, b=b, dt=dt)
    D = K + J * K
    x0 = jnp.concatenate(
        [F * jnp.ones(K, dtype=jnp.float32), jnp.zeros(J * K, dtype=jnp.float32)]
    ) + 0.05 * jax.random.normal(key, (D,))

    def _scan_step(state, _):
        new = fwd.step(state, dt)
        return new, new

    state, _ = jax.lax.scan(_scan_step, x0, None, length=burn_in)
    _, traj = jax.lax.scan(_scan_step, state, None, length=T)
    return jnp.concatenate([state[None, :], traj], axis=0)


def generate_l96_2l_problem(
    *,
    key: PRNGKeyArray,
    K: int = 8,
    J: int = 8,
    F: float = 20.0,
    h: float = 1.0,
    c: float = 10.0,
    b: float = 10.0,
    T: int = 40,
    dt: float = 0.005,
    obs_every_space: int = 2,
    obs_every_time: int = 4,
    obs_noise: float = 0.5,
    prior_std: float = 5.0,
) -> LorenzL96TwoLevelProblem:
    """Generate the canonical two-level Lorenz-96 benchmark problem.

    Defaults follow Wilks 2005's strongly-chaotic regime
    (``h = 1, c = 10, b = 10, F = 20``) scaled down to a tractable
    ``K = 8`` slow / ``J = 8`` fast geometry (state dim ``D = 72``).

    Observation pattern: only the **slow** ``X_k`` variables are
    observed, sparsely in space and time:

    - every ``obs_every_space``-th slow grid point (default 2 → 4 of 8)
    - every ``obs_every_time``-th time step (default 4 → 11 of 41)
    - additive Gaussian noise with std ``obs_noise``

    Total observed scalars (defaults): ``4 * 11 = 44``. State has
    ``41 * 72 = 2952`` entries — heavily under-determined. The
    *fast* variables receive no direct observations; each method's
    ability to reconstruct them tests how much sub-grid information
    leaks through the slow-fast coupling.

    Parameters
    ----------
    K, J: slow and fast dimensions. Total state dim is ``K + J*K``.
    F, h, c, b: Lorenz-96-2L parameters. The defaults are the
        strongly-coupled, strongly-chaotic regime; ``F = 10`` gives
        a weakly-chaotic alternative.
    T, dt: trajectory length and integration step.
    obs_every_space, obs_every_time: slow-variable obs stride.
    obs_noise: std of Gaussian observation noise.
    prior_std: diagonal background covariance std for both slow and
        fast variables.
    """
    k_truth, k_obs = jax.random.split(key)
    truth = _simulate_l96_2l_truth(
        k_truth, K=K, J=J, F=F, h=h, c=c, b=b, T=T, dt=dt
    ).astype(jnp.float32)

    D = K + J * K
    T_plus_1 = T + 1

    # Mask covers only the slow block (first K columns) and only at
    # obs_every_time-spaced time slices.
    mask = jnp.zeros((T_plus_1, D), dtype=jnp.float32)
    space_idx_slow = jnp.arange(0, K, obs_every_space)
    time_idx = jnp.arange(0, T_plus_1, obs_every_time)
    mask = mask.at[time_idx[:, None], space_idx_slow[None, :]].set(1.0)

    noise = obs_noise * jax.random.normal(k_obs, (T_plus_1, D))
    obs = (truth + noise) * mask

    prior_mean = jnp.zeros((T_plus_1, D), dtype=jnp.float32)
    prior_mean_state = jnp.zeros(D, dtype=jnp.float32)

    B_diag = jnp.full((T_plus_1, D), prior_std**2, dtype=jnp.float32)
    R_diag = jnp.full((T_plus_1, D), obs_noise**2, dtype=jnp.float32)
    state_B_diag = jnp.full((D,), prior_std**2, dtype=jnp.float32)
    state_R_diag = jnp.full((D,), obs_noise**2, dtype=jnp.float32)

    def _pd(op: lx.AbstractLinearOperator) -> lx.AbstractLinearOperator:
        return lx.TaggedLinearOperator(op, lx.positive_semidefinite_tag)

    return LorenzL96TwoLevelProblem(
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
        J=J,
        F=F,
        h=h,
        c=c,
        b=b,
        dt=dt,
        T=T,
        obs_every_space=obs_every_space,
        obs_every_time=obs_every_time,
        obs_noise=obs_noise,
    )
