"""Regression tests for the two-level Lorenz-96 vector field.

PR #73 review flagged a bug where the slow-variable back-reaction
term was computed as ``-h*c*Y_mean`` instead of the documented
``-(h*c/b) * sum_j Y[j,k]``. These tests pin down the coupling
coefficients and the (K, J) state layout against hand-computed
values so the regression can't sneak back.
"""

from __future__ import annotations

import jax.numpy as jnp
from assimilation import Lorenz96TwoLevelForward, Lorenz96TwoLevelVF


def _vf(
    *,
    K: int = 2,
    J: int = 3,
    F: float = 5.0,
    h: float = 1.0,
    c: float = 10.0,
    b: float = 10.0,
) -> Lorenz96TwoLevelVF:
    return Lorenz96TwoLevelVF(K=K, J=J, F=F, h=h, c=c, b=b)


def test_zero_state_gives_pure_forcing_on_slow():
    """At ``state = 0`` the only nonzero slow tendency is ``F``."""
    vf = _vf(K=2, J=3, F=5.0)
    state = jnp.zeros(2 + 3 * 2)
    dot = vf(state)
    # Slow: dX_k = 0*(0-0) - 0 + F - 0 = F.
    assert jnp.allclose(dot[:2], jnp.array([5.0, 5.0]))
    # Fast: dY = 0 - 0 + 0 (X = 0).
    assert jnp.allclose(dot[2:], 0.0)


def test_slow_back_reaction_uses_block_sum_over_b():
    """Slow tendency back-reaction is ``-(h*c/b) * sum_j Y[k, j]``.

    Build a state with X=0, F=0, all fast entries equal to ``y0``
    so the quadratic L96 terms vanish on the slow side and the
    only nonzero contribution is the back-reaction.
    """
    K, J = 2, 4
    h, c, b = 1.0, 10.0, 5.0  # h*c/b = 2.0
    y0 = 0.5
    vf = _vf(K=K, J=J, F=0.0, h=h, c=c, b=b)

    state = jnp.zeros(K + J * K).at[K:].set(y0)
    dot = vf(state)

    # Block sum is J * y0 = 2.0; back-reaction is -(h*c/b)*J*y0 = -4.0.
    expected_slow = -h * c / b * J * y0
    assert jnp.allclose(dot[:K], expected_slow, atol=1e-6), (
        f"slow back-reaction wrong: got {dot[:K]}, expected {expected_slow}"
    )


def test_block_layout_is_K_blocks_of_J():
    """Each slow X_k couples to the contiguous block Y[k, :], NOT to
    every J-th element of Y.

    Probe: set Y_{k=0, j} = 1 for all j, Y_{k>0, *} = 0. The back-
    reaction should only fire on slow index 0.
    """
    K, J = 3, 4
    h, c, b = 1.0, 10.0, 10.0
    vf = _vf(K=K, J=J, F=0.0, h=h, c=c, b=b)

    Y = jnp.zeros((K, J)).at[0].set(1.0).reshape(K * J)
    state = jnp.concatenate([jnp.zeros(K), Y])
    dot = vf(state)
    slow_dot = dot[:K]

    # Only X_0 back-reaction should be nonzero: -(h*c/b)*J*1.
    assert jnp.isclose(slow_dot[0], -h * c / b * J), f"slow[0] wrong: {slow_dot[0]}"
    assert jnp.allclose(slow_dot[1:], 0.0, atol=1e-6), (
        f"slow[1:] should be zero, got {slow_dot[1:]}"
    )


def test_fast_forcing_repeats_X_across_block():
    """Fast forcing source for Y_{Jk + j} is X_k (same for all j in
    that block)."""
    K, J = 3, 2
    h, c, b = 1.0, 10.0, 10.0
    vf = _vf(K=K, J=J, F=0.0, h=h, c=c, b=b)

    X = jnp.array([1.0, 2.0, 3.0])
    state = jnp.concatenate([X, jnp.zeros(J * K)])
    dot = vf(state)

    # Fast tendency for zero Y: only the (h*c/b) * X_repeat term
    # survives (advection and damping are zero at Y=0).
    fast_dot = dot[K:].reshape(K, J)
    expected = (h * c / b) * X
    for k in range(K):
        assert jnp.allclose(fast_dot[k], expected[k]), (
            f"block k={k} forcing mismatch: {fast_dot[k]} vs {expected[k]}"
        )


def test_forward_roundtrip_zero_error():
    """Re-integrating the truth's initial state with the same forward
    must reproduce the truth bit-for-bit."""
    import jax
    from assimilation import generate_l96_2l_problem

    prob = generate_l96_2l_problem(key=jax.random.PRNGKey(0), T_assim=10, T_total=10)
    fwd = Lorenz96TwoLevelForward(
        K=prob.K,
        J=prob.J,
        F=prob.F,
        h=prob.h,
        c=prob.c,
        b=prob.b,
        dt=prob.dt,
    )

    def step(s, _):
        new = fwd.step(s, fwd.dt)
        return new, new

    _, traj = jax.lax.scan(step, prob.truth[0], None, length=prob.T_total)
    rt = jnp.concatenate([prob.truth[0][None, :], traj], axis=0)
    assert float(jnp.max(jnp.abs(rt - prob.truth))) < 1e-5
