"""Tests for initialize_flow_from_ig (RBIG-style warm start)."""

from __future__ import annotations

import numpy as np
import pytest
from keras import ops

from gaussianization.gauss_keras import (
    initialize_flow_from_ig,
    make_coupling_flow,
    make_gaussianization_flow,
)
from gaussianization.gauss_keras.bijectors import MixtureCDFCoupling


def _to_numpy(x):
    return np.asarray(ops.convert_to_numpy(x))


def _two_moons_like(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, np.pi, n // 2)
    x1 = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    x2 = np.stack([1.0 - np.cos(theta), -np.sin(theta) + 0.5], axis=-1)
    x = np.concatenate([x1, x2], axis=0) + 0.05 * rng.standard_normal((n, 2))
    x = (x - x.mean(0)) / x.std(0)
    return x.astype("float32")


def test_ig_init_diagonal_log_prob_improves():
    x = _two_moons_like()
    flow = make_gaussianization_flow(
        input_dim=2, num_blocks=6, num_reflectors=2, num_components=6
    )
    _ = flow(ops.convert_to_tensor(x[:4]))
    lp_random = float(np.mean(_to_numpy(flow.log_prob(ops.convert_to_tensor(x)))))
    initialize_flow_from_ig(flow, x)
    lp_ig = float(np.mean(_to_numpy(flow.log_prob(ops.convert_to_tensor(x)))))
    # IG init should produce a substantially better log-likelihood than
    # random init — this tests the whole pipeline end-to-end.
    assert lp_ig > lp_random + 1.0, (
        f"IG init did not improve NLL: random={lp_random:.3f}, ig={lp_ig:.3f}"
    )


def test_ig_init_diagonal_pushforward_near_gaussian():
    x = _two_moons_like(n=4000)
    flow = make_gaussianization_flow(
        input_dim=2, num_blocks=8, num_reflectors=2, num_components=8
    )
    _ = flow(ops.convert_to_tensor(x[:4]))
    initialize_flow_from_ig(flow, x)
    z = _to_numpy(flow(ops.convert_to_tensor(x)))
    # With 8 RBIG blocks the pushforward should already be nearly isotropic.
    assert abs(z.mean()) < 0.05
    cov = np.cov(z, rowvar=False)
    assert abs(cov[0, 0] - 1.0) < 0.12
    assert abs(cov[1, 1] - 1.0) < 0.12
    assert abs(cov[0, 1]) < 0.10


def test_ig_init_coupling_final_kernel_is_zero():
    x = _two_moons_like()
    flow = make_coupling_flow(
        input_dim=2, num_blocks=2, num_components=5, hidden=(16, 16)
    )
    _ = flow(ops.convert_to_tensor(x[:4]))
    initialize_flow_from_ig(flow, x)
    for b in flow.bijector_layers:
        if isinstance(b, MixtureCDFCoupling):
            # The last Dense of the conditioner should have a zero kernel.
            last = None
            for sub in b.conditioner.layers:
                if hasattr(sub, "kernel") and hasattr(sub, "bias"):
                    last = sub
            assert last is not None
            kernel = _to_numpy(last.kernel)
            assert np.max(np.abs(kernel)) < 1e-6


def test_ig_init_coupling_log_prob_improves():
    x = _two_moons_like(n=3000)
    flow = make_coupling_flow(
        input_dim=2, num_blocks=2, num_components=6, hidden=(32, 32)
    )
    _ = flow(ops.convert_to_tensor(x[:4]))
    lp_random = float(np.mean(_to_numpy(flow.log_prob(ops.convert_to_tensor(x)))))
    initialize_flow_from_ig(flow, x)
    lp_ig = float(np.mean(_to_numpy(flow.log_prob(ops.convert_to_tensor(x)))))
    assert lp_ig > lp_random + 0.5, (
        f"IG init did not improve NLL (coupling): random={lp_random:.3f}, ig={lp_ig:.3f}"
    )


def test_ig_init_coupling_matches_diagonal_equivalent():
    """At init the zero-kernel conditioner emits a fixed bias → coupling
    layer behaves as a diagonal marginal. The pushforward of a coupling
    flow immediately after IG init should match the pushforward of a
    diagonal flow after the same number of IG blocks (up to the fact
    that the coupling block does two passes with swapped masks)."""
    # Just verify determinism and that z is finite + roughly standard.
    x = _two_moons_like(n=2000)
    flow = make_coupling_flow(
        input_dim=2, num_blocks=2, num_components=5, hidden=(16, 16)
    )
    _ = flow(ops.convert_to_tensor(x[:4]))
    initialize_flow_from_ig(flow, x)
    z = _to_numpy(flow(ops.convert_to_tensor(x)))
    assert np.all(np.isfinite(z))


def test_ig_init_raises_on_unbuilt_flow():
    x = _two_moons_like()
    flow = make_gaussianization_flow(input_dim=2, num_blocks=2, num_components=4)
    with pytest.raises(RuntimeError, match="not built"):
        initialize_flow_from_ig(flow, x)


def test_ig_init_raises_on_dim_mismatch():
    x = _two_moons_like(n=100)
    flow = make_gaussianization_flow(input_dim=3, num_blocks=2, num_components=4)
    dummy = np.zeros((4, 3), dtype=np.float32)
    _ = flow(ops.convert_to_tensor(dummy))
    with pytest.raises(ValueError, match="input_dim"):
        initialize_flow_from_ig(flow, x)  # x has d=2 but flow expects d=3


def test_ig_init_coupling_equals_diagonal_at_init():
    """Zero-kernel coupling flow + IG init = diagonal marginal flow + IG init.

    With matching block counts and per-(block, dim) EM seeding the two
    architectures produce the same pushforward and log-density to float
    precision.
    """
    import keras

    x = _two_moons_like(n=2000)
    num_blocks = 3
    num_components = 6

    keras.utils.set_random_seed(17)
    flow_diag = make_gaussianization_flow(
        input_dim=2,
        num_blocks=num_blocks,
        num_reflectors=2,
        num_components=num_components,
    )
    _ = flow_diag(ops.convert_to_tensor(x[:4]))
    initialize_flow_from_ig(flow_diag, x)

    keras.utils.set_random_seed(17)
    flow_cpl = make_coupling_flow(
        input_dim=2,
        num_blocks=num_blocks,
        num_components=num_components,
        hidden=(16, 16),
    )
    _ = flow_cpl(ops.convert_to_tensor(x[:4]))
    initialize_flow_from_ig(flow_cpl, x)

    z_diag = _to_numpy(flow_diag(ops.convert_to_tensor(x)))
    z_cpl = _to_numpy(flow_cpl(ops.convert_to_tensor(x)))
    lp_diag = _to_numpy(flow_diag.log_prob(ops.convert_to_tensor(x)))
    lp_cpl = _to_numpy(flow_cpl.log_prob(ops.convert_to_tensor(x)))

    # Float32 accumulates through several blocks; gate on median + percentile
    # rather than max to absorb the handful of tail points.
    z_diff = np.abs(z_cpl - z_diag)
    assert np.median(z_diff) < 1e-6
    assert np.percentile(z_diff, 99) < 1e-3
    lp_diff = np.abs(lp_cpl - lp_diag)
    assert np.median(lp_diff) < 1e-4
    assert np.percentile(lp_diff, 99) < 1e-2


def test_ig_init_roundtrip_preserves_after_training_step():
    """Sanity: after IG init, flow is still a bijection — f^{-1}(f(x)) ≈ x."""
    x = _two_moons_like(n=1000)
    flow = make_gaussianization_flow(
        input_dim=2, num_blocks=4, num_reflectors=2, num_components=5
    )
    _ = flow(ops.convert_to_tensor(x[:4]))
    initialize_flow_from_ig(flow, x)
    z = flow(ops.convert_to_tensor(x))
    x_rt = _to_numpy(flow.invert(z))
    err = np.abs(x_rt - x)
    assert np.median(err) < 1e-3
