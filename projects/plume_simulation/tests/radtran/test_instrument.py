"""JAX PSF + GSD operator tests.

Each operator must satisfy three invariants:

1. **Forward sanity** — apply produces the right output shape & sensible
   values on toy inputs.
2. **Linearity** — `apply(a·x + b·y) == a·apply(x) + b·apply(y)` to
   floating-point precision (operators are linear, so this should hold
   exactly in exact arithmetic).
3. **Adjoint identity** — `⟨A x, y⟩ = ⟨x, Aᵀ y⟩` to ~1e-10. This is the
   single most important invariant for variational data assimilation: if
   the adjoint is wrong, every gradient is wrong.

Plus: gradients must flow through `apply` via `jax.grad`, since the
3DVAR solver relies on autodiff through the obs operator chain.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from plume_simulation.radtran.instrument import (
    GroundSamplingDistance,
    PointSpreadFunction,
)


jax.config.update("jax_enable_x64", True)


# ── PSF ──────────────────────────────────────────────────────────────────────


class TestPointSpreadFunction:
    def test_gaussian_kernel_normalised(self):
        psf = PointSpreadFunction.gaussian(fwhm_pixels=2.0, kernel_size=9)
        assert psf.kernel.shape == (9, 9)
        np.testing.assert_allclose(psf.kernel.sum(), 1.0, atol=1e-12)

    def test_apply_preserves_shape_and_interior_constant(self):
        psf = PointSpreadFunction.gaussian(fwhm_pixels=1.5, kernel_size=7)
        x = jnp.ones((10, 10, 3))
        y = psf.apply(x)
        assert y.shape == x.shape
        # Zero-padding leaks at the edges; assert the interior (≥ kernel/2
        # away from the boundary) reproduces the constant exactly.
        np.testing.assert_allclose(np.asarray(y[3:-3, 3:-3, :]), 1.0, atol=1e-10)

    def test_linearity(self):
        rng = np.random.default_rng(0)
        psf = PointSpreadFunction.gaussian(fwhm_pixels=1.5, kernel_size=5)
        x = jnp.asarray(rng.standard_normal((6, 6, 2)))
        z = jnp.asarray(rng.standard_normal((6, 6, 2)))
        a, b = 0.7, -1.3
        lhs = psf.apply(a * x + b * z)
        rhs = a * psf.apply(x) + b * psf.apply(z)
        np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs), atol=1e-12)

    def test_adjoint_inner_product_identity(self):
        """⟨A x, u⟩ = ⟨x, Aᵀ u⟩ — the bedrock invariant for DA."""
        rng = np.random.default_rng(1)
        psf = PointSpreadFunction.gaussian(fwhm_pixels=1.5, kernel_size=5)
        x = jnp.asarray(rng.standard_normal((7, 7, 2)))
        u = jnp.asarray(rng.standard_normal((7, 7, 2)))
        Ax = psf.apply(x)
        Au = psf.adjoint(u)
        # Convert to numpy for stable inner products.
        lhs = float(jnp.sum(Ax * u))
        rhs = float(jnp.sum(x * Au))
        # Boundary discretisation gives ~1e-10 agreement; assert generously.
        assert abs(lhs - rhs) < 1e-9 * max(abs(lhs), abs(rhs), 1.0)

    def test_symmetric_kernel_is_self_adjoint(self):
        psf = PointSpreadFunction.gaussian(fwhm_pixels=1.5, kernel_size=5)
        np.testing.assert_allclose(psf.kernel, psf.kernel[::-1, ::-1], atol=1e-15)

    def test_jacobian_equals_apply(self):
        psf = PointSpreadFunction.gaussian(fwhm_pixels=1.5, kernel_size=5)
        x = jnp.asarray(np.random.default_rng(2).standard_normal((5, 5, 2)))
        np.testing.assert_allclose(np.asarray(psf.jacobian(x)), np.asarray(psf.apply(x)))

    def test_gradient_flows_through_apply(self):
        """jax.grad through apply == adjoint of `1`."""
        psf = PointSpreadFunction.gaussian(fwhm_pixels=1.5, kernel_size=5)

        def loss(x):
            return jnp.sum(psf.apply(x))

        x0 = jnp.zeros((5, 5, 2))
        g = jax.grad(loss)(x0)
        # ∂/∂x ⟨A x, 1⟩ = Aᵀ 1
        ones = jnp.ones((5, 5, 2))
        expected = psf.adjoint(ones)
        np.testing.assert_allclose(np.asarray(g), np.asarray(expected), atol=1e-9)

    def test_rejects_even_kernel_size(self):
        with pytest.raises(ValueError, match="odd"):
            PointSpreadFunction(kernel=np.ones((4, 4)))

    def test_rejects_non_square_kernel(self):
        with pytest.raises(ValueError, match="square"):
            PointSpreadFunction(kernel=np.ones((3, 5)))


# ── GSD ──────────────────────────────────────────────────────────────────────


class TestGroundSamplingDistance:
    def test_identity_when_factor_is_one(self):
        gsd = GroundSamplingDistance(downsample_factor=1)
        x = jnp.asarray(np.random.default_rng(3).standard_normal((4, 4, 2)))
        np.testing.assert_allclose(np.asarray(gsd.apply(x)), np.asarray(x))

    def test_block_mean(self):
        gsd = GroundSamplingDistance(downsample_factor=2)
        x = jnp.asarray(
            np.array(
                [[1.0, 1.0, 5.0, 5.0],
                 [1.0, 1.0, 5.0, 5.0],
                 [3.0, 3.0, 7.0, 7.0],
                 [3.0, 3.0, 7.0, 7.0]]
            )[..., None]
        )
        y = gsd.apply(x)
        assert y.shape == (2, 2, 1)
        np.testing.assert_allclose(np.asarray(y[..., 0]), [[1.0, 5.0], [3.0, 7.0]])

    def test_truncates_non_divisible_input(self):
        gsd = GroundSamplingDistance(downsample_factor=3)
        x = jnp.zeros((8, 8, 2))  # 8 // 3 = 2 → output 2×2×2
        assert gsd.apply(x).shape == (2, 2, 2)

    def test_adjoint_inner_product_identity(self):
        rng = np.random.default_rng(4)
        gsd = GroundSamplingDistance(downsample_factor=2)
        hr_shape = (6, 6, 2)
        x = jnp.asarray(rng.standard_normal(hr_shape))
        u = jnp.asarray(rng.standard_normal(gsd.lr_shape(hr_shape)))
        Ax = gsd.apply(x)
        Au = gsd.adjoint(u, hr_shape)
        lhs = float(jnp.sum(Ax * u))
        rhs = float(jnp.sum(x * Au))
        assert abs(lhs - rhs) < 1e-12 * max(abs(lhs), abs(rhs), 1.0)

    def test_adjoint_handles_truncation(self):
        rng = np.random.default_rng(5)
        gsd = GroundSamplingDistance(downsample_factor=3)
        hr_shape = (8, 8, 1)  # truncates to 6×6 in apply
        x = jnp.asarray(rng.standard_normal(hr_shape))
        u = jnp.asarray(rng.standard_normal(gsd.lr_shape(hr_shape)))
        Ax = gsd.apply(x)
        Au = gsd.adjoint(u, hr_shape)
        # Inner-product identity must hold even with truncation since the
        # truncated rows/cols carry zero weight in apply (and zero output in
        # adjoint).
        lhs = float(jnp.sum(Ax * u))
        rhs = float(jnp.sum(x * Au))
        assert abs(lhs - rhs) < 1e-12 * max(abs(lhs), abs(rhs), 1.0)
        assert Au.shape == hr_shape

    def test_from_optics_recovers_factor(self):
        # 0.35 mm sensor, 35 mm focal, 1000 px wide, 100 m altitude,
        # 0.5 m HR pixel → GSD = 1.0 m/px → factor = 2.
        gsd = GroundSamplingDistance.from_optics(
            sensor_width_mm=0.35,
            focal_length_mm=35.0,
            image_width_px=1000,
            altitude_m=100.0,
            pixel_size_hr_m=0.5,
        )
        assert gsd.downsample_factor == 2

    def test_gradient_flows_through_apply(self):
        gsd = GroundSamplingDistance(downsample_factor=2)
        hr_shape = (4, 4, 1)

        def loss(x):
            return jnp.sum(gsd.apply(x) ** 2)

        x0 = jnp.asarray(np.random.default_rng(6).standard_normal(hr_shape))
        g = jax.grad(loss)(x0)
        # ∂/∂x ‖A x‖² = 2 Aᵀ A x
        expected = 2.0 * gsd.adjoint(gsd.apply(x0), hr_shape)
        np.testing.assert_allclose(np.asarray(g), np.asarray(expected), atol=1e-12)

    def test_rejects_zero_factor(self):
        with pytest.raises(ValueError, match="downsample_factor"):
            GroundSamplingDistance(downsample_factor=0)

    def test_rejects_oversized_factor(self):
        gsd = GroundSamplingDistance(downsample_factor=10)
        with pytest.raises(ValueError, match="too small"):
            gsd.apply(jnp.zeros((4, 4, 1)))
