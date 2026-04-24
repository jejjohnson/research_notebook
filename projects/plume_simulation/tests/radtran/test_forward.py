"""Forward models: nonlinear, Maclaurin, Taylor (+ normalised variants)."""

from __future__ import annotations

import numpy as np
import pytest
from plume_simulation.radtran.forward import (
    forward_maclaurin,
    forward_maclaurin_normalized,
    forward_nonlinear,
    forward_nonlinear_normalized,
    forward_taylor,
    forward_taylor_normalized,
)


NU_OBS = np.linspace(4100.0, 4400.0, 41)


def test_forward_nonlinear_returns_nonneg_transmittance(synthetic_lut):
    result = forward_nonlinear(
        synthetic_lut, NU_OBS,
        T_K=280.0, p_atm=1.0, vmr=2e-6,
        path_length_cm=8.4e5, amf=2.0,
    )
    assert (result.transmittance >= 0).all()
    assert (result.transmittance <= 1.0 + 1e-12).all()
    # Radiance shape matches nu_obs.
    assert result.radiance.shape == NU_OBS.shape


def test_nonlinear_decreases_with_vmr(synthetic_lut):
    # Pick the line centre where absorption is strongest.
    nu_line = np.array([4300.0])
    r_small = forward_nonlinear(
        synthetic_lut, nu_line,
        T_K=280.0, p_atm=1.0, vmr=1e-7,
        path_length_cm=8.4e5, amf=2.0,
    )
    r_large = forward_nonlinear(
        synthetic_lut, nu_line,
        T_K=280.0, p_atm=1.0, vmr=5e-6,
        path_length_cm=8.4e5, amf=2.0,
    )
    assert float(r_large.radiance.item()) < float(r_small.radiance.item())


def test_maclaurin_order1_matches_nonlinear_in_thin_limit(synthetic_lut):
    kwargs = dict(
        T_K=280.0, p_atm=1.0, vmr=1e-8,
        path_length_cm=8.4e5, amf=2.0,
    )
    ref = forward_nonlinear(synthetic_lut, NU_OBS, **kwargs)
    lin = forward_maclaurin(synthetic_lut, NU_OBS, order=1, **kwargs)
    np.testing.assert_allclose(lin.radiance, ref.radiance, rtol=5e-3)


def test_maclaurin_convergence_with_order(synthetic_lut):
    # At a larger VMR, order 2 is closer to exact than order 1.
    kwargs = dict(
        T_K=280.0, p_atm=1.0, vmr=5e-6,
        path_length_cm=8.4e5, amf=2.0,
    )
    ref = forward_nonlinear(synthetic_lut, NU_OBS, **kwargs)
    lin1 = forward_maclaurin(synthetic_lut, NU_OBS, order=1, **kwargs)
    lin2 = forward_maclaurin(synthetic_lut, NU_OBS, order=2, **kwargs)
    err1 = np.linalg.norm(lin1.radiance - ref.radiance)
    err2 = np.linalg.norm(lin2.radiance - ref.radiance)
    assert err2 < err1


def test_taylor_exact_at_background(synthetic_lut):
    kwargs = dict(
        T_K=280.0, p_atm=1.0,
        path_length_cm=8.4e5, amf=2.0,
    )
    vmr_b = 2e-6
    tay = forward_taylor(synthetic_lut, NU_OBS, vmr=vmr_b, vmr_background=vmr_b, **kwargs)
    exact = forward_nonlinear(synthetic_lut, NU_OBS, vmr=vmr_b, **kwargs)
    # Taylor at the linearisation point equals the exact forward model.
    np.testing.assert_allclose(tay.radiance, exact.radiance, rtol=1e-10)


def test_nonlinear_normalized_is_unity_at_zero(synthetic_lut):
    kwargs = dict(T_K=280.0, p_atm=1.0, path_length_cm=8.4e5, amf=2.0)
    r = forward_nonlinear_normalized(
        synthetic_lut, NU_OBS, vmr_background=2e-6, delta_vmr=0.0, **kwargs,
    )
    np.testing.assert_allclose(r.radiance, 1.0, atol=1e-12)


def test_normalised_nonlinear_below_one_for_positive_delta(synthetic_lut):
    r = forward_nonlinear_normalized(
        synthetic_lut, NU_OBS,
        T_K=280.0, p_atm=1.0,
        vmr_background=2e-6, delta_vmr=1e-6,
        path_length_cm=8.4e5, amf=2.0,
    )
    # Inside the absorption bands the normalised transmittance dips below 1.
    assert r.radiance.min() < 1.0
    assert r.radiance.max() <= 1.0 + 1e-12


def test_maclaurin_normalized_and_taylor_normalized_agree_at_order_1(synthetic_lut):
    kwargs = dict(
        T_K=280.0, p_atm=1.0, delta_vmr=5e-7,
        path_length_cm=8.4e5, amf=2.0,
    )
    mac = forward_maclaurin_normalized(synthetic_lut, NU_OBS, order=1, **kwargs)
    tay = forward_taylor_normalized(synthetic_lut, NU_OBS, **kwargs)
    np.testing.assert_allclose(mac.radiance, tay.radiance, atol=1e-14)


def test_jacobian_matches_finite_difference(synthetic_lut):
    kwargs = dict(T_K=280.0, p_atm=1.0, path_length_cm=8.4e5, amf=2.0)
    h = 1e-10
    r_plus = forward_nonlinear(synthetic_lut, NU_OBS, vmr=2e-6 + h, **kwargs)
    r_minus = forward_nonlinear(synthetic_lut, NU_OBS, vmr=2e-6 - h, **kwargs)
    fd = (r_plus.radiance - r_minus.radiance) / (2 * h)
    r = forward_nonlinear(synthetic_lut, NU_OBS, vmr=2e-6, **kwargs)
    np.testing.assert_allclose(r.jacobian, fd, rtol=1e-4, atol=1e-5)


def test_forward_rejects_bad_order():
    with pytest.raises(ValueError, match="order"):
        forward_maclaurin(
            None, NU_OBS,  # type: ignore[arg-type]
            T_K=280.0, p_atm=1.0, vmr=2e-6,
            path_length_cm=8.4e5, amf=2.0, order=4,
        )


def test_forward_rejects_missing_var(synthetic_lut):
    with pytest.raises(KeyError, match="not in dataset"):
        forward_nonlinear(
            synthetic_lut, NU_OBS,
            T_K=280.0, p_atm=1.0, vmr=2e-6,
            path_length_cm=8.4e5, amf=2.0, var="missing",
        )
