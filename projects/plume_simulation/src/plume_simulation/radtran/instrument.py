"""JAX point-spread + ground-sampling instrument operators.

These two operators model the **deterministic** part of an imaging-spectrometer
forward chain that sits between the radiative-transfer output and the
band-integration step:

    HR radiance cube  ──PSF──►  blurred HR cube  ──GSD──►  LR cube  ──SRF──►  bands

The :class:`~plume_simulation.radtran.srf.SpectralResponseFunction` already
covers the SRF stage; this module fills in the PSF (spatial blur) and GSD
(detector down-sampling). Both operators here are:

- **JAX-native** — every method is `jit`-able and `vmap`-able; gradients flow
  through ``apply``/``jacobian``/``adjoint`` via reverse-mode AD without
  manual bookkeeping. This is the prerequisite for the variational retrieval
  module :mod:`plume_simulation.assimilation`.
- **Linear** — `jacobian == apply` (state-independent), and `adjoint`
  implements the Euclidean transpose so the inner-product identity
  ``⟨A x, y⟩ = ⟨x, Aᵀ y⟩`` holds to machine precision.
- **Cube-shaped (band-last)** — inputs are ``(ny, nx, n_lambda)``; channels
  are processed independently. This matches the convention in
  :mod:`plume_simulation.radtran.srf` and plays nicely with `jax.vmap`
  along the spectral axis.

The numpy reference for the PSF/GSD logic lives in
``jej_vc_snippets/methane_retrieval/lut_obs_op.py``; this is the JAX rewrite
that lets the same operators participate in autodiff-based 3D-Var solves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


PSFType = Literal["gaussian", "airy"]
HRCube = Float[Array, "ny nx n_lambda"]
LRCube = Float[Array, "ny_lr nx_lr n_lambda"]


# ── Point Spread Function ────────────────────────────────────────────────────


@dataclass(frozen=True, eq=False)
class PointSpreadFunction:
    """2D PSF blur applied independently per spectral channel.

    The PSF is a fixed convolution kernel (``kernel_size × kernel_size``,
    L1-normalised) so the operator is linear and shift-invariant. The
    Jacobian is the operator itself; the adjoint is convolution with the
    180°-flipped kernel (real-valued case).

    The class is ``frozen=True, eq=False`` so the default
    ``object.__eq__`` (identity comparison) and ``object.__hash__``
    (``id``-based) are retained. An auto-generated dataclass ``__eq__``
    would do *element-wise* equality on the ``kernel: np.ndarray`` field
    — ambiguous in bool context — and the corresponding ``__hash__``
    would crash because numpy arrays are unhashable. Identity semantics
    are what ``jax.jit`` static-arg caching actually wants: a PSF
    instance is re-used as-is or replaced wholesale; it never needs
    structural equality.

    Attributes
    ----------
    kernel : np.ndarray
        L1-normalised 2-D kernel, shape ``(kernel_size, kernel_size)``.
        Stored as numpy for transparent inspection; the JAX copies used by
        :meth:`apply` / :meth:`adjoint` are cached as private fields below.
    """

    kernel: np.ndarray
    _kernel_jax: Array = field(init=False, repr=False, compare=False)
    _kernel_adj: Array = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        k = np.asarray(self.kernel, dtype=np.float64)
        if k.ndim != 2:
            raise ValueError(f"PointSpreadFunction: kernel must be 2-D, got shape {k.shape}.")
        if k.shape[0] != k.shape[1]:
            raise ValueError(f"PointSpreadFunction: kernel must be square, got {k.shape}.")
        if k.shape[0] % 2 != 1 or k.shape[0] < 3:
            raise ValueError(
                f"PointSpreadFunction: kernel side length must be odd ≥ 3 (got {k.shape[0]})."
            )
        s = float(k.sum())
        if s == 0.0:
            raise ValueError("PointSpreadFunction: kernel sums to zero; cannot normalise.")
        # Frozen dataclass — bypass __setattr__ for derived fields.
        normalised = k / s
        object.__setattr__(self, "kernel", normalised)
        object.__setattr__(self, "_kernel_jax", jnp.asarray(normalised))
        object.__setattr__(self, "_kernel_adj", jnp.asarray(normalised[::-1, ::-1]))

    @classmethod
    def gaussian(
        cls, fwhm_pixels: float, kernel_size: int = 9
    ) -> "PointSpreadFunction":
        """Build an isotropic Gaussian PSF.

        ``FWHM = 2.355 σ``; the kernel is sampled on the discrete pixel grid
        and L1-normalised in ``__post_init__``.
        """
        if fwhm_pixels <= 0.0:
            raise ValueError("PointSpreadFunction.gaussian: `fwhm_pixels` must be > 0.")
        c = kernel_size // 2
        y, x = np.ogrid[-c : c + 1, -c : c + 1]
        r2 = x * x + y * y
        sigma = fwhm_pixels / 2.355
        k = np.exp(-r2 / (2.0 * sigma * sigma))
        return cls(kernel=k)

    @classmethod
    def airy(cls, fwhm_pixels: float, kernel_size: int = 9) -> "PointSpreadFunction":
        """Build an Airy-disk-like PSF as a diffraction-ish surrogate."""
        from scipy.special import j1

        if fwhm_pixels <= 0.0:
            raise ValueError("PointSpreadFunction.airy: `fwhm_pixels` must be > 0.")
        c = kernel_size // 2
        y, x = np.ogrid[-c : c + 1, -c : c + 1]
        r = np.sqrt(x * x + y * y)
        xarg = np.where(r == 0.0, 1e-12, 2.0 * np.pi * r / (1.22 * fwhm_pixels))
        k = (2.0 * j1(xarg) / xarg) ** 2
        return cls(kernel=k)

    # ── linear operator ──────────────────────────────────────────────────────

    def apply(self, radiance: HRCube) -> HRCube:
        """Forward blur: ``y = K * x`` per channel."""
        x = jnp.asarray(radiance)
        if x.ndim != 3:
            raise ValueError(
                f"PointSpreadFunction.apply: input must be 3-D (ny, nx, n_lambda), "
                f"got shape {x.shape}."
            )
        return _convolve_channels(x, self._kernel_jax)

    def jacobian(self, tangent: HRCube) -> HRCube:
        """JVP — same as :meth:`apply` because the operator is linear."""
        return self.apply(tangent)

    def adjoint(self, cotangent: HRCube) -> HRCube:
        """Adjoint: convolution with the flipped kernel.

        For a real-valued L1-normalised kernel ``K`` the discrete adjoint of
        the per-channel cross-correlation `y = K ⋆ x` is `Kᵀ ⋆ u = K[::-1] ⋆ u`.
        Symmetric kernels (e.g. Gaussian) are self-adjoint.
        """
        u = jnp.asarray(cotangent)
        if u.ndim != 3:
            raise ValueError(
                f"PointSpreadFunction.adjoint: input must be 3-D, got shape {u.shape}."
            )
        return _convolve_channels(u, self._kernel_adj)


def _convolve_channels(cube: Array, kernel: Array) -> Array:
    """``SAME``-mode 2-D convolution per channel with **zero** padding.

    Zero-padding is what makes the adjoint cheap and exact: the discrete
    transpose of ``y = K * x`` (SAME-mode, zero-pad) is just convolution
    with the 180°-flipped kernel under the same boundary convention. Reflect
    padding would model instrument boundaries slightly more realistically,
    but its adjoint folds boundary contributions back into the interior —
    breaking the simple ``flip-and-convolve`` adjoint that variational DA
    relies on.

    ``vmap`` over the spectral axis fuses the per-channel convolutions in a
    single JIT-compiled kernel.
    """
    from jax.scipy.signal import convolve2d

    def _conv(channel: Array) -> Array:
        return convolve2d(channel, kernel, mode="same", boundary="fill", fillvalue=0.0)

    return jax.vmap(_conv, in_axes=-1, out_axes=-1)(cube)


# ── Ground Sampling Distance (block-mean down-sampling) ─────────────────────


@dataclass(frozen=True)
class GroundSamplingDistance:
    """Block-mean down-sampling operator for a fixed integer factor.

    Maps an HR cube ``(ny, nx, n_lambda)`` to LR
    ``(ny // f, nx // f, n_lambda)`` by averaging non-overlapping ``f × f``
    spatial blocks. If ``(ny, nx)`` is not divisible by ``f``, the input is
    truncated to ``(ny // f * f, nx // f * f)`` (top-left crop).

    The adjoint distributes each LR value uniformly across the corresponding
    HR block with weight ``1 / f²`` so that ``⟨A x, y⟩ = ⟨x, Aᵀ y⟩`` holds
    exactly under the standard Euclidean inner product.

    Attributes
    ----------
    downsample_factor : int
        Integer block size ``f``.

    Notes
    -----
    The classmethod :meth:`from_optics` derives ``f`` from the standard
    pinhole-camera GSD formula
    ``GSD [m/px] = sensor_width [mm] · altitude [m] · 1000 / (focal_length [mm] · image_width [px])``
    and rounds to the nearest integer. For non-integer ratios you should
    pre-resample the HR cube — block-mean is only well-defined for integer
    factors.
    """

    downsample_factor: int

    def __post_init__(self) -> None:
        if self.downsample_factor < 1:
            raise ValueError(
                f"GroundSamplingDistance: `downsample_factor` must be ≥ 1 "
                f"(got {self.downsample_factor})."
            )

    @classmethod
    def from_optics(
        cls,
        sensor_width_mm: float,
        focal_length_mm: float,
        image_width_px: int,
        altitude_m: float,
        pixel_size_hr_m: float,
    ) -> "GroundSamplingDistance":
        """Build a GSD operator from camera + altitude + HR pixel size.

        Dimensional analysis:

            GSD [m/px] = (sensor_pixel_size [m/px]) · (altitude [m]) / (focal_length [m])
                       = (sensor_width_mm / image_width_px) · altitude_m / focal_length_mm

        The ``mm`` units on numerator and denominator cancel, so no unit-conversion
        factor is needed — the numpy reference snippet we ported from had a spurious
        ``× 1000`` that inflated every realistic camera's GSD by a factor of 1000.
        """
        gsd_m_per_px = (sensor_width_mm * altitude_m) / (
            focal_length_mm * image_width_px
        )
        f = int(round(gsd_m_per_px / pixel_size_hr_m))
        return cls(downsample_factor=max(1, f))

    # ── shape helpers ────────────────────────────────────────────────────────

    def lr_shape(self, hr_shape: tuple[int, int, int]) -> tuple[int, int, int]:
        """LR shape implied by an HR shape and the current factor."""
        ny_hr, nx_hr, n_lambda = hr_shape
        f = self.downsample_factor
        return (ny_hr // f, nx_hr // f, n_lambda)

    # ── linear operator ──────────────────────────────────────────────────────

    def apply(self, radiance: HRCube) -> LRCube:
        """Forward down-sample by block mean."""
        x = jnp.asarray(radiance)
        if x.ndim != 3:
            raise ValueError(
                f"GroundSamplingDistance.apply: input must be 3-D, got shape {x.shape}."
            )
        ny_hr, nx_hr, n_lambda = x.shape
        f = self.downsample_factor
        ny_lr, nx_lr = ny_hr // f, nx_hr // f
        if ny_lr == 0 or nx_lr == 0:
            raise ValueError(
                f"GroundSamplingDistance.apply: input shape {x.shape[:2]} too small "
                f"for factor {f}."
            )
        x = x[: ny_lr * f, : nx_lr * f, :]
        return x.reshape(ny_lr, f, nx_lr, f, n_lambda).mean(axis=(1, 3))

    def jacobian(self, tangent: HRCube) -> LRCube:
        """JVP — same as :meth:`apply`."""
        return self.apply(tangent)

    def adjoint(self, cotangent: LRCube, hr_shape: tuple[int, int, int]) -> HRCube:
        """Adjoint: spread each LR value uniformly across its HR block.

        Parameters
        ----------
        cotangent : Array
            LR cotangent, shape ``(ny_lr, nx_lr, n_lambda)``.
        hr_shape : tuple
            Target HR shape ``(ny_hr, nx_hr, n_lambda)``. Truncated entries
            from :meth:`apply` (when ``ny_hr`` or ``nx_hr`` are not multiples
            of ``f``) are filled with zeros — the adjoint of the truncation
            is the inclusion ``(... → ny_lr·f rows)``.
        """
        u = jnp.asarray(cotangent)
        if u.ndim != 3:
            raise ValueError(
                f"GroundSamplingDistance.adjoint: input must be 3-D, got shape {u.shape}."
            )
        ny_hr, nx_hr, n_lambda = hr_shape
        f = self.downsample_factor
        ny_lr, nx_lr = ny_hr // f, nx_hr // f
        if u.shape != (ny_lr, nx_lr, n_lambda):
            raise ValueError(
                f"GroundSamplingDistance.adjoint: cotangent shape {u.shape} does not "
                f"match expected {(ny_lr, nx_lr, n_lambda)} for hr_shape={hr_shape}."
            )
        # Spread + scale: each LR value contributes 1/f² to every HR pixel
        # in its block (the gradient of a mean reduction).
        block = jnp.repeat(jnp.repeat(u, f, axis=0), f, axis=1) / (f * f)
        if ny_lr * f == ny_hr and nx_lr * f == nx_hr:
            return block
        # Pad with zeros along the truncated rows/cols.
        pad_y = ny_hr - ny_lr * f
        pad_x = nx_hr - nx_lr * f
        return jnp.pad(block, ((0, pad_y), (0, pad_x), (0, 0)))
