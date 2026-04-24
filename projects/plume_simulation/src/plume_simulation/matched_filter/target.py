"""Target signature construction from the radiance forward model.

The matched-filter target ``t(ν)`` is the spectral signature that a unit
plume enhancement would leave in the observed radiance. Because we already
have a JAX :class:`plume_simulation.assimilation.RadianceObservationModel`,
we do *not* reimplement the Beer–Lambert derivation from scratch (1379 lines
in the original snippet). Instead:

- **Linear target** is a directional derivative ``∂L/∂VMR`` at the background
  state, obtained by a single :func:`jax.jvp` call — robust, cheap, and
  exactly consistent with the forward model used by the 3D-Var retrieval.
- **Nonlinear target** is the finite-amplitude response
  ``L(x_b + α·dx) − L(x_b)``, which captures Beer–Lambert saturation for
  strong plumes. For small ``α`` this reduces to ``α · t_linear`` and the MF
  score becomes scale-invariant to that choice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


if TYPE_CHECKING:
    from plume_simulation.assimilation.obs_operator import RadianceObservationModel


Pattern = Literal["uniform", "impulse"]


def _build_pattern(
    vmr_field: Float[Array, "H W"],
    pattern: Pattern | Float[Array, "H W"],
    pixel: tuple[int, int] | None,
) -> Float[Array, "H W"]:
    if isinstance(pattern, str):
        if pattern == "uniform":
            return jnp.ones_like(vmr_field)
        if pattern == "impulse":
            i, j = (
                pixel
                if pixel is not None
                else (
                    vmr_field.shape[0] // 2,
                    vmr_field.shape[1] // 2,
                )
            )
            return jnp.zeros_like(vmr_field).at[i, j].set(1.0)
        raise ValueError(
            f"target: pattern must be 'uniform', 'impulse', or an array; got {pattern!r}."
        )
    arr = jnp.asarray(pattern)
    if arr.shape != vmr_field.shape:
        raise ValueError(
            f"target: pattern array shape {arr.shape} must match vmr_field shape "
            f"{vmr_field.shape}."
        )
    return arr


def _extract_pixel(
    cube: Float[Array, "H W B"],
    pattern: Pattern | Float[Array, "H W"],
    pixel: tuple[int, int] | None,
) -> Float[Array, "B"]:
    """Pick one pixel spectrum out of a response cube.

    For spatially-uniform patterns we take the scene centre (every pixel sees
    the same signature). For impulse or custom patterns we return the pixel
    matching the perturbation location — or the supplied ``pixel`` override.
    """
    h, w, _ = cube.shape
    if pixel is not None:
        i, j = pixel
    elif isinstance(pattern, str) and pattern == "impulse":
        i, j = h // 2, w // 2
    else:
        i, j = h // 2, w // 2
    return cube[i, j]


def linear_target_from_obs(
    obs: RadianceObservationModel,
    vmr_background: Float[Array, "H W"],
    *,
    pattern: Pattern | Float[Array, "H W"] = "uniform",
    pixel: tuple[int, int] | None = None,
    linear_forward: bool = False,
) -> Float[Array, "B"]:
    """Matched-filter target via the tangent-linear of the forward operator.

    Computes

    .. math::
        t = \\frac{\\partial}{\\partial \\alpha}
            H(x_b + \\alpha \\, \\delta x) \\Big|_{\\alpha=0}

    at a single output pixel. For the default ``pattern='uniform'`` this
    captures the spectral signature of a uniform VMR enhancement across the
    scene (appropriate when the plume fills several PSF footprints). For
    ``pattern='impulse'`` it captures the instrument's point response to a
    single-pixel enhancement (appropriate when the plume is subpixel or
    compact compared to the PSF).

    Parameters
    ----------
    obs
        The JAX observation model.
    vmr_background
        Background VMR field, shape ``(H, W)``. Typically
        ``jnp.full((H, W), x_ref)`` where ``x_ref == obs.vmr_reference``.
    pattern
        ``'uniform'`` (default), ``'impulse'``, or an explicit 2-D perturbation
        pattern of the same shape as ``vmr_background``.
    pixel
        Output pixel ``(i, j)`` at which to read the signature. Defaults to
        the scene centre, or the impulse location if applicable.
    linear_forward
        If True, use the linearised forward ``1 - Δτ`` instead of
        ``exp(-Δτ)``. For small ``vmr_background`` the two agree to within
        Beer–Lambert curvature.

    Returns
    -------
    Float[Array, "B"]
        Target signature of length ``obs.n_bands``.
    """
    dx = _build_pattern(vmr_background, pattern, pixel)
    fwd = obs.make_forward(linear=linear_forward)
    _, dL = jax.jvp(fwd, (vmr_background,), (dx,))
    return _extract_pixel(dL, pattern, pixel)


def nonlinear_target_from_obs(
    obs: RadianceObservationModel,
    vmr_background: Float[Array, "H W"],
    *,
    amplitude: float = 1.0,
    pattern: Pattern | Float[Array, "H W"] = "uniform",
    pixel: tuple[int, int] | None = None,
) -> Float[Array, "B"]:
    """Matched-filter target via a finite-amplitude difference.

    Computes ``t = H(x_b + α·δx) − H(x_b)`` without linearising Beer–Lambert.
    This is the "nonlinear" variant from the original snippets; it captures
    the saturation curvature that matters for strong plumes
    (``α · σ · N · ΔX ≳ 0.1`` optical depth).

    For small ``amplitude`` this reduces to ``amplitude × linear_target``.
    """
    dx = _build_pattern(vmr_background, pattern, pixel)
    fwd = obs.make_forward(linear=False)
    L_perturbed = fwd(vmr_background + amplitude * dx)
    L_base = fwd(vmr_background)
    return _extract_pixel(L_perturbed - L_base, pattern, pixel)
