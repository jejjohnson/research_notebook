"""Iterative-Gaussianization (RBIG-style) warm-start for a built flow.

Walks the flow's bijector list in order. For each bijector:

- ``FixedOrtho``   — fit PCA on current ``Y`` and assign ``q``.
- ``Householder``  — fit PCA, Householder-QR-decompose the rotation,
  assign ``v``. Requires ``num_reflectors == d``.
- ``MixtureCDFGaussianization`` — fit a per-dim mixture of ``K``
  Gaussians (``sklearn.mixture.GaussianMixture``), assign
  ``(logits, means, log_scales)`` directly.
- ``MixtureCDFCoupling`` — fit per-b-dim GMMs on the b-dim subset of
  ``Y``, then overwrite the conditioner's **last** ``Dense`` so that
  kernel is zero and bias equals the packed mixture params (inverting
  the layer's ``log_scale_clamp`` so the clamped output matches the
  raw GMM fits). Inner Dense layers are left untouched.

After each bijector is parameterised, the current ``Y`` is propagated
forward through it (in numpy, avoiding Keras's loss side-channel) so
the next bijector sees the already-Gaussianized state. Laparra & Malo
(2011) show this alone drives ``Y`` towards a standard normal at a
geometric rate; the flow can then be refined by gradient descent from
this starting point.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from gaussianization.gauss_keras.bijectors import (
    FixedOrtho,
    GaussianizationFlow,
    Householder,
    MixtureCDFCoupling,
    MixtureCDFGaussianization,
)
from gaussianization.gauss_keras.bijectors._householder_decomp import (
    apply_reflectors,
    householder_decompose,
)


__all__ = ["initialize_flow_from_ig"]


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def _fit_gmm_per_dim(
    y_col: np.ndarray, num_components: int, random_state: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a 1-D diagonal GMM and return ``(logits, means, log_scales)``.

    Each returned array has shape ``(num_components,)``.
    """
    col = np.asarray(y_col, dtype=np.float64).reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=int(num_components),
        covariance_type="diag",
        random_state=int(random_state),
        reg_covar=1e-6,
    ).fit(col)
    weights = gmm.weights_.astype(np.float64)
    means = gmm.means_.reshape(-1).astype(np.float64)
    variances = gmm.covariances_.reshape(-1).astype(np.float64)
    log_scales = 0.5 * np.log(np.maximum(variances, 1e-12))
    logits = np.log(np.maximum(weights, 1e-20))
    return logits, means, log_scales


def _invert_log_scale_clamp(
    clamp: Callable[[Any], Any], clamped: np.ndarray
) -> np.ndarray:
    """Compute the raw pre-clamp value so that ``clamp(raw) ≈ clamped``."""
    kind = getattr(clamp, "kind", None)
    if kind == "tanh":
        bound = getattr(clamp, "bound", 3.0)
        y = np.clip(np.asarray(clamped, dtype=np.float64) / bound, -0.9999, 0.9999)
        return np.arctanh(y)
    if kind == "sigmoid":
        lo = getattr(clamp, "lo", -3.0)
        hi = getattr(clamp, "hi", 3.0)
        p = np.clip(
            (np.asarray(clamped, dtype=np.float64) - lo) / (hi - lo), 1e-6, 1 - 1e-6
        )
        return np.log(p / (1.0 - p))
    # Unknown clamp: fall back to treating it as identity (may be wrong —
    # we raise so the caller notices).
    raise ValueError(
        "initialize_flow_from_ig: coupling layer uses an un-tagged "
        "log_scale_clamp. Use tanh_log_scale_clamp or "
        "sigmoid_log_scale_clamp, or attach a .kind attribute so we "
        "can invert it."
    )


def _apply_marginal_numpy(
    y: np.ndarray,
    logits: np.ndarray,
    means: np.ndarray,
    log_scales: np.ndarray,
) -> np.ndarray:
    """Per-dim mixture-CDF Gaussianization ``z = Φ⁻¹(F(y))`` in numpy."""
    y = np.asarray(y, dtype=np.float64)
    n, d = y.shape
    z = np.empty_like(y)
    k = logits.shape[-1]
    weights = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    scales = np.exp(log_scales)
    for i in range(d):
        # shape (n, k)
        zc = (y[:, i : i + 1] - means[i][None, :]) / scales[i][None, :]
        comp_cdf = norm.cdf(zc)
        u = np.clip(np.sum(weights[i][None, :] * comp_cdf, axis=-1), 1e-6, 1 - 1e-6)
        z[:, i] = norm.ppf(u)
    return z.astype(np.float32)


def _find_last_dense(conditioner: Any) -> Any:
    import keras

    layers = list(getattr(conditioner, "layers", [conditioner]))
    for layer in reversed(layers):
        if isinstance(layer, keras.layers.Dense):
            return layer
    raise ValueError("No Dense layer found in conditioner")


def _is_shared_output(conditioner: Any, d_b: int, num_components: int) -> bool:
    """True if the last Dense emits ``3*K`` (shared) rather than ``3*d_b*K`` (per-dim)."""
    last = _find_last_dense(conditioner)
    out = int(last.kernel.shape[-1])
    if out == 3 * d_b * num_components:
        return False
    if out == 3 * num_components:
        return True
    raise ValueError(
        f"Unexpected conditioner output width {out}; expected "
        f"{3 * d_b * num_components} (per-dim) or {3 * num_components} "
        "(shared)."
    )


def _pack_bias(
    logits: np.ndarray,
    means: np.ndarray,
    raw_log_scales: np.ndarray,
    d_b: int,
    num_components: int,
    shared: bool,
) -> np.ndarray:
    """Flatten mixture params into the last Dense bias vector.

    The coupling layer reshapes ``flat -> (batch, 3, d_b, K)`` and splits
    on axis 1: ``[:, 0] -> logits``, ``[:, 1] -> means``, ``[:, 2] -> log_scales``.
    So the per-dim bias is ``stack([logits, means, raw_log_scales], axis=0).ravel()``
    with shape ``(3, d_b, K)``. For shared the last Dense emits one
    per-param block of size ``3*K`` which is tiled to ``d_b`` dims by
    the downstream ``_RepeatAcrossDims``; we pack a single ``(3, K)``.
    """
    k = int(num_components)
    if shared:
        # Average per-b-dim fits into a single mixture.
        logits_s = logits.mean(axis=0)
        means_s = means.mean(axis=0)
        raw_log_scales_s = raw_log_scales.mean(axis=0)
        return np.stack(
            [logits_s, means_s, raw_log_scales_s], axis=0
        ).ravel().astype(np.float32)  # shape (3*K,)
    # Per-dim.
    return np.stack(
        [logits, means, raw_log_scales], axis=0
    ).ravel().astype(np.float32)  # shape (3*d_b*K,)


# ------------------------------------------------------------------ #
# Per-bijector initialisers (each returns the forward-propagated Y)     #
# ------------------------------------------------------------------ #


def _init_fixed_ortho(layer: FixedOrtho, y: np.ndarray) -> np.ndarray:
    d = y.shape[-1]
    pca = PCA(n_components=d).fit(y)
    q = pca.components_.T  # (d, d); each column is a principal axis
    layer.q.assign(q.astype(np.float32))
    return (y @ q).astype(np.float32)


def _init_householder(layer: Householder, y: np.ndarray) -> np.ndarray:
    d = y.shape[-1]
    pca = PCA(n_components=d).fit(y)
    q = pca.components_.T
    v = householder_decompose(q, expected_num_reflectors=layer.num_reflectors)
    layer.v.assign(v.astype(np.float32))
    return apply_reflectors(v, y).astype(np.float32)


def _dim_seed(block_idx: int, dim_idx: int, random_state: int) -> int:
    """Per-(block, dim) EM seed.

    Making the GMM seed depend only on ``(block_idx, dim_idx)`` — not on
    the order layers are visited within a block — means a diagonal flow
    and a coupling flow fitted on the same data assign the *same* fit to
    the same dim at the same block. That is what makes the equivalence
    ``zero-kernel coupling ≡ diagonal marginal`` hold to float precision.
    """
    return int(random_state) + int(block_idx) * 1000 + int(dim_idx)


def _init_marginal(
    layer: MixtureCDFGaussianization,
    y: np.ndarray,
    block_idx: int,
    random_state: int,
) -> np.ndarray:
    d = y.shape[-1]
    k = layer.num_components
    logits = np.zeros((d, k), dtype=np.float64)
    means = np.zeros((d, k), dtype=np.float64)
    log_scales = np.zeros((d, k), dtype=np.float64)
    for i in range(d):
        logits[i], means[i], log_scales[i] = _fit_gmm_per_dim(
            y[:, i], k, _dim_seed(block_idx, i, random_state)
        )
    layer.logits.assign(logits.astype(np.float32))
    layer.means.assign(means.astype(np.float32))
    layer.log_scales.assign(log_scales.astype(np.float32))
    return _apply_marginal_numpy(y, logits, means, log_scales)


def _init_coupling(
    layer: MixtureCDFCoupling,
    y: np.ndarray,
    block_idx: int,
    random_state: int,
) -> np.ndarray:
    """Zero the conditioner's final kernel and set bias = GMM fits of y[:, b_idx]."""
    b_idx = layer._b_idx
    d_b = layer.d_b
    k = layer.num_components

    # Per-b-dim GMM fits, seeded by (block_idx, dim_idx) so a coupling
    # pair in one block uses the same seeds as a diagonal marginal on
    # the same dims.
    logits = np.zeros((d_b, k), dtype=np.float64)
    means = np.zeros((d_b, k), dtype=np.float64)
    log_scales = np.zeros((d_b, k), dtype=np.float64)
    for j in range(d_b):
        dim_idx = int(b_idx[j])
        col = y[:, dim_idx]
        logits[j], means[j], log_scales[j] = _fit_gmm_per_dim(
            col, k, _dim_seed(block_idx, dim_idx, random_state)
        )

    # Invert the layer's log-scale clamp so the clamped output = log_scales.
    raw_log_scales = _invert_log_scale_clamp(layer.log_scale_clamp, log_scales)

    shared = _is_shared_output(layer.conditioner, d_b, k)
    bias = _pack_bias(logits, means, raw_log_scales, d_b, k, shared)

    last_dense = _find_last_dense(layer.conditioner)
    kernel_shape = last_dense.kernel.shape
    last_dense.kernel.assign(np.zeros(tuple(int(s) for s in kernel_shape), dtype=np.float32))
    last_dense.bias.assign(bias)

    # Propagate y numpy-side: a-dims passthrough, b-dims transformed by
    # the (now-fixed) per-b-dim mixture. For shared, every b-dim uses the
    # averaged params.
    y_new = y.copy()
    if shared:
        logits_eff = np.broadcast_to(
            logits.mean(axis=0, keepdims=True), (d_b, k)
        )
        means_eff = np.broadcast_to(
            means.mean(axis=0, keepdims=True), (d_b, k)
        )
        log_scales_eff = np.broadcast_to(
            log_scales.mean(axis=0, keepdims=True), (d_b, k)
        )
    else:
        logits_eff, means_eff, log_scales_eff = logits, means, log_scales

    y_b = y[:, b_idx.astype(np.int64)]
    z_b = _apply_marginal_numpy(y_b, logits_eff, means_eff, log_scales_eff)
    y_new[:, b_idx.astype(np.int64)] = z_b
    return y_new.astype(np.float32)


# ------------------------------------------------------------------ #
# Public entry point                                                   #
# ------------------------------------------------------------------ #


def initialize_flow_from_ig(
    flow: GaussianizationFlow,
    x: np.ndarray,
    random_state: int = 0,
) -> np.ndarray:
    """Warm-start a built flow's weights via iterative Gaussianization.

    Args:
        flow: a ``GaussianizationFlow`` whose bijectors are already
            built (run one forward pass first, e.g. ``_ = flow(x[:4])``).
        x: training data of shape ``(n, d)``.
        random_state: base seed for the sklearn GMM EM fits. Each
            per-dim GMM fit uses seed ``random_state + block_idx * 1000
            + dim_idx``, where ``block_idx`` increments from 0 on the
            first rotation (``FixedOrtho`` / ``Householder``) and then
            by one each subsequent rotation. If the flow has no
            rotation before the first marginal/coupling layer,
            ``block_idx`` is clamped to 0 for that leading transform.
            This matching means a coupling flow and a diagonal flow
            with the same block structure assign *identical* fits to
            the same dim at the same block.

    Returns:
        The final pushforward of ``x`` through the initialised flow as
        a ``float32`` numpy array — useful for diagnostics (it should
        already be approximately ``N(0, I)`` before any training).
    """
    if not isinstance(flow, GaussianizationFlow):
        raise TypeError(f"expected GaussianizationFlow; got {type(flow).__name__}")
    y = np.asarray(x, dtype=np.float32)
    if y.ndim != 2:
        raise ValueError(f"x must be 2-D (n, d); got shape {y.shape}")
    if y.shape[-1] != flow.input_dim:
        raise ValueError(
            f"x has d={y.shape[-1]} but flow.input_dim={flow.input_dim}"
        )

    block_idx = -1
    for layer in flow.bijector_layers:
        if not layer.built:
            raise RuntimeError(
                f"bijector {type(layer).__name__} is not built; run a "
                "forward pass through the flow before calling "
                "initialize_flow_from_ig so all weights exist."
            )
        if isinstance(layer, FixedOrtho):
            block_idx += 1
            y = _init_fixed_ortho(layer, y)
        elif isinstance(layer, Householder):
            block_idx += 1
            y = _init_householder(layer, y)
        elif isinstance(layer, MixtureCDFGaussianization):
            y = _init_marginal(layer, y, max(block_idx, 0), random_state)
        elif isinstance(layer, MixtureCDFCoupling):
            y = _init_coupling(layer, y, max(block_idx, 0), random_state)
        else:
            raise ValueError(
                f"initialize_flow_from_ig: unsupported bijector type "
                f"{type(layer).__name__}"
            )
    return y
