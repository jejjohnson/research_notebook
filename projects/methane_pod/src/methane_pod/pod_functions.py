"""
Probability of Detection (POD) Functions — Equinox Modules
==========================================================

A family of GLM-style POD models P_d(x) ∈ [0, 1] for methane plume
detectability from satellite remote sensing, packaged as ``eqx.Module``.

Each module bundles:
  • Covariate transformation + link function in ``__call__``.
  • ``sample_priors(cls)`` → NumPyro prior factory for MCMC.

The progression mirrors the intensity module: simple → complex, each
relaxing an assumption of the last.

Physical Context
----------------
Detection is NOT a deterministic threshold. It is a probabilistic event
governed by the interplay of source strength, atmospheric dilution,
instrument optics, and radiometric conditions. We model this as a
Bernoulli GLM:

    Y ~ Bernoulli(P_d(x))

where Y ∈ {0, 1} (detected / not detected) and P_d is the probability
of detection evaluated at covariates x.

Convention
----------
  • Flux Q in [kg hr⁻¹].
  • Wind speed U in [m s⁻¹].
  • Pixel size p in [m].
  • Albedo A in [–], ∈ [0, 1].
  • Solar zenith angle θ in [°]; models use cos(θ) ∈ [0.26, 1.0].
  • Spectral bands n_bands [dimensionless count].
  • Noise σ_noise in [ppb] or instrument-specific units.
  • All modules are immutable pytrees, jit/vmap/grad compatible.

Dependencies: jax, equinox, numpyro
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import erf as _jax_erf
import equinox as eqx
import numpyro
import numpyro.distributions as dist


# ═════════════════════════════════════════════════════════════════════════════
# COVARIATE TRANSFORMS
# ═════════════════════════════════════════════════════════════════════════════
# Standalone pure functions: each maps raw physical observables to the
# feature space consumed by the GLM linear predictor.
# All are jit/vmap/grad safe.
# ═════════════════════════════════════════════════════════════════════════════

def log_flux(Q: jnp.ndarray) -> jnp.ndarray:
    """Log-transform of methane mass flux.

    Physical Motivation
    -------------------
    Flux spans 4+ orders of magnitude (1–10 000 kg hr⁻¹). The log
    compresses the super-emitter tail and linearises the relationship
    with detection probability in logit space.

    Equation
    --------
        x = ln(Q)

    Parameters
    ----------
    Q : jnp.ndarray, shape (N,)
        Methane mass flux [kg hr⁻¹]. Must be > 0.

    Returns
    -------
    x : jnp.ndarray, shape (N,)
        Log-flux [ln(kg hr⁻¹)].
    """
    return jnp.log(jnp.clip(Q, min=1e-8))


def log_concentration_proxy(
    Q: jnp.ndarray,
    U: jnp.ndarray,
    p: jnp.ndarray,
) -> jnp.ndarray:
    """Log concentration proxy combining flux, wind, and pixel size.

    Physical Motivation
    -------------------
    What a satellite actually "sees" is the column-integrated methane
    enhancement ΔΩ ∝ Q / (U · p). This proxy captures the three dominant
    first-order controls on detectability in a single dimensionless
    scalar:
      • Q ↑ → more CH₄ in the pixel → easier to detect.
      • U ↑ → plume diluted faster → harder to detect.
      • p ↑ → more background in the pixel → fill-factor dilution.

    Used by Varon et al. (2018), Cusworth et al. (2021).

    Equation
    --------
        x = ln(Q / (U · p)) = ln(Q) − ln(U) − ln(p)

    Parameters
    ----------
    Q : jnp.ndarray, shape (N,)
        Methane mass flux [kg hr⁻¹].
    U : jnp.ndarray, shape (N,)
        Wind speed [m s⁻¹].
    p : jnp.ndarray, shape (N,)
        Pixel ground sampling distance [m].

    Returns
    -------
    x : jnp.ndarray, shape (N,)
        Log concentration proxy [dimensionless].
    """
    return (
        jnp.log(jnp.clip(Q, min=1e-8))
        - jnp.log(jnp.clip(U, min=1e-3))
        - jnp.log(jnp.clip(p, min=1.0))
    )


def log_flux_wind_ratio(
    Q: jnp.ndarray,
    U: jnp.ndarray,
) -> jnp.ndarray:
    """Log flux-to-wind ratio (pixel-free proxy).

    Physical Motivation
    -------------------
    When pixel size is fixed for a single instrument, the Q/U ratio alone
    governs the column enhancement. This is the proxy used by the IMEO
    james_jax.py POD fits (Q/U in [kg hr⁻¹ / (m s⁻¹)]).

    Equation
    --------
        x = ln(Q / U)

    Parameters
    ----------
    Q : jnp.ndarray, shape (N,)
        Methane mass flux [kg hr⁻¹].
    U : jnp.ndarray, shape (N,)
        Wind speed [m s⁻¹].

    Returns
    -------
    x : jnp.ndarray, shape (N,)
        Log flux-wind ratio [ln(kg hr⁻¹ / (m s⁻¹))].
    """
    return jnp.log(jnp.clip(Q, min=1e-8)) - jnp.log(jnp.clip(U, min=1e-3))


def snr_proxy(
    Q: jnp.ndarray,
    U: jnp.ndarray,
    p: jnp.ndarray,
    albedo: jnp.ndarray,
    sigma_noise: jnp.ndarray,
) -> jnp.ndarray:
    """Signal-to-noise ratio proxy for radiometric detection.

    Physical Motivation
    -------------------
    Detection fundamentally reduces to SNR. The signal is the column
    enhancement ΔΩ ∝ Q/(U·p). The noise floor depends on:
      • Albedo A: brighter surfaces → more reflected photons → lower
        shot noise → better SNR.
      • Instrument noise σ_n: dark current, read noise, spectral fitting
        residuals.

    Equation
    --------
        SNR = (Q / (U · p)) · √A / σ_noise

    The model consumes ln(SNR) to linearise the logit relationship.

    Parameters
    ----------
    Q : jnp.ndarray, shape (N,)
        Methane mass flux [kg hr⁻¹].
    U : jnp.ndarray, shape (N,)
        Wind speed [m s⁻¹].
    p : jnp.ndarray, shape (N,)
        Pixel size [m].
    albedo : jnp.ndarray, shape (N,)
        Surface albedo [–].
    sigma_noise : jnp.ndarray, shape (N,)
        Instrument noise floor [ppb or instrument units].

    Returns
    -------
    x : jnp.ndarray, shape (N,)
        ln(SNR) [dimensionless].
    """
    signal = jnp.clip(Q, min=1e-8) / (jnp.clip(U, min=1e-3) * jnp.clip(p, min=1.0))
    noise_inv = jnp.sqrt(jnp.clip(albedo, min=1e-4)) / jnp.clip(sigma_noise, min=1e-6)
    return jnp.log(signal * noise_inv + 1e-12)


def cos_zenith(theta_deg: jnp.ndarray) -> jnp.ndarray:
    """Cosine of solar zenith angle.

    Physical Motivation
    -------------------
    Solar zenith θ enters the radiative transfer equation through the
    air-mass factor 1/cos(θ). Higher sun (smaller θ, larger cos θ)
    means shorter path length → less atmospheric scattering → cleaner
    retrieval. Most instruments reject observations at θ > 70°.

    Equation
    --------
        x = cos(θ · π / 180)

    Parameters
    ----------
    theta_deg : jnp.ndarray, shape (N,)
        Solar zenith angle [degrees].

    Returns
    -------
    x : jnp.ndarray, shape (N,)
        cos(θ) [dimensionless], ∈ [0.26, 1.0] for θ ∈ [0°, 75°].
    """
    return jnp.cos(jnp.deg2rad(theta_deg))


# ═════════════════════════════════════════════════════════════════════════════
# 1. LOGISTIC (Simplest GLM — 1D Sigmoid)
# ═════════════════════════════════════════════════════════════════════════════

class LogisticPOD(eqx.Module):
    """Single-covariate logistic POD: the classic S-curve.

    Physical Motivation
    -------------------
    The simplest detection model: probability depends only on flux Q
    through a logistic sigmoid. This is the "Q₅₀" model universally
    used in first-pass satellite characterisation. The satellite has
    a detection midpoint Q₅₀ where P_d = 0.5, and a steepness k
    controlling how sharply the curve transitions from 0 → 1.

    Ignores wind, albedo, zenith — appropriate for controlled-release
    experiments under constant atmospheric conditions or as a baseline.

    Equation
    --------
        P_d(Q) = σ(k · (Q − Q₅₀)) = 1 / (1 + exp(−k · (Q − Q₅₀)))

    Equivalent logit form:
        logit(P_d) = k · (Q − Q₅₀) = k·Q − k·Q₅₀ = β₁·Q + β₀

    Attributes
    ----------
    Q_50 : float
        Flux at 50% detection probability [kg hr⁻¹].
        Typical: GHGSat ≈ 100, TROPOMI ≈ 3000, MethaneSAT ≈ 200.
    k : float
        Steepness of the sigmoid [hr kg⁻¹].
        Typical: 0.005–0.05.
    """

    Q_50: float
    k: float

    def __call__(self, Q: jnp.ndarray) -> jnp.ndarray:
        """Evaluate P_d.

        Parameters
        ----------
        Q : jnp.ndarray, shape (N,)
            Methane mass flux [kg hr⁻¹].

        Returns
        -------
        P_d : jnp.ndarray, shape (N,)
            Probability of detection [0, 1].
        """
        return 1.0 / (1.0 + jnp.exp(-self.k * (Q - self.Q_50)))

    @classmethod
    def sample_priors(cls) -> "LogisticPOD":
        """Sample NumPyro priors.

        Priors
        ------
            Q₅₀ ~ LogNormal(ln(500), 1.0)   [kg hr⁻¹]
            k   ~ HalfNormal(0.02)           [hr kg⁻¹]
        """
        return cls(
            Q_50=numpyro.sample("Q_50", dist.LogNormal(jnp.log(500.0), 1.0)),
            k=numpyro.sample("k", dist.HalfNormal(0.02)),
        )


# ═════════════════════════════════════════════════════════════════════════════
# 2. LOG-LOGISTIC (Log-Flux GLM)
# ═════════════════════════════════════════════════════════════════════════════

class LogLogisticPOD(eqx.Module):
    """Logistic POD on log-flux: the lognormal CDF approximation.

    Physical Motivation
    -------------------
    Because flux spans orders of magnitude, the logistic curve on raw Q
    is poorly calibrated: it transitions too slowly across the super-
    emitter tail. Operating in log-space produces the lognormal CDF
    shape used by IMEO/Kamdar et al. for per-satellite POD fits.

    This is the model behind james_jax.py: P_d(Q) ≈ Φ_LN(Q; Q₅₀, σ).

    Equation
    --------
        logit(P_d) = β₀ + β₁ · ln(Q)

    Equivalently, the transition midpoint is at Q = exp(−β₀/β₁).

    Attributes
    ----------
    beta_0 : float
        Intercept [logit units]. Controls the detection threshold.
    beta_1 : float
        Slope w.r.t. ln(Q) [logit per ln(kg hr⁻¹)].
        Positive → higher flux → more detectable.
    """

    beta_0: float
    beta_1: float

    def __call__(self, Q: jnp.ndarray) -> jnp.ndarray:
        """Evaluate P_d.

        Parameters
        ----------
        Q : jnp.ndarray, shape (N,)
            Methane mass flux [kg hr⁻¹].

        Returns
        -------
        P_d : jnp.ndarray, shape (N,)
            Probability of detection [0, 1].
        """
        logit = self.beta_0 + self.beta_1 * log_flux(Q)
        return 1.0 / (1.0 + jnp.exp(-logit))

    @classmethod
    def sample_priors(cls) -> "LogLogisticPOD":
        """Sample NumPyro priors.

        Priors
        ------
            β₀ ~ Normal(−8.0, 3.0)    [logit]
            β₁ ~ HalfNormal(2.0)      [logit per ln(kg hr⁻¹)]
        """
        return cls(
            beta_0=numpyro.sample("beta_0", dist.Normal(-8.0, 3.0)),
            beta_1=numpyro.sample("beta_1", dist.HalfNormal(2.0)),
        )


# ═════════════════════════════════════════════════════════════════════════════
# 3. CONCENTRATION-PROXY GLM (Flux + Wind + Pixel)
# ═════════════════════════════════════════════════════════════════════════════

class ConcentrationProxyPOD(eqx.Module):
    """GLM on the log-concentration proxy ln(Q/(U·p)).

    Physical Motivation
    -------------------
    The column-integrated methane enhancement ΔΩ ∝ Q/(U·p) is the
    primary observable. This model captures the three dominant controls
    in a single log-linear predictor. Assumes their effects are
    multiplicatively separable in concentration space (additive in
    log-space).

    Used in Varon et al. (2018), the canonical IME framework.

    Equation
    --------
        logit(P_d) = β₀ + β₁ · ln(Q / (U · p))

    Attributes
    ----------
    beta_0 : float
        Intercept [logit units].
    beta_1 : float
        Sensitivity to log-concentration proxy
        [logit per dimensionless unit].
    """

    beta_0: float
    beta_1: float

    def __call__(
        self,
        Q: jnp.ndarray,
        U: jnp.ndarray,
        p: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate P_d.

        Parameters
        ----------
        Q : jnp.ndarray, shape (N,)
            Methane mass flux [kg hr⁻¹].
        U : jnp.ndarray, shape (N,)
            Wind speed [m s⁻¹].
        p : jnp.ndarray, shape (N,)
            Pixel size [m].

        Returns
        -------
        P_d : jnp.ndarray, shape (N,)
            Probability of detection [0, 1].
        """
        x = log_concentration_proxy(Q, U, p)
        logit = self.beta_0 + self.beta_1 * x
        return 1.0 / (1.0 + jnp.exp(-logit))

    @classmethod
    def sample_priors(cls) -> "ConcentrationProxyPOD":
        """Sample NumPyro priors.

        Priors
        ------
            β₀ ~ Normal(−3.0, 2.0)   [logit]
            β₁ ~ HalfNormal(2.0)     [logit per unit]
        """
        return cls(
            beta_0=numpyro.sample("beta_0", dist.Normal(-3.0, 2.0)),
            beta_1=numpyro.sample("beta_1", dist.HalfNormal(2.0)),
        )


# ═════════════════════════════════════════════════════════════════════════════
# 4. ADDITIVE MULTI-COVARIATE GLM (Separable Effects)
# ═════════════════════════════════════════════════════════════════════════════

class AdditiveMultiCovariatePOD(eqx.Module):
    """Additive GLM with separate coefficients for each covariate.

    Physical Motivation
    -------------------
    Relaxes the single-proxy assumption: flux, wind, pixel size, albedo,
    and illumination each receive an independent coefficient. This allows
    the model to learn that, e.g., wind has a stronger dilution effect
    than pixel size, or that albedo matters more than zenith angle.

    The linear predictor is a standard multivariate logistic regression.

    Equation
    --------
        logit(P_d) = β₀ + β_Q·ln(Q) + β_U·ln(U) + β_p·ln(p)
                        + β_A·A + β_θ·cos(θ)

    Expected sign constraints (encoded in priors):
      • β_Q > 0  (more flux → more detectable)
      • β_U < 0  (more wind → more dilution → less detectable)
      • β_p < 0  (larger pixel → more fill-factor dilution)
      • β_A > 0  (brighter surface → better SNR)
      • β_θ > 0  (higher sun → less atmospheric path → cleaner)

    Attributes
    ----------
    beta_0 : float
        Intercept [logit].
    beta_Q : float
        Log-flux coefficient [logit per ln(kg hr⁻¹)].
    beta_U : float
        Log-wind coefficient [logit per ln(m s⁻¹)].
    beta_p : float
        Log-pixel coefficient [logit per ln(m)].
    beta_A : float
        Albedo coefficient [logit per unit albedo].
    beta_theta : float
        Cosine-zenith coefficient [logit per unit cos(θ)].
    """

    beta_0: float
    beta_Q: float
    beta_U: float
    beta_p: float
    beta_A: float
    beta_theta: float

    def __call__(
        self,
        Q: jnp.ndarray,
        U: jnp.ndarray,
        p: jnp.ndarray,
        albedo: jnp.ndarray,
        cos_theta: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate P_d.

        Parameters
        ----------
        Q : jnp.ndarray, shape (N,)
            Methane mass flux [kg hr⁻¹].
        U : jnp.ndarray, shape (N,)
            Wind speed [m s⁻¹].
        p : jnp.ndarray, shape (N,)
            Pixel size [m].
        albedo : jnp.ndarray, shape (N,)
            Surface albedo [–].
        cos_theta : jnp.ndarray, shape (N,)
            Cosine of solar zenith angle [–].

        Returns
        -------
        P_d : jnp.ndarray, shape (N,)
            Probability of detection [0, 1].
        """
        logit = (
            self.beta_0
            + self.beta_Q * log_flux(Q)
            + self.beta_U * jnp.log(jnp.clip(U, min=1e-3))
            + self.beta_p * jnp.log(jnp.clip(p, min=1.0))
            + self.beta_A * albedo
            + self.beta_theta * cos_theta
        )
        return 1.0 / (1.0 + jnp.exp(-logit))

    @classmethod
    def sample_priors(cls) -> "AdditiveMultiCovariatePOD":
        """Sample NumPyro priors.

        Priors (sign-constrained where physics dictates)
        ------
            β₀  ~ Normal(−5.0, 3.0)
            β_Q ~ HalfNormal(2.0)       (positive: more flux → detected)
            β_U ~ Normal(−1.0, 1.0)     (negative: dilution)
            β_p ~ Normal(−0.5, 0.5)     (negative: fill-factor)
            β_A ~ HalfNormal(1.5)       (positive: brighter → better)
            β_θ ~ HalfNormal(1.0)       (positive: higher sun → better)
        """
        return cls(
            beta_0=numpyro.sample("beta_0", dist.Normal(-5.0, 3.0)),
            beta_Q=numpyro.sample("beta_Q", dist.HalfNormal(2.0)),
            beta_U=numpyro.sample("beta_U", dist.Normal(-1.0, 1.0)),
            beta_p=numpyro.sample("beta_p", dist.Normal(-0.5, 0.5)),
            beta_A=numpyro.sample("beta_A", dist.HalfNormal(1.5)),
            beta_theta=numpyro.sample("beta_theta", dist.HalfNormal(1.0)),
        )


# ═════════════════════════════════════════════════════════════════════════════
# 5. VARYING COEFFICIENT GLM (Environment-Dependent Sensitivity)
# ═════════════════════════════════════════════════════════════════════════════

class VaryingCoefficientPOD(eqx.Module):
    """Varying coefficient GLM: sensitivity β₁ depends on environment.

    Physical Motivation
    -------------------
    The slope of the detection curve is NOT constant — it changes with
    observing conditions. Under high albedo and overhead sun, the
    instrument's spectral retrieval is cleaner, so the sigmoid is
    *steeper* (the satellite becomes a better detector). Under low
    albedo or high zenith, the slope flattens.

    This is the "Refined Varying Coefficient Formulation" from the
    pod_numpyro.py module:

        β₁(Z) = γ_base + γ_A · A + γ_θ · cos(θ)

    The key insight: albedo and zenith don't shift the curve left/right
    (that's the additive model) — they change *how sharply* the
    instrument discriminates between detectable and undetectable plumes.

    Equation
    --------
        β₁(Z) = γ_base + γ_A · A + γ_θ · cos(θ)
        logit(P_d) = β₀ + β₁(Z) · ln(Q / (U · p))

    Attributes
    ----------
    beta_0 : float
        Baseline threshold [logit]. Typical: [−6, −2].
    gamma_base : float
        Intrinsic instrument sensitivity [logit per log-proxy-unit].
        Typical: [0.5, 3.0].
    gamma_albedo : float
        Albedo enhancement [logit per log-unit per unit albedo].
        Typical: [0.5, 2.0].
    gamma_cos_theta : float
        Illumination enhancement [logit per log-unit per unit cos(θ)].
        Typical: [0.3, 1.5].
    """

    beta_0: float
    gamma_base: float
    gamma_albedo: float
    gamma_cos_theta: float

    def __call__(
        self,
        Q: jnp.ndarray,
        U: jnp.ndarray,
        p: jnp.ndarray,
        albedo: jnp.ndarray,
        cos_theta: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate P_d.

        Parameters
        ----------
        Q : jnp.ndarray, shape (N,)
            Methane mass flux [kg hr⁻¹].
        U : jnp.ndarray, shape (N,)
            Wind speed [m s⁻¹].
        p : jnp.ndarray, shape (N,)
            Pixel size [m].
        albedo : jnp.ndarray, shape (N,)
            Surface albedo [–].
        cos_theta : jnp.ndarray, shape (N,)
            Cosine of solar zenith angle [–].

        Returns
        -------
        P_d : jnp.ndarray, shape (N,)
            Probability of detection [0, 1].
        """
        x = log_concentration_proxy(Q, U, p)
        beta_1 = (
            self.gamma_base
            + self.gamma_albedo * albedo
            + self.gamma_cos_theta * cos_theta
        )
        logit = self.beta_0 + beta_1 * x
        return 1.0 / (1.0 + jnp.exp(-logit))

    @classmethod
    def sample_priors(cls) -> "VaryingCoefficientPOD":
        """Sample NumPyro priors.

        Priors
        ------
            β₀  ~ Normal(−3.5, 2.0)
            γ_b ~ HalfNormal(1.5)
            γ_A ~ HalfNormal(1.0)
            γ_θ ~ Normal(0.5, 0.5)
        """
        return cls(
            beta_0=numpyro.sample("beta_0", dist.Normal(-3.5, 2.0)),
            gamma_base=numpyro.sample("gamma_base", dist.HalfNormal(1.5)),
            gamma_albedo=numpyro.sample("gamma_albedo", dist.HalfNormal(1.0)),
            gamma_cos_theta=numpyro.sample("gamma_cos_theta", dist.Normal(0.5, 0.5)),
        )


# ═════════════════════════════════════════════════════════════════════════════
# 6. SNR-BASED GLM (Radiometric First-Principles)
# ═════════════════════════════════════════════════════════════════════════════

class SNRBasedPOD(eqx.Module):
    """GLM on log-SNR: radiometric first-principles detection model.

    Physical Motivation
    -------------------
    At the hardware level, detection reduces to a hypothesis test on the
    retrieved methane column: "is the enhancement above the noise floor?"
    The test statistic is the signal-to-noise ratio:

        SNR = ΔΩ / σ_retrieval ∝ (Q / (U·p)) · √A / σ_noise

    where σ_noise encapsulates dark current, read noise, spectral fitting
    residuals, and atmospheric interference. Detection occurs when
    SNR > τ, which in the probabilistic GLM becomes:

        P_d = σ(β₀ + β₁ · ln(SNR))

    This model is appropriate when instrument noise characteristics are
    known or co-estimated, e.g., from calibration campaigns.

    Equation
    --------
        SNR = (Q / (U · p)) · √A / σ_noise
        logit(P_d) = β₀ + β₁ · ln(SNR)

    Attributes
    ----------
    beta_0 : float
        Intercept [logit]. Encodes the detection threshold in SNR space.
    beta_1 : float
        SNR sensitivity [logit per ln(SNR)].
    """

    beta_0: float
    beta_1: float

    def __call__(
        self,
        Q: jnp.ndarray,
        U: jnp.ndarray,
        p: jnp.ndarray,
        albedo: jnp.ndarray,
        sigma_noise: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate P_d.

        Parameters
        ----------
        Q : jnp.ndarray, shape (N,)
            Methane mass flux [kg hr⁻¹].
        U : jnp.ndarray, shape (N,)
            Wind speed [m s⁻¹].
        p : jnp.ndarray, shape (N,)
            Pixel size [m].
        albedo : jnp.ndarray, shape (N,)
            Surface albedo [–].
        sigma_noise : jnp.ndarray, shape (N,)
            Instrument noise [ppb or retrieval units].

        Returns
        -------
        P_d : jnp.ndarray, shape (N,)
            Probability of detection [0, 1].
        """
        x = snr_proxy(Q, U, p, albedo, sigma_noise)
        logit = self.beta_0 + self.beta_1 * x
        return 1.0 / (1.0 + jnp.exp(-logit))

    @classmethod
    def sample_priors(cls) -> "SNRBasedPOD":
        """Sample NumPyro priors.

        Priors
        ------
            β₀ ~ Normal(−2.0, 2.0)   [logit]
            β₁ ~ HalfNormal(2.0)     [logit per ln(SNR)]
        """
        return cls(
            beta_0=numpyro.sample("beta_0", dist.Normal(-2.0, 2.0)),
            beta_1=numpyro.sample("beta_1", dist.HalfNormal(2.0)),
        )


# ═════════════════════════════════════════════════════════════════════════════
# 7. SPECTRAL-AWARE GLM (Hyperspectral vs. Multispectral)
# ═════════════════════════════════════════════════════════════════════════════

class SpectralAwarePOD(eqx.Module):
    """GLM with spectral band count and resolution as covariates.

    Physical Motivation
    -------------------
    Hyperspectral imagers (GHGSat: ~200 bands in SWIR, PRISMA: 250,
    EnMAP: 230) resolve individual CH₄ absorption lines, enabling
    matched-filter retrievals with far lower detection limits than
    multispectral instruments (Sentinel-2: ~13 bands, Landsat: ~11).
    The number of spectral channels in the CH₄-sensitive window
    (1600–1700 nm or 2300–2400 nm) directly governs retrieval precision.

    Spectral resolution Δλ [nm] determines how well absorption lines
    are resolved. Finer resolution → better spectral discrimination →
    lower false alarm rate.

    Equation
    --------
        logit(P_d) = β₀ + β₁ · ln(Q/(U·p)) + β_n · ln(n_bands)
                        + β_Δλ · ln(Δλ) + β_A · A

    Attributes
    ----------
    beta_0 : float
        Intercept [logit].
    beta_proxy : float
        Concentration proxy sensitivity [logit per unit].
    beta_n_bands : float
        Spectral channel count effect [logit per ln(count)].
    beta_spectral_res : float
        Spectral resolution effect [logit per ln(nm)].
    beta_albedo : float
        Albedo effect [logit per unit albedo].
    """

    beta_0: float
    beta_proxy: float
    beta_n_bands: float
    beta_spectral_res: float
    beta_albedo: float

    def __call__(
        self,
        Q: jnp.ndarray,
        U: jnp.ndarray,
        p: jnp.ndarray,
        albedo: jnp.ndarray,
        n_bands: jnp.ndarray,
        spectral_res_nm: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate P_d.

        Parameters
        ----------
        Q : jnp.ndarray, shape (N,)
            Methane mass flux [kg hr⁻¹].
        U : jnp.ndarray, shape (N,)
            Wind speed [m s⁻¹].
        p : jnp.ndarray, shape (N,)
            Pixel size [m].
        albedo : jnp.ndarray, shape (N,)
            Surface albedo [–].
        n_bands : jnp.ndarray, shape (N,)
            Number of spectral channels in CH₄ window [count].
        spectral_res_nm : jnp.ndarray, shape (N,)
            Spectral resolution Δλ [nm].

        Returns
        -------
        P_d : jnp.ndarray, shape (N,)
            Probability of detection [0, 1].
        """
        x = log_concentration_proxy(Q, U, p)
        logit = (
            self.beta_0
            + self.beta_proxy * x
            + self.beta_n_bands * jnp.log(jnp.clip(n_bands, min=1.0))
            + self.beta_spectral_res * jnp.log(jnp.clip(spectral_res_nm, min=0.01))
            + self.beta_albedo * albedo
        )
        return 1.0 / (1.0 + jnp.exp(-logit))

    @classmethod
    def sample_priors(cls) -> "SpectralAwarePOD":
        """Sample NumPyro priors.

        Priors
        ------
            β₀   ~ Normal(−5.0, 3.0)
            β_x  ~ HalfNormal(2.0)
            β_n  ~ HalfNormal(1.0)       (more bands → better)
            β_Δλ ~ Normal(−0.5, 0.5)     (finer resolution → better, negative)
            β_A  ~ HalfNormal(1.0)
        """
        return cls(
            beta_0=numpyro.sample("beta_0", dist.Normal(-5.0, 3.0)),
            beta_proxy=numpyro.sample("beta_proxy", dist.HalfNormal(2.0)),
            beta_n_bands=numpyro.sample("beta_n_bands", dist.HalfNormal(1.0)),
            beta_spectral_res=numpyro.sample("beta_spectral_res", dist.Normal(-0.5, 0.5)),
            beta_albedo=numpyro.sample("beta_albedo", dist.HalfNormal(1.0)),
        )


# ═════════════════════════════════════════════════════════════════════════════
# 8. FULL VARYING-COEFFICIENT GLM (The Kitchen Sink)
# ═════════════════════════════════════════════════════════════════════════════

class FullVaryingCoefficientPOD(eqx.Module):
    """Full varying-coefficient GLM with spectral and radiometric controls.

    Physical Motivation
    -------------------
    The most expressive model in the family. Both the *intercept* and
    the *slope* of the detection curve vary with environmental and
    instrumental state. This captures:

      • **Intercept modifiers** (β₀ terms): direct additive effects on
        the logit baseline — e.g., more spectral bands directly lower
        the false-alarm threshold regardless of source strength.
      • **Slope modifiers** (γ terms): environment-dependent sensitivity
        — e.g., under high albedo, the instrument extracts more signal
        per unit concentration, steepening the POD curve.

    Equation
    --------
        β₁(Z) = γ_base + γ_A · A + γ_θ · cos(θ) + γ_n · ln(n_bands)
        β₀(Z) = β₀_base + β₀_noise · ln(σ_noise)
        logit(P_d) = β₀(Z) + β₁(Z) · ln(Q / (U · p))

    Attributes
    ----------
    beta_0_base : float
        Baseline intercept [logit].
    beta_0_noise : float
        Noise intercept modifier [logit per ln(noise unit)].
    gamma_base : float
        Intrinsic slope [logit per log-proxy-unit].
    gamma_albedo : float
        Albedo slope modifier [logit per log-unit per albedo].
    gamma_cos_theta : float
        Illumination slope modifier [logit per log-unit per cos(θ)].
    gamma_n_bands : float
        Spectral slope modifier [logit per log-unit per ln(bands)].
    """

    beta_0_base: float
    beta_0_noise: float
    gamma_base: float
    gamma_albedo: float
    gamma_cos_theta: float
    gamma_n_bands: float

    def __call__(
        self,
        Q: jnp.ndarray,
        U: jnp.ndarray,
        p: jnp.ndarray,
        albedo: jnp.ndarray,
        cos_theta: jnp.ndarray,
        n_bands: jnp.ndarray,
        sigma_noise: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate P_d.

        Parameters
        ----------
        Q : jnp.ndarray, shape (N,)
            Methane mass flux [kg hr⁻¹].
        U : jnp.ndarray, shape (N,)
            Wind speed [m s⁻¹].
        p : jnp.ndarray, shape (N,)
            Pixel size [m].
        albedo : jnp.ndarray, shape (N,)
            Surface albedo [–].
        cos_theta : jnp.ndarray, shape (N,)
            Cosine of solar zenith angle [–].
        n_bands : jnp.ndarray, shape (N,)
            Spectral channels in CH₄ window [count].
        sigma_noise : jnp.ndarray, shape (N,)
            Instrument noise [ppb or retrieval units].

        Returns
        -------
        P_d : jnp.ndarray, shape (N,)
            Probability of detection [0, 1].
        """
        x = log_concentration_proxy(Q, U, p)

        beta_0 = (
            self.beta_0_base
            + self.beta_0_noise * jnp.log(jnp.clip(sigma_noise, min=1e-6))
        )
        beta_1 = (
            self.gamma_base
            + self.gamma_albedo * albedo
            + self.gamma_cos_theta * cos_theta
            + self.gamma_n_bands * jnp.log(jnp.clip(n_bands, min=1.0))
        )
        logit = beta_0 + beta_1 * x
        return 1.0 / (1.0 + jnp.exp(-logit))

    @classmethod
    def sample_priors(cls) -> "FullVaryingCoefficientPOD":
        """Sample NumPyro priors.

        Priors
        ------
            β₀_base  ~ Normal(−3.5, 2.0)
            β₀_noise ~ Normal(−0.5, 0.5)    (more noise → lower baseline)
            γ_base   ~ HalfNormal(1.5)
            γ_A      ~ HalfNormal(1.0)
            γ_θ      ~ Normal(0.5, 0.5)
            γ_n      ~ HalfNormal(0.5)
        """
        return cls(
            beta_0_base=numpyro.sample("beta_0_base", dist.Normal(-3.5, 2.0)),
            beta_0_noise=numpyro.sample("beta_0_noise", dist.Normal(-0.5, 0.5)),
            gamma_base=numpyro.sample("gamma_base", dist.HalfNormal(1.5)),
            gamma_albedo=numpyro.sample("gamma_albedo", dist.HalfNormal(1.0)),
            gamma_cos_theta=numpyro.sample("gamma_cos_theta", dist.Normal(0.5, 0.5)),
            gamma_n_bands=numpyro.sample("gamma_n_bands", dist.HalfNormal(0.5)),
        )


# ═════════════════════════════════════════════════════════════════════════════
# 9. PROBIT GLM (Gaussian Link Alternative)
# ═════════════════════════════════════════════════════════════════════════════

class ProbitPOD(eqx.Module):
    """Probit GLM on the concentration proxy.

    Physical Motivation
    -------------------
    The probit link Φ(·) (standard normal CDF) arises naturally when the
    latent signal strength Y* = β₀ + β₁·x + ε, with ε ~ N(0,1). A
    detection occurs when Y* > 0. This is the correct model if the
    retrieval noise is Gaussian — which is a reasonable first
    approximation for well-calibrated optimal-estimation retrievals
    (e.g., IMAP-DOAS on TROPOMI, or the CO₂M matched filter).

    The probit and logit links produce nearly identical curves; the
    probit has slightly thinner tails, producing a sharper transition.

    Equation
    --------
        P_d = Φ(β₀ + β₁ · ln(Q / (U · p)))

    where Φ is the standard normal CDF.

    Attributes
    ----------
    beta_0 : float
        Intercept [probit units].
    beta_1 : float
        Slope [probit per log-proxy unit].
    """

    beta_0: float
    beta_1: float

    def __call__(
        self,
        Q: jnp.ndarray,
        U: jnp.ndarray,
        p: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate P_d.

        Parameters
        ----------
        Q : jnp.ndarray, shape (N,)
            Methane mass flux [kg hr⁻¹].
        U : jnp.ndarray, shape (N,)
            Wind speed [m s⁻¹].
        p : jnp.ndarray, shape (N,)
            Pixel size [m].

        Returns
        -------
        P_d : jnp.ndarray, shape (N,)
            Probability of detection [0, 1].
        """
        x = log_concentration_proxy(Q, U, p)
        z = self.beta_0 + self.beta_1 * x
        # Φ(z) via the error function: Φ(z) = 0.5 · (1 + erf(z / √2))
        return 0.5 * (1.0 + _jax_erf(z / jnp.sqrt(2.0)))

    @classmethod
    def sample_priors(cls) -> "ProbitPOD":
        """Sample NumPyro priors.

        Priors
        ------
            β₀ ~ Normal(−2.0, 2.0)   [probit]
            β₁ ~ HalfNormal(1.5)     [probit per unit]
        """
        return cls(
            beta_0=numpyro.sample("beta_0", dist.Normal(-2.0, 2.0)),
            beta_1=numpyro.sample("beta_1", dist.HalfNormal(1.5)),
        )


# ═════════════════════════════════════════════════════════════════════════════
# 10. CLOGLOG GLM (Complementary Log-Log — Asymmetric Extremes)
# ═════════════════════════════════════════════════════════════════════════════

class CloglogPOD(eqx.Module):
    """Complementary log-log GLM for asymmetric detection tails.

    Physical Motivation
    -------------------
    The logistic and probit links are symmetric: the curve approaches 0
    and 1 at the same rate. In practice, P_d often approaches 1 much
    faster than it departs from 0 — once a plume is large enough,
    detection is near-certain, but very small plumes can linger near
    P_d ≈ 0 for a wide range of Q. The cloglog link captures this
    asymmetry naturally.

    Arises from an extreme-value / Gumbel latent variable, appropriate
    when detection is a "first exceedance" event: the retrieval
    algorithm fires when the enhancement exceeds a noise threshold in
    at least one pixel.

    Equation
    --------
        P_d = 1 − exp(−exp(β₀ + β₁ · ln(Q / (U · p))))

    Attributes
    ----------
    beta_0 : float
        Intercept [cloglog units].
    beta_1 : float
        Slope [cloglog per log-proxy unit].
    """

    beta_0: float
    beta_1: float

    def __call__(
        self,
        Q: jnp.ndarray,
        U: jnp.ndarray,
        p: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate P_d.

        Parameters
        ----------
        Q : jnp.ndarray, shape (N,)
            Methane mass flux [kg hr⁻¹].
        U : jnp.ndarray, shape (N,)
            Wind speed [m s⁻¹].
        p : jnp.ndarray, shape (N,)
            Pixel size [m].

        Returns
        -------
        P_d : jnp.ndarray, shape (N,)
            Probability of detection [0, 1].
        """
        x = log_concentration_proxy(Q, U, p)
        eta = self.beta_0 + self.beta_1 * x
        return 1.0 - jnp.exp(-jnp.exp(eta))

    @classmethod
    def sample_priors(cls) -> "CloglogPOD":
        """Sample NumPyro priors.

        Priors
        ------
            β₀ ~ Normal(−3.0, 2.0)   [cloglog]
            β₁ ~ HalfNormal(1.5)     [cloglog per unit]
        """
        return cls(
            beta_0=numpyro.sample("beta_0", dist.Normal(-3.0, 2.0)),
            beta_1=numpyro.sample("beta_1", dist.HalfNormal(1.5)),
        )


# ═════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ═════════════════════════════════════════════════════════════════════════════

POD_REGISTRY = {
    "logistic": LogisticPOD,
    "log_logistic": LogLogisticPOD,
    "concentration_proxy": ConcentrationProxyPOD,
    "additive_multi": AdditiveMultiCovariatePOD,
    "varying_coefficient": VaryingCoefficientPOD,
    "snr_based": SNRBasedPOD,
    "spectral_aware": SpectralAwarePOD,
    "full_varying": FullVaryingCoefficientPOD,
    "probit": ProbitPOD,
    "cloglog": CloglogPOD,
}
"""str → POD Module class mapping for dynamic dispatch."""
