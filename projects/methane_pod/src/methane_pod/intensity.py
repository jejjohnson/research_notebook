"""
Methane Source Intensity Functions — Equinox Modules
====================================================

Physically motivated intensity functions λ(t) [events day⁻¹] for CH₄
emission sources observed by satellite remote sensing, packaged as
``equinox.Module`` dataclasses.

Each module bundles:
  • Deterministic parameters as frozen fields.
  • ``__call__(t)`` → λ(t) evaluation, compatible with jit/vmap/grad.
  • ``sample_priors(cls)`` → NumPyro prior factory returning a configured
    instance for MCMC inference.

Convention
----------
  • Time ``t`` always in **days**. Fractional days encode sub-daily resolution.
  • Intensity ``λ(t)`` always in **events day⁻¹**.
  • All modules are immutable pytrees (equinox).

Dependencies: jax, equinox, numpyro
"""

from __future__ import annotations

import jax.numpy as jnp
import equinox as eqx
import numpyro
import numpyro.distributions as dist


# ═════════════════════════════════════════════════════════════════════════════
# 1. CONSTANT (Homogeneous Poisson)
# ═════════════════════════════════════════════════════════════════════════════

class ConstantIntensity(eqx.Module):
    """Constant intensity for passive, continuously leaking infrastructure.

    Physical Motivation
    -------------------
    Abandoned oil/gas wells, corroded subsurface pipelines, orphaned wellheads.
    No external forcing — the infrastructure is under constant, unchanging
    stress producing uncoordinated micro-leaks at a flat background rate.

    Typical sources: abandoned wells (~1–5 events day⁻¹), legacy distribution
    networks, chronic flange leaks.

    Equation
    --------
        λ(t) = λ₀
        Λ(T) = λ₀ · T

    Attributes
    ----------
    lambda_0 : float
        Constant emission rate [events day⁻¹]. Typical: 0.1–10.0.
    """

    lambda_0: float

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate intensity.

        Parameters
        ----------
        t : jnp.ndarray, shape (N,)
            Time coordinates [days].

        Returns
        -------
        λ : jnp.ndarray, shape (N,)
            Intensity [events day⁻¹].
        """
        return jnp.full_like(t, self.lambda_0)

    @classmethod
    def sample_priors(cls) -> "ConstantIntensity":
        """Sample NumPyro priors.

        Priors
        ------
            λ₀ ~ HalfNormal(3.0)

        Returns
        -------
        model : ConstantIntensity
        """
        lambda_0 = numpyro.sample("lambda_0", dist.HalfNormal(3.0))
        return cls(lambda_0=lambda_0)


# ═════════════════════════════════════════════════════════════════════════════
# 2. DIURNAL SINUSOIDAL (Inhomogeneous Poisson — Sub-Daily)
# ═════════════════════════════════════════════════════════════════════════════

class DiurnalSinusoidalIntensity(eqx.Module):
    """Diurnal sinusoidal intensity for thermally driven infrastructure.

    Physical Motivation
    -------------------
    Solar heating during afternoon hours (≈14:00) expands gas volumes inside
    storage tanks and pipeline networks, increasing PRV venting probability.
    Overnight cooling contracts the gas and suppresses emissions.

    Dominant sub-daily pattern for geostationary platforms (MTG-IRS) or
    high-revisit LEO constellations (GHGSat daily, Tanager daily targeted).

    Typical sources: tank farms, active compressor stations, LNG terminals,
    midstream gathering systems.

    Equation
    --------
        λ(t) = max(0, λ₀ + A · sin(2πt − φ))
        φ = 2π · (φ_h / 24)

    Period = 1 day. Over one full cycle: Λ(1) = λ₀.

    Attributes
    ----------
    lambda_0 : float
        Baseline mean rate [events day⁻¹]. Must satisfy λ₀ > A. Typical: 2–8.
    amplitude : float
        Diurnal half-amplitude [events day⁻¹]. Typical: 1–4.
    phase_hours : float
        Hour-of-day of peak intensity [hours]. Typical: 14.0 (2 PM).
    """

    lambda_0: float
    amplitude: float
    phase_hours: float = 14.0

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate intensity.

        Parameters
        ----------
        t : jnp.ndarray, shape (N,)
            Time coordinates [days].

        Returns
        -------
        λ : jnp.ndarray, shape (N,)
            Intensity [events day⁻¹], clamped ≥ 0.
        """
        phi = 2.0 * jnp.pi * (self.phase_hours / 24.0)
        raw = self.lambda_0 + self.amplitude * jnp.sin(2.0 * jnp.pi * t - phi)
        return jnp.clip(raw, min=0.0)

    @classmethod
    def sample_priors(cls) -> "DiurnalSinusoidalIntensity":
        """Sample NumPyro priors.

        Priors
        ------
            λ₀  ~ HalfNormal(5.0)
            A   ~ HalfNormal(3.0)
            φ_h ~ Normal(14.0, 2.0)
        """
        return cls(
            lambda_0=numpyro.sample("lambda_0", dist.HalfNormal(5.0)),
            amplitude=numpyro.sample("amplitude", dist.HalfNormal(3.0)),
            phase_hours=numpyro.sample("phase_hours", dist.Normal(14.0, 2.0)),
        )


# ═════════════════════════════════════════════════════════════════════════════
# 3. SEASONAL SINUSOIDAL (Inhomogeneous Poisson — Annual)
# ═════════════════════════════════════════════════════════════════════════════

class SeasonalSinusoidalIntensity(eqx.Module):
    """Annual sinusoidal intensity for biogenic / climate-driven sources.

    Physical Motivation
    -------------------
    Microbial methanogenesis in saturated soils peaks during warmest, wettest
    months. Dominates at regional/area-flux scales observed by wide-swath
    mappers: TROPOMI (7×5.5 km, daily), MethaneSAT (200 m, ~weekly).

    Typical sources: tropical/boreal wetlands, rice paddies (monsoon Asia),
    permafrost thermokarst lakes, CAFOs.

    Equation
    --------
        λ(t) = max(0, λ₀ + A · sin(2π/365.25 · (t − t_peak)))

    Over one full year: Λ(365.25) = λ₀ · 365.25.

    Attributes
    ----------
    lambda_0 : float
        Annual mean rate [events day⁻¹]. Typical: 0.5–3.0.
    amplitude : float
        Seasonal half-amplitude [events day⁻¹]. Typical: 0.3–2.0.
    peak_day : float
        Day-of-year of emission peak [days from Jan 1]. Typical: 196 (Jul 15).
    """

    lambda_0: float
    amplitude: float
    peak_day: float = 196.0

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate intensity.

        Parameters
        ----------
        t : jnp.ndarray, shape (N,)
            Time coordinates [days].

        Returns
        -------
        λ : jnp.ndarray, shape (N,)
            Intensity [events day⁻¹], clamped ≥ 0.
        """
        ω = 2.0 * jnp.pi / 365.25
        raw = self.lambda_0 + self.amplitude * jnp.sin(ω * (t - self.peak_day))
        return jnp.clip(raw, min=0.0)

    @classmethod
    def sample_priors(cls) -> "SeasonalSinusoidalIntensity":
        """Sample NumPyro priors.

        Priors
        ------
            λ₀      ~ HalfNormal(2.0)
            A       ~ HalfNormal(1.5)
            t_peak  ~ Normal(196.0, 30.0)
        """
        return cls(
            lambda_0=numpyro.sample("lambda_0", dist.HalfNormal(2.0)),
            amplitude=numpyro.sample("amplitude", dist.HalfNormal(1.5)),
            peak_day=numpyro.sample("peak_day", dist.Normal(196.0, 30.0)),
        )


# ═════════════════════════════════════════════════════════════════════════════
# 4. DIURNAL + SEASONAL COMPOUND (Additive Multi-Scale)
# ═════════════════════════════════════════════════════════════════════════════

class DiurnalSeasonalIntensity(eqx.Module):
    """Additive diurnal + seasonal intensity for thermally active biogenic sources.

    Physical Motivation
    -------------------
    Many sources exhibit *both* sub-daily thermal cycling and annual
    modulation. Landfills decompose faster on hot summer afternoons than cold
    winter nights. The aliasing between a satellite's fixed overpass time
    (Sentinel-2 at 10:30 AM, Landsat at 10:00 AM) and the diurnal peak
    systematically biases detection statistics.

    Typical sources: municipal landfills, wastewater treatment plants,
    agricultural biodigesters, tropical rice paddies.

    Equation
    --------
        λ(t) = max(0, λ₀ + A_d·sin(2πt − φ_d) + A_s·sin(ω_s·(t − t_peak)))
        ω_s = 2π/365.25,  φ_d = 2π·(φ_h/24)

    Attributes
    ----------
    lambda_0 : float
        Grand mean intensity [events day⁻¹]. Typical: 1–5.
    amp_diurnal : float
        Diurnal half-amplitude [events day⁻¹].
    amp_seasonal : float
        Seasonal half-amplitude [events day⁻¹].
    phase_hours : float
        Hour of diurnal peak [hours]. Default: 14.0.
    peak_day : float
        Day of seasonal peak [days from Jan 1]. Default: 196.0.
    """

    lambda_0: float
    amp_diurnal: float
    amp_seasonal: float
    phase_hours: float = 14.0
    peak_day: float = 196.0

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate intensity.

        Parameters
        ----------
        t : jnp.ndarray, shape (N,)
            Time coordinates [days].

        Returns
        -------
        λ : jnp.ndarray, shape (N,)
            Intensity [events day⁻¹], clamped ≥ 0.
        """
        φ_d = 2.0 * jnp.pi * (self.phase_hours / 24.0)
        ω_s = 2.0 * jnp.pi / 365.25
        raw = (
            self.lambda_0
            + self.amp_diurnal * jnp.sin(2.0 * jnp.pi * t - φ_d)
            + self.amp_seasonal * jnp.sin(ω_s * (t - self.peak_day))
        )
        return jnp.clip(raw, min=0.0)

    @classmethod
    def sample_priors(cls) -> "DiurnalSeasonalIntensity":
        """Sample NumPyro priors.

        Priors
        ------
            λ₀   ~ HalfNormal(3.0)
            A_d  ~ HalfNormal(2.0)
            A_s  ~ HalfNormal(1.5)
            φ_h  ~ Normal(14.0, 2.0)
            t_pk ~ Normal(196.0, 30.0)
        """
        return cls(
            lambda_0=numpyro.sample("lambda_0", dist.HalfNormal(3.0)),
            amp_diurnal=numpyro.sample("amp_diurnal", dist.HalfNormal(2.0)),
            amp_seasonal=numpyro.sample("amp_seasonal", dist.HalfNormal(1.5)),
            phase_hours=numpyro.sample("phase_hours", dist.Normal(14.0, 2.0)),
            peak_day=numpyro.sample("peak_day", dist.Normal(196.0, 30.0)),
        )


# ═════════════════════════════════════════════════════════════════════════════
# 5. SEASONALLY MODULATED DIURNAL (Multiplicative Coupling)
# ═════════════════════════════════════════════════════════════════════════════

class SeasonallyModulatedDiurnalIntensity(eqx.Module):
    """Diurnal intensity whose amplitude is seasonally modulated.

    Physical Motivation
    -------------------
    Summer days produce much larger thermal swings (ΔT ≈ 20 K) than winter
    (ΔT ≈ 5 K). The diurnal emission peak is 3–4× larger in July than
    January. This *multiplicative* coupling is the dominant pattern at large
    landfills and open-pit coal mines observed from orbiting hyperspectral
    imagers (GHGSat 25 m, Tanager 30 m).

    Equation
    --------
        A_d(t) = (A_sum + A_win)/2 + (A_sum − A_win)/2 · sin(ω_s·(t − t_pk))
        λ(t) = max(0, λ₀ + A_d(t) · sin(2πt − φ_d))

    Attributes
    ----------
    lambda_0 : float
        Grand mean intensity [events day⁻¹].
    amp_summer : float
        Diurnal half-amplitude at seasonal peak [events day⁻¹].
    amp_winter : float
        Diurnal half-amplitude at seasonal trough [events day⁻¹].
    phase_hours : float
        Hour of diurnal peak [hours]. Default: 14.0.
    peak_day : float
        Day of seasonal maximum [days from Jan 1]. Default: 196.0.
    """

    lambda_0: float
    amp_summer: float
    amp_winter: float
    phase_hours: float = 14.0
    peak_day: float = 196.0

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate intensity.

        Parameters
        ----------
        t : jnp.ndarray, shape (N,)
            Time coordinates [days].

        Returns
        -------
        λ : jnp.ndarray, shape (N,)
            Intensity [events day⁻¹], clamped ≥ 0.
        """
        φ_d = 2.0 * jnp.pi * (self.phase_hours / 24.0)
        ω_s = 2.0 * jnp.pi / 365.25
        amp_mean = (self.amp_summer + self.amp_winter) / 2.0
        amp_diff = (self.amp_summer - self.amp_winter) / 2.0
        A_t = amp_mean + amp_diff * jnp.sin(ω_s * (t - self.peak_day))
        raw = self.lambda_0 + A_t * jnp.sin(2.0 * jnp.pi * t - φ_d)
        return jnp.clip(raw, min=0.0)

    @classmethod
    def sample_priors(cls) -> "SeasonallyModulatedDiurnalIntensity":
        """Sample NumPyro priors.

        Priors
        ------
            λ₀      ~ HalfNormal(4.0)
            A_sum   ~ HalfNormal(3.0)
            A_win   ~ HalfNormal(1.0)
            φ_h     ~ Normal(14.0, 2.0)
            t_peak  ~ Normal(196.0, 30.0)
        """
        return cls(
            lambda_0=numpyro.sample("lambda_0", dist.HalfNormal(4.0)),
            amp_summer=numpyro.sample("amp_summer", dist.HalfNormal(3.0)),
            amp_winter=numpyro.sample("amp_winter", dist.HalfNormal(1.0)),
            phase_hours=numpyro.sample("phase_hours", dist.Normal(14.0, 2.0)),
            peak_day=numpyro.sample("peak_day", dist.Normal(196.0, 30.0)),
        )


# ═════════════════════════════════════════════════════════════════════════════
# 6. OPERATIONAL SCHEDULE (Smooth Rectangular On/Off)
# ═════════════════════════════════════════════════════════════════════════════

class OperationalScheduleIntensity(eqx.Module):
    """Step-function intensity for facilities with fixed shift schedules.

    Physical Motivation
    -------------------
    Compressor stations, pump-jacks, and mine ventilation fans operate on
    strict crew shifts. Machinery ON → pressure cycling and mechanical stress
    generate leaks at λ_active. Machinery OFF → passive background at λ_idle.

    Critical for interpreting polar-orbiting satellites with fixed equatorial
    crossing times: Sentinel-2 (10:30), Landsat (10:00), TROPOMI (13:30).
    A satellite that always observes during the active window systematically
    overestimates the 24-hour average.

    Equation
    --------
        λ(t) = λ_idle + (λ_active − λ_idle) · [σ(h − h_start) − σ(h − h_end)]
        h = (t mod 1) · 24,  σ(x) = 1/(1 + e^(−50x))

    Uses smooth sigmoid transitions (k=50) for differentiability in NUTS.

    Attributes
    ----------
    lambda_active : float
        Intensity during operational hours [events day⁻¹]. Typical: 3–15.
    lambda_idle : float
        Intensity during shutdown [events day⁻¹]. Typical: 0.1–1.0.
    start_hour : float
        Start of active window [hours 0–24]. Default: 6.0.
    end_hour : float
        End of active window [hours 0–24]. Default: 18.0.
    """

    lambda_active: float
    lambda_idle: float
    start_hour: float = 6.0
    end_hour: float = 18.0

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate intensity.

        Parameters
        ----------
        t : jnp.ndarray, shape (N,)
            Time coordinates [days].

        Returns
        -------
        λ : jnp.ndarray, shape (N,)
            Intensity [events day⁻¹].
        """
        h = (t % 1.0) * 24.0
        k = 50.0
        duty = (
            1.0 / (1.0 + jnp.exp(-k * (h - self.start_hour)))
            - 1.0 / (1.0 + jnp.exp(-k * (h - self.end_hour)))
        )
        return self.lambda_idle + (self.lambda_active - self.lambda_idle) * duty

    @classmethod
    def sample_priors(cls) -> "OperationalScheduleIntensity":
        """Sample NumPyro priors.

        Priors
        ------
            λ_active ~ HalfNormal(8.0)
            λ_idle   ~ HalfNormal(1.0)
            h_start  ~ Normal(6.0, 1.0)
            h_end    ~ Normal(18.0, 1.0)
        """
        return cls(
            lambda_active=numpyro.sample("lambda_active", dist.HalfNormal(8.0)),
            lambda_idle=numpyro.sample("lambda_idle", dist.HalfNormal(1.0)),
            start_hour=numpyro.sample("start_hour", dist.Normal(6.0, 1.0)),
            end_hour=numpyro.sample("end_hour", dist.Normal(18.0, 1.0)),
        )


# ═════════════════════════════════════════════════════════════════════════════
# 7. WEIBULL RENEWAL (Pressure-Relief Valve Recharge)
# ═════════════════════════════════════════════════════════════════════════════

class WeibullRenewalIntensity(eqx.Module):
    """Weibull hazard function for pressure-driven, recharging sources.

    Physical Motivation
    -------------------
    PRVs on LNG tanks and high-pressure wellheads don't fail randomly. After
    venting, pressure drops to zero and must physically *recharge*. The hazard
    starts at zero and climbs — a textbook increasing failure rate (IFR).

    This is a **Renewal Process**: intensity depends on elapsed time since the
    most recent event (Δt), not calendar time. Generalises the memoryless
    Exponential (k=1) to IFR (k>1) or DFR (k<1).

    Observable from any high-revisit platform (GHGSat, Tanager, geostationary).

    Equation
    --------
        h(Δt) = (k/η) · (Δt/η)^(k−1)

    k > 1 → hazard increases (pressure recharge).
    k = 1 → constant hazard (memoryless Exponential).

    Attributes
    ----------
    scale : float
        Weibull scale η [days]. Typical: 0.05–2.0.
    shape : float
        Weibull shape k [dimensionless]. Typical: 2.0–5.0.

    Notes
    -----
    ``__call__`` takes ``t_since_last`` (Δt), not absolute time.
    """

    scale: float
    shape: float

    def __call__(self, t_since_last: jnp.ndarray) -> jnp.ndarray:
        """Evaluate hazard rate.

        Parameters
        ----------
        t_since_last : jnp.ndarray, shape (N,)
            Elapsed time since last event [days]. Must be > 0.

        Returns
        -------
        h : jnp.ndarray, shape (N,)
            Instantaneous hazard [events day⁻¹].
        """
        Δt = jnp.clip(t_since_last, min=1e-8)
        k, η = self.shape, self.scale
        return (k / η) * (Δt / η) ** (k - 1.0)

    @classmethod
    def sample_priors(cls) -> "WeibullRenewalIntensity":
        """Sample NumPyro priors.

        Priors
        ------
            η ~ HalfNormal(0.5)     [days]
            k ~ LogNormal(1.0, 0.5) [dimensionless]
        """
        return cls(
            scale=numpyro.sample("weibull_scale", dist.HalfNormal(0.5)),
            shape=numpyro.sample("weibull_shape", dist.LogNormal(1.0, 0.5)),
        )


# ═════════════════════════════════════════════════════════════════════════════
# 8. PERIODIC BATCH VENTING (Impulse Train)
# ═════════════════════════════════════════════════════════════════════════════

class PeriodicBatchIntensity(eqx.Module):
    """Periodic narrow-pulse intensity for scheduled batch venting.

    Physical Motivation
    -------------------
    Glycol dehydrators flash accumulated CH₄ every 4–8 hours. Tank batteries
    undergo liquid unloading every 12–48 hours producing sharp spikes lasting
    minutes. Coal mine longwall shearers expose fresh seam every 6–12 hours.

    Between batches, only minor fugitive leaks persist. The result is a
    comb-like Gaussian pulse train on a low background.

    Observable primarily by high-revisit tasked satellites (GHGSat, Tanager)
    or continuous ground-based sensors.

    Equation
    --------
        λ(t) = λ_bg + (λ_pk − λ_bg) · exp(−((t mod P) − P/2)² / (2·(f·P)²))

    Attributes
    ----------
    lambda_background : float
        Inter-batch fugitive rate [events day⁻¹]. Typical: 0.1–1.0.
    lambda_peak : float
        Peak rate during a batch vent [events day⁻¹]. Typical: 10–50.
    period_days : float
        Time between successive batches [days]. Typical: 0.2–2.0.
    duty_fraction : float
        Pulse width as fraction of period [dimensionless]. Default: 0.05.
    """

    lambda_background: float
    lambda_peak: float
    period_days: float
    duty_fraction: float = 0.05

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate intensity.

        Parameters
        ----------
        t : jnp.ndarray, shape (N,)
            Time coordinates [days].

        Returns
        -------
        λ : jnp.ndarray, shape (N,)
            Intensity [events day⁻¹].
        """
        σ = self.duty_fraction * self.period_days
        phase = (t % self.period_days) - (self.period_days / 2.0)
        pulse = jnp.exp(-0.5 * (phase / σ) ** 2)
        return self.lambda_background + (self.lambda_peak - self.lambda_background) * pulse

    @classmethod
    def sample_priors(cls) -> "PeriodicBatchIntensity":
        """Sample NumPyro priors.

        Priors
        ------
            λ_bg  ~ HalfNormal(1.0)
            λ_pk  ~ HalfNormal(20.0)
            P     ~ LogNormal(ln(0.5), 0.5)   [days]
            f     ~ Beta(2, 20)               [dimensionless]
        """
        return cls(
            lambda_background=numpyro.sample("lambda_bg", dist.HalfNormal(1.0)),
            lambda_peak=numpyro.sample("lambda_peak", dist.HalfNormal(20.0)),
            period_days=numpyro.sample("period_days", dist.LogNormal(jnp.log(0.5), 0.5)),
            duty_fraction=numpyro.sample("duty_fraction", dist.Beta(2.0, 20.0)),
        )


# ═════════════════════════════════════════════════════════════════════════════
# 9. COAL MINE VENTILATION (Shift + Weekly Maintenance)
# ═════════════════════════════════════════════════════════════════════════════

class CoalMineVentilationIntensity(eqx.Module):
    """Multi-scale operational intensity for underground coal mines.

    Physical Motivation
    -------------------
    Underground coal mines (>40 Mt CH₄ yr⁻¹ globally) exhibit three regimes:
      1. Extraction shifts (06–22, Mon–Sat): longwall shearer advance exposes
         fresh seam, releasing adsorbed CH₄. Fans at full capacity.
      2. Night idle (22–06): reduced crew, fans at minimum, slower desorption.
      3. Weekly maintenance (Sunday): extraction halted, CH₄ builds in panels.

    Observable by TROPOMI (daily, 7 km — integrates mine complex), MethaneSAT
    (weekly, 200 m — resolves vent shafts), GHGSat/Tanager (shaft-level).

    Equation
    --------
        maintenance_gate = exp(−(d_week − d_maint)² / (2·0.3²))
        shift_gate = σ(h − h_start) − σ(h − h_end)
        λ(t) = gate_m · λ_maint + (1 − gate_m) · [gate_s · λ_ext + (1 − gate_s) · λ_idle]

    Attributes
    ----------
    lambda_extraction : float
        Active extraction rate [events day⁻¹]. Typical: 5–20.
    lambda_maintenance : float
        Maintenance shutdown rate [events day⁻¹]. Typical: 0.5–2.
    lambda_idle : float
        Nightly idle rate [events day⁻¹]. Typical: 1–4.
    shift_start : float
        Shift start [hours]. Default: 6.0.
    shift_end : float
        Shift end [hours]. Default: 22.0.
    maintenance_day : float
        Day of 7-day week for maintenance [0=Mon, 6=Sun]. Default: 6.0.
    """

    lambda_extraction: float
    lambda_maintenance: float
    lambda_idle: float
    shift_start: float = 6.0
    shift_end: float = 22.0
    maintenance_day: float = 6.0

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate intensity.

        Parameters
        ----------
        t : jnp.ndarray, shape (N,)
            Time coordinates [days].

        Returns
        -------
        λ : jnp.ndarray, shape (N,)
            Intensity [events day⁻¹].
        """
        h = (t % 1.0) * 24.0
        d = t % 7.0
        k = 50.0
        shift = (
            1.0 / (1.0 + jnp.exp(-k * (h - self.shift_start)))
            - 1.0 / (1.0 + jnp.exp(-k * (h - self.shift_end)))
        )
        maint = jnp.exp(-0.5 * ((d - self.maintenance_day) / 0.3) ** 2)
        return (
            maint * self.lambda_maintenance
            + (1.0 - maint) * (
                shift * self.lambda_extraction + (1.0 - shift) * self.lambda_idle
            )
        )

    @classmethod
    def sample_priors(cls) -> "CoalMineVentilationIntensity":
        """Sample NumPyro priors.

        Priors
        ------
            λ_ext   ~ HalfNormal(10.0)
            λ_maint ~ HalfNormal(1.5)
            λ_idle  ~ HalfNormal(3.0)
        """
        return cls(
            lambda_extraction=numpyro.sample("lambda_extraction", dist.HalfNormal(10.0)),
            lambda_maintenance=numpyro.sample("lambda_maintenance", dist.HalfNormal(1.5)),
            lambda_idle=numpyro.sample("lambda_idle", dist.HalfNormal(3.0)),
        )


# ═════════════════════════════════════════════════════════════════════════════
# 10. LANDFILL (Seasonal + Diurnal + Barometric Pump)
# ═════════════════════════════════════════════════════════════════════════════

class LandfillIntensity(eqx.Module):
    """Multi-component intensity for municipal solid waste landfills.

    Physical Motivation
    -------------------
    Three superimposed physical mechanisms:
      1. Seasonal temperature (annual): soil T governs microbial activity.
         Summer peak → 2–3× winter rates.
      2. Diurnal thermal (daily): solar heating warms cover soil, enhancing
         gas diffusion. Peak ≈ 15:00 local.
      3. Barometric pumping (synoptic, ~3–7 days): passing low-pressure
         systems literally suck CH₄ out through the cover soil.

    Observable by all satellite tiers: TROPOMI (regional), MethaneSAT
    (complex-scale), GHGSat/Tanager (cell-level).

    Equation
    --------
        λ(t) = max(0, λ₀ + A_s·sin(ω_s(t−t_pk)) + A_d·sin(ω_d·t−φ_d) + A_b·sin(ω_b·t))
        ω_s=2π/365.25, ω_d=2π, ω_b=2π/P_baro

    Attributes
    ----------
    lambda_0 : float
        Annual mean [events day⁻¹]. Typical: 2–6.
    amp_seasonal : float
        Seasonal half-amplitude [events day⁻¹]. Typical: 1–3.
    amp_diurnal : float
        Diurnal half-amplitude [events day⁻¹]. Typical: 0.5–2.
    amp_barometric : float
        Barometric pumping half-amplitude [events day⁻¹]. Typical: 0.5–1.5.
    baro_period_days : float
        Synoptic weather period [days]. Default: 5.0.
    peak_day : float
        Seasonal peak [days from Jan 1]. Default: 196.0.
    phase_hours : float
        Diurnal peak hour [hours]. Default: 15.0.
    """

    lambda_0: float
    amp_seasonal: float
    amp_diurnal: float
    amp_barometric: float
    baro_period_days: float = 5.0
    peak_day: float = 196.0
    phase_hours: float = 15.0

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate intensity.

        Parameters
        ----------
        t : jnp.ndarray, shape (N,)
            Time coordinates [days].

        Returns
        -------
        λ : jnp.ndarray, shape (N,)
            Intensity [events day⁻¹], clamped ≥ 0.
        """
        ω_s = 2.0 * jnp.pi / 365.25
        ω_d = 2.0 * jnp.pi
        ω_b = 2.0 * jnp.pi / self.baro_period_days
        φ_d = 2.0 * jnp.pi * (self.phase_hours / 24.0)
        raw = (
            self.lambda_0
            + self.amp_seasonal * jnp.sin(ω_s * (t - self.peak_day))
            + self.amp_diurnal * jnp.sin(ω_d * t - φ_d)
            + self.amp_barometric * jnp.sin(ω_b * t)
        )
        return jnp.clip(raw, min=0.0)

    @classmethod
    def sample_priors(cls) -> "LandfillIntensity":
        """Sample NumPyro priors.

        Priors
        ------
            λ₀      ~ HalfNormal(4.0)
            A_s     ~ HalfNormal(2.0)
            A_d     ~ HalfNormal(1.5)
            A_b     ~ HalfNormal(1.0)
            P_baro  ~ LogNormal(ln5, 0.3)
            t_peak  ~ Normal(196, 20)
            φ_h     ~ Normal(15, 1.5)
        """
        return cls(
            lambda_0=numpyro.sample("lambda_0", dist.HalfNormal(4.0)),
            amp_seasonal=numpyro.sample("amp_seasonal", dist.HalfNormal(2.0)),
            amp_diurnal=numpyro.sample("amp_diurnal", dist.HalfNormal(1.5)),
            amp_barometric=numpyro.sample("amp_barometric", dist.HalfNormal(1.0)),
            baro_period_days=numpyro.sample("baro_period_days", dist.LogNormal(jnp.log(5.0), 0.3)),
            peak_day=numpyro.sample("peak_day", dist.Normal(196.0, 20.0)),
            phase_hours=numpyro.sample("phase_hours", dist.Normal(15.0, 1.5)),
        )


# ═════════════════════════════════════════════════════════════════════════════
# 11. OFFSHORE PLATFORM (Tidal + Operational)
# ═════════════════════════════════════════════════════════════════════════════

class OffshorePlatformIntensity(eqx.Module):
    """Combined tidal and operational intensity for offshore oil & gas platforms.

    Physical Motivation
    -------------------
    Semi-diurnal tides (M2: ~12.42 hr) modulate subsurface hydrostatic
    pressure, altering gas migration through sediments. Crew shift schedules
    simultaneously drive compressor cycling and flaring. These two independent
    forcings superimpose.

    Observable from Sentinel-2 (2–5 day, flare detection) and GHGSat/Tanager
    (hyperspectral, fugitive plume quantification).

    Equation
    --------
        λ(t) = max(0, λ₀ + A_tide·sin(2π/P_tide · t) + A_ops·[σ(h−h_on)−σ(h−h_off)])

    Attributes
    ----------
    lambda_0 : float
        Baseline platform rate [events day⁻¹]. Typical: 3–8.
    amp_tidal : float
        Tidal half-amplitude [events day⁻¹]. Typical: 0.5–2.
    amp_operational : float
        Crew-shift half-amplitude [events day⁻¹]. Typical: 1–4.
    tidal_period_days : float
        Dominant tidal period [days]. Default: 0.5175 (M2 ≈ 12.42 hr).
    shift_start : float
        Day crew start [hours]. Default: 7.0.
    shift_end : float
        Day crew end [hours]. Default: 19.0.
    """

    lambda_0: float
    amp_tidal: float
    amp_operational: float
    tidal_period_days: float = 0.5175
    shift_start: float = 7.0
    shift_end: float = 19.0

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate intensity.

        Parameters
        ----------
        t : jnp.ndarray, shape (N,)
            Time coordinates [days].

        Returns
        -------
        λ : jnp.ndarray, shape (N,)
            Intensity [events day⁻¹], clamped ≥ 0.
        """
        h = (t % 1.0) * 24.0
        k = 50.0
        crew = (
            1.0 / (1.0 + jnp.exp(-k * (h - self.shift_start)))
            - 1.0 / (1.0 + jnp.exp(-k * (h - self.shift_end)))
        )
        raw = (
            self.lambda_0
            + self.amp_tidal * jnp.sin(2.0 * jnp.pi / self.tidal_period_days * t)
            + self.amp_operational * crew
        )
        return jnp.clip(raw, min=0.0)

    @classmethod
    def sample_priors(cls) -> "OffshorePlatformIntensity":
        """Sample NumPyro priors.

        Priors
        ------
            λ₀     ~ HalfNormal(5.0)
            A_tide ~ HalfNormal(1.5)
            A_ops  ~ HalfNormal(2.0)
        """
        return cls(
            lambda_0=numpyro.sample("lambda_0", dist.HalfNormal(5.0)),
            amp_tidal=numpyro.sample("amp_tidal", dist.HalfNormal(1.5)),
            amp_operational=numpyro.sample("amp_operational", dist.HalfNormal(2.0)),
        )


# ═════════════════════════════════════════════════════════════════════════════
# 12. WETLAND / PERMAFROST (Sigmoid-Gated Seasonal)
# ═════════════════════════════════════════════════════════════════════════════

class WetlandPermafrostIntensity(eqx.Module):
    """Sigmoid-gated seasonal intensity for high-latitude wetland/permafrost.

    Physical Motivation
    -------------------
    Arctic/boreal wetlands (~200 Mt CH₄ yr⁻¹) are gated by a binary physical
    constraint: soil must be thawed and saturated for methanogenesis. When the
    active layer freezes (Oct–Apr), emissions collapse to near zero. Thaw
    onset (May) ramps emissions over ~2–4 weeks. This sharp on/off is
    fundamentally different from a smooth sinusoid.

    Observable by TROPOMI (daily, 7 km — diffuse wetland complexes),
    MethaneSAT (weekly, 200 m — thermokarst lake clusters), GOSAT (~3 day).

    Equation
    --------
        G_thaw(t)  = σ((t_yr − t_thaw) / w)
        G_freeze(t) = 1 − σ((t_yr − t_freeze) / w)
        λ(t) = λ_frozen + (λ_peak − λ_frozen) · G_thaw · G_freeze
        t_yr = t mod 365.25

    Attributes
    ----------
    lambda_peak : float
        Peak summer rate [events day⁻¹]. Typical: 2–8.
    lambda_frozen : float
        Winter baseline (residual ebullition) [events day⁻¹]. Typical: 0.01–0.3.
    thaw_onset_day : float
        Day-of-year of thaw onset [days]. Default: 120 (≈ May 1).
    freeze_onset_day : float
        Day-of-year of freeze onset [days]. Default: 280 (≈ Oct 7).
    transition_width : float
        Sigmoid transition width [days]. Default: 15.0.
    """

    lambda_peak: float
    lambda_frozen: float
    thaw_onset_day: float = 120.0
    freeze_onset_day: float = 280.0
    transition_width: float = 15.0

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate intensity.

        Parameters
        ----------
        t : jnp.ndarray, shape (N,)
            Time coordinates [days].

        Returns
        -------
        λ : jnp.ndarray, shape (N,)
            Intensity [events day⁻¹].
        """
        t_yr = t % 365.25
        w = self.transition_width
        g_thaw = 1.0 / (1.0 + jnp.exp(-(t_yr - self.thaw_onset_day) / w))
        g_freeze = 1.0 - 1.0 / (1.0 + jnp.exp(-(t_yr - self.freeze_onset_day) / w))
        return self.lambda_frozen + (self.lambda_peak - self.lambda_frozen) * g_thaw * g_freeze

    @classmethod
    def sample_priors(cls) -> "WetlandPermafrostIntensity":
        """Sample NumPyro priors.

        Priors
        ------
            λ_peak   ~ HalfNormal(5.0)
            λ_frozen ~ HalfNormal(0.2)
            t_thaw   ~ Normal(120, 15)
            t_freeze ~ Normal(280, 15)
            w        ~ HalfNormal(15.0)
        """
        return cls(
            lambda_peak=numpyro.sample("lambda_peak", dist.HalfNormal(5.0)),
            lambda_frozen=numpyro.sample("lambda_frozen", dist.HalfNormal(0.2)),
            thaw_onset_day=numpyro.sample("thaw_onset_day", dist.Normal(120.0, 15.0)),
            freeze_onset_day=numpyro.sample("freeze_onset_day", dist.Normal(280.0, 15.0)),
            transition_width=numpyro.sample("transition_width", dist.HalfNormal(15.0)),
        )


# ═════════════════════════════════════════════════════════════════════════════
# 13. LIVESTOCK FEEDLOT (Bimodal Feeding + Seasonal)
# ═════════════════════════════════════════════════════════════════════════════

class LivestockFeedlotIntensity(eqx.Module):
    """Bimodal diurnal + seasonal intensity for concentrated livestock operations.

    Physical Motivation
    -------------------
    Enteric fermentation in cattle (~100–120 kg CH₄ cow⁻¹ yr⁻¹) peaks 1–3
    hours after feeding as rumen microbial activity surges. Large CAFOs have
    fixed AM/PM feeding schedules → characteristic bimodal diurnal pattern.
    Seasonal modulation from temperature-dependent manure lagoon emissions
    and feed composition changes.

    Observable at facility scale by GHGSat (25 m), Tanager (30 m), and
    regionally by TROPOMI and MethaneSAT.

    Equation
    --------
        G_am(t) = exp(−(h − h_am)² / (2w²))
        G_pm(t) = exp(−(h − h_pm)² / (2w²))
        λ(t) = max(0, λ₀ + A_f·(G_am + G_pm) + A_s·sin(ω_s(t − t_peak)))
        h = (t mod 1)·24

    Attributes
    ----------
    lambda_0 : float
        Baseline enteric rate [events day⁻¹]. Typical: 1–4.
    amp_feeding : float
        Post-feeding surge amplitude [events day⁻¹]. Typical: 1–3.
    amp_seasonal : float
        Seasonal half-amplitude [events day⁻¹]. Typical: 0.5–1.5.
    morning_feed_hour : float
        AM feed time [hours]. Default: 7.0.
    evening_feed_hour : float
        PM feed time [hours]. Default: 17.0.
    feed_response_width : float
        Gaussian width of feeding pulse [hours]. Default: 2.0.
    peak_day : float
        Seasonal peak [days from Jan 1]. Default: 196.0.
    """

    lambda_0: float
    amp_feeding: float
    amp_seasonal: float
    morning_feed_hour: float = 7.0
    evening_feed_hour: float = 17.0
    feed_response_width: float = 2.0
    peak_day: float = 196.0

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate intensity.

        Parameters
        ----------
        t : jnp.ndarray, shape (N,)
            Time coordinates [days].

        Returns
        -------
        λ : jnp.ndarray, shape (N,)
            Intensity [events day⁻¹], clamped ≥ 0.
        """
        h = (t % 1.0) * 24.0
        w = self.feed_response_width
        g_am = jnp.exp(-0.5 * ((h - self.morning_feed_hour) / w) ** 2)
        g_pm = jnp.exp(-0.5 * ((h - self.evening_feed_hour) / w) ** 2)
        seasonal = self.amp_seasonal * jnp.sin(2.0 * jnp.pi / 365.25 * (t - self.peak_day))
        raw = self.lambda_0 + self.amp_feeding * (g_am + g_pm) + seasonal
        return jnp.clip(raw, min=0.0)

    @classmethod
    def sample_priors(cls) -> "LivestockFeedlotIntensity":
        """Sample NumPyro priors.

        Priors
        ------
            λ₀   ~ HalfNormal(3.0)
            A_f  ~ HalfNormal(2.0)
            A_s  ~ HalfNormal(1.0)
            h_am ~ Normal(7.0, 1.0)
            h_pm ~ Normal(17.0, 1.0)
            w    ~ HalfNormal(2.0)
        """
        return cls(
            lambda_0=numpyro.sample("lambda_0", dist.HalfNormal(3.0)),
            amp_feeding=numpyro.sample("amp_feeding", dist.HalfNormal(2.0)),
            amp_seasonal=numpyro.sample("amp_seasonal", dist.HalfNormal(1.0)),
            morning_feed_hour=numpyro.sample("morning_feed_hour", dist.Normal(7.0, 1.0)),
            evening_feed_hour=numpyro.sample("evening_feed_hour", dist.Normal(17.0, 1.0)),
            feed_response_width=numpyro.sample("feed_response_width", dist.HalfNormal(2.0)),
        )


# ═════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ═════════════════════════════════════════════════════════════════════════════

INTENSITY_REGISTRY = {
    "constant": ConstantIntensity,
    "diurnal": DiurnalSinusoidalIntensity,
    "seasonal": SeasonalSinusoidalIntensity,
    "diurnal_seasonal": DiurnalSeasonalIntensity,
    "modulated_diurnal": SeasonallyModulatedDiurnalIntensity,
    "operational": OperationalScheduleIntensity,
    "weibull_renewal": WeibullRenewalIntensity,
    "periodic_batch": PeriodicBatchIntensity,
    "coal_mine": CoalMineVentilationIntensity,
    "landfill": LandfillIntensity,
    "offshore": OffshorePlatformIntensity,
    "wetland_permafrost": WetlandPermafrostIntensity,
    "livestock_feedlot": LivestockFeedlotIntensity,
}
"""str → Module class mapping for dynamic dispatch and configuration files."""
