---
title: "Tier V.B — Point process model (TMTPP)"
short_title: "Tier V.B — TMTPP"
subject: "plumax — generative TMTPP foundation"
authors:
  - name: J. Emmanuel Johnson
    affiliations: [UNEP, IMEO, MARS]
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: [TMTPP, marked point process, intensity function, Hill function, POD, lognormal-Pareto, Hawkes, log-Gaussian Cox process, methane_pod]
---

# Tier V.B — Point process model (TMTPP)

**Generative model:** events arrive in time according to an intensity $\lambda(t)$, each carries a mark $Q \sim f(Q)$, and each is observed with probability $P_d(Q; \text{satellite})$. The "thinned marked temporal point process" (TMTPP) is the mathematical object on which everything else at Tier V is built — see {cite:p}`daley2003,daleyVereJones2008` for foundations.

The full mathematical derivation lives in [`methane_pod/notebooks/01_mttpp_theory`](../../../methane_pod/notebooks/01_mttpp_theory.md). This page gives the architectural view: the components, their interfaces, and where they plug into the rest of `plumax`.

:::{tip} Units convention
Everything in SI internally — $Q$ in kg/s, time in seconds. Catalog ingestion ([06a](06a_instantaneous.md)) normalises to SI; rendering layers convert to operational units (t/h, kg/h) on display.
:::

---

(vb-three-components)=
## The three components

(vb-temporal)=
### Temporal — $\lambda(t)$ (events / second)

The intensity function tells you how rapidly events arrive at time $t$. Examples from the catalogue in [`02_intensity_zoo.md`](../../../methane_pod/notebooks/02_intensity_zoo.md):

```{math}
:label: eq-vb-lambda-constant
\lambda(t) \;=\; \lambda_0 \qquad \text{(homogeneous Poisson; baseline)}
```

```{math}
:label: eq-vb-lambda-diurnal
\lambda(t) \;=\; \lambda_0 + A \sin\!\left(\frac{2\pi t}{T_\text{day}} + \varphi\right) \qquad \text{(solar-heated tank cycles)}
```

```{math}
:label: eq-vb-lambda-step
\lambda(t) \;=\; \lambda_0 \cdot \mathbb{1}[t > t_\text{open}] \qquad \text{(valve fails open at known time)}
```

```{math}
:label: eq-vb-lambda-decay
\lambda(t) \;=\; \lambda_0\, \exp(-t/\tau) \qquad \text{(pressure-relief blowdown)}
```

```{math}
:label: eq-vb-lambda-hawkes
\lambda(t) \;=\; \mu + \sum_{t_i < t} \alpha\, \exp\!\bigl(-\beta(t - t_i)\bigr) \qquad \text{(Hawkes; self-exciting, event-driven clustering)}
```

```{math}
:label: eq-vb-lambda-lgcp
\log \lambda(t) \;\sim\; \mathcal{GP}(\mu, K) \qquad \text{(log-Gaussian Cox process; environmentally-driven clustering)}
```

13 deterministic / Hawkes kernels currently implemented in [`methane_pod.intensity`](../../../methane_pod/src/methane_pod/intensity.py); LGCP is the v1.5 next kernel — it's the natural model when clustering is environmental rather than self-exciting.

Each kernel is an `equinox.Module` exposing the same `__call__(t) → λ` and `sample_priors()` interface. Adding a new kernel is a one-file PR.

(vb-marks)=
### Marks — $f(Q)$ (probability density on kg/s)

The mark distribution gives the size of an event conditional on it happening.

```{list-table} Mark families — form and operational fit.
:label: tbl-vb-marks
:header-rows: 1

* - Family
  - Form
  - When to use
* - **Lognormal**
  - $Q \sim \operatorname{LN}(\mu, \sigma^{2})$
  - single-class facility populations
* - **Pure power-law**
  - $f(Q) \propto Q^{-\alpha}$ for $Q > Q_\text{min}$
  - baseline only; over-emphasises the tail
* - **Lognormal-Pareto**
  - $\operatorname{LN}(\mu, \sigma^{2})$ body, $\operatorname{Pareto}(\alpha, Q_\text{break})$ tail
  - **v1 default** — Cusworth 2021 / Sherwin 2024 operational standard
* - **Mixture-of-lognormals**
  - $\sum_k \pi_k \cdot \operatorname{LN}(\mu_k, \sigma_k^{2})$
  - multi-class facility populations (wells + tanks + pipelines)
```

$Q_\text{min}$ (power-law) and $Q_\text{break}$ (lognormal-Pareto) are themselves parameters.

:::{tip} v1 default for $Q_\text{break}$
$Q_\text{break} \sim \operatorname{LogNormal}(\log Q_{\text{break,published}}, 0.5^{2})$ — informative prior from instrument-class detection floor, not a hard constraint. Joint inference is cleaner but adds an identifiability concern with $\alpha$.
:::

The mark distribution is what Tier V actually wants to recover — it's the population-scale answer to "how big are the leaks at this kind of facility?"

(vb-pod)=
### Detection thinning — $P_d(Q; \text{satellite})$ (probability)

Not every event is observed. Each satellite has a probability of detection that depends on the leak size, viewing geometry, surface, and atmospheric state.

The operational form in the methane literature is the **Hill function** (Cusworth 2021, Sherwin 2024):

```{math}
:label: eq-vb-pod-hill
P_d(Q) \;=\; \frac{1}{1 + (Q_{50}/Q)^{k}}
```

where $Q_{50}$ is the leak size at which detection probability is 0.5 and $k$ controls the steepness.

:::{caution} Hill ≠ logistic-on-log-Q
**This is *not* the same as a logistic on $\log Q$** — Hill is rational in $Q$, logistic is sigmoidal in $\log Q$. Doc previously listed "logistic"; the operational convention is Hill. (`methane_pod.pod_functions` currently implements both; the population fitter should default to Hill.)
:::

(vb-pod-calibration)=
#### POD calibration uncertainty — hierarchical prior

Per-instrument controlled-release campaigns (Sherwin et al. 2024) deliver a **posterior** on $(Q_{50}, k)$, not a point. v1 default: hierarchical prior carrying calibration-campaign uncertainty:

```{math}
:label: eq-vb-pod-hier-prior
Q_{50,\text{inst}} \sim \operatorname{LogNormal}(\log Q_{50,\text{pub}}, \sigma^{2}_{Q_{50},\text{pub}}), \qquad
k_\text{inst} \sim \operatorname{LogNormal}(\log k_\text{pub}, \sigma^{2}_{k,\text{pub}})
```

This is the middle ground between (a) hard-coding published values (biased when those are uncertain) and (b) full joint inference (cleanest but identifiability concern with $\lambda$). v2 promotes to joint inference when basin data warrants.

10 POD models currently in [`methane_pod.pod_functions`](../../../methane_pod/src/methane_pod/pod_functions.py), described visually in [`05_pod_gallery`](../../../methane_pod/notebooks/05_pod_gallery.ipynb). Variants:

- **Hill** — operational standard.
- **Varying-coefficient Hill** — $Q_{50} = g(\text{albedo}, \text{SZA}, \text{scene class})$.
- **Spectral-aware** — explicitly carries the SWIR retrieval noise floor as a function of column XCH₄.
- **Full GLM** — generalised linear model with multiple scene covariates.

---

(vb-likelihood)=
## TMTPP likelihood — canonical form

For a set of detected events with per-event posteriors $\{p(Q \mid \text{observation}_i)\}_{i=1}^{n}$ and detection times $\{t_i\}$ over a window $[0, T]$:

```{math}
:label: eq-vb-loglik
\log L(\lambda, f, P_d \mid \text{data}) \;=\;
\sum_{i} \!\Bigl[ \log \lambda(t_i) \;+\; \log\!\!\int P_d(Q)\, L_i(Q)\, f(Q)\, \mathrm{d}Q \Bigr]
\;-\; \int_{0}^{T} \lambda(t) \!\!\int P_d(Q)\, f(Q)\, \mathrm{d}Q\, \mathrm{d}t
```

The first sum scores each detected event under: (a) the temporal intensity at the detection time, and (b) the integrated mark contribution that combines the per-event likelihood with the population mark distribution and the satellite POD. The second integral is the **expected number of events that would have been detected** under the model — subtracts the right amount so the posterior is consistent.

(vb-practical-eval)=
### Practical evaluation

The mark integral $\int P_d(Q)\, L_i(Q)\, f(Q)\, \mathrm{d}Q$ is computed via the importance-weighted Monte Carlo estimator from [06a § Mark likelihood](06a_instantaneous.md#va-mark-likelihood):

```{math}
:label: eq-vb-mark-iw
\int P_d(Q)\, L_i(Q)\, f(Q)\, \mathrm{d}Q \;\approx\; \frac{1}{S}\, \sum_{s} P_d(Q_i^{(s)}) \cdot \frac{f(Q_i^{(s)})}{\pi_\text{per-event}(Q_i^{(s)})}
```

with samples $Q_i^{(s)} \sim p(Q \mid \text{observation}_i)$ and $\pi_\text{per-event}(Q)$ the per-event prior used at Tier I–IV. The $1 / \pi_\text{per-event}$ factor is the importance weight; without it the population fit double-counts the per-event prior.

(vb-point-regime)=
### Point-regime simplification

When per-event posteriors are tightly concentrated ($\operatorname{CV} < 20\%$) and $f$ is smooth on that scale, the importance-weighted MC reduces to the **Point regime** of 06a:

```{math}
:label: eq-vb-loglik-point
\log L_\text{point} \;=\; \sum_{i} \!\Bigl[ \log \lambda(t_i) + \log f(Q_i) + \log P_d(Q_i) \Bigr]
\;-\; \int_{0}^{T} \lambda(t) \!\!\int P_d(Q)\, f(Q)\, \mathrm{d}Q\, \mathrm{d}t
```

This is the form currently implemented in [`methane_pod.fitting.pod_powerlaw_model`](../../../methane_pod/src/methane_pod/fitting.py). It's the **simplification**, not the canonical form — explicit regime selection per [06a § Regime selection rule](06a_instantaneous.md#va-regime-rule) decides when it's safe to use.

(vb-numerical-stability)=
### Numerical stability of the integrated thinned-rate term

The integral $\int P_d(Q)\, f(Q)\, \mathrm{d}Q$ over heavy-tailed $f$ (power-law tail, Pareto) and saturating $P_d$ (Hill) **must not** be evaluated by naive quadrature in linear $Q$ — the heavy tail underflows.

:::{important} Standard fix — log-space Gauss-Hermite quadrature
Hill × LogNormal becomes a tractable polynomial-times-sigmoid in log-space; ≤ 16 nodes give 4-decimal accuracy. For Pareto tails, switch to importance sampling with a Pareto proposal.

**Bug class:** any $\alpha > 2$ power-law silently underestimates the thinned-rate integral with linear quadrature, biasing $\lambda$ low. Add as a hard regression test.
:::

---

(vb-plug-in)=
## Where it plugs into `plumax`

```{list-table} TMTPP inputs.
:label: tbl-vb-inputs
:header-rows: 1

* - Input
  - Source
* - $(t_i, \text{instrument\_id}_i, \text{per-event payload})$ per detection
  - [`06a_instantaneous.md`](06a_instantaneous.md) — Tier V.A adapter
* - Per-event $L_i(Q)$ (samples + $\pi_\text{per-event,logpdf}$)
  - Tiers I–IV inversion + posterior export
* - Per-instrument POD calibration $(Q_{50,\text{pub}}, \sigma_{Q_{50},\text{pub}}, k_\text{pub}, \sigma_{k,\text{pub}})$
  - Sherwin 2024 / Cusworth 2021 / Kamdar IMEO controlled-release campaigns; alternatively joint inference with the population
* - Per-instrument overpass coverage (for the integrated rate)
  - [`06a § Non-detection events`](06a_instantaneous.md#va-detection-floor) — catalog ingest
```

```{list-table} TMTPP outputs.
:label: tbl-vb-outputs
:header-rows: 1

* - Output
  - Consumer
* - Posterior $\lambda(t)$
  - [`06c_persistency.md`](06c_persistency.md) — wait times, dispatch windows
* - Posterior $f(Q)$
  - [`06d_total_emission.md`](06d_total_emission.md) — total mass under POD-thinning correction
* - Per-instrument POD posterior $(Q_{50}, k)_\text{inst}$
  - instrument-design and cross-mission calibration questions; multi-satellite fusion (06d)
* - Joint $(\lambda, f, P_d)$ posterior
  - sensitivity studies, satellite-tasking optimisation
```

---

(vb-population-vs-source)=
## Population vs. per-source — the v1 commitment

The TMTPP fits *aggregate* the population. Two distinct framings:

- **Across-population** (v1 default for inventory accounting): fit one $(\lambda, f, P_d)$ over all sources of a class within a basin/region. $Q$ means "size of an event drawn from this class".
- **Per-source longitudinal** (v1 for persistency forecasting on a known facility): fit one $(\lambda_\text{facility}(t), f_\text{facility}(Q), P_d)$ per facility, with hierarchical shrinkage to the population. $Q$ means "size of an event from this specific facility".

Both have library support; the choice is driven by the scientific question, not by the methodology. Inventory totals (06d) use across-population; dispatch decisions for a known leak history (06c) use per-source.

---

(vb-modules)=
## Module layout

```{list-table} Tier V.B module layout — concern, target module, status.
:label: tbl-vb-modules
:header-rows: 1

* - Concern
  - Module
  - Status
* - Intensity registry — deterministic + Hawkes
  - [`methane_pod.intensity`](../../../methane_pod/src/methane_pod/intensity.py)
  - ✓ (13 kernels)
* - Intensity registry — log-Gaussian Cox process
  - `methane_pod.intensity.lgcp`
  - ☐ — v1.5
* - Mark registry
  - `methane_pod.marks` (currently inline in `fitting`)
  - 🚧 — power-law only; lognormal, lognormal-Pareto, mixture-of-lognormals pending
* - POD models $P_d(\cdot)$ (Hill + variants)
  - [`methane_pod.pod_functions`](../../../methane_pod/src/methane_pod/pod_functions.py)
  - ✓ (10 models)
* - POD time-of-day binning (v1 time-varying POD)
  - `methane_pod.pod_functions.tod_binned`
  - ☐
* - POD continuous $P_d(Q, t)$ (v2)
  - `methane_pod.pod_functions.continuous_t`
  - ☐
* - Hierarchical POD calibration prior
  - `methane_pod.pod_functions.calibration_prior`
  - ☐
* - TMTPP likelihood — point regime
  - [`methane_pod.fitting.pod_powerlaw_model`](../../../methane_pod/src/methane_pod/fitting.py)
  - ✓
* - TMTPP likelihood — full importance-corrected regime
  - `methane_pod.fitting.tmtpp_iw`
  - ☐ — consumes [`population.adapter.importance`](06a_instantaneous.md#va-modules)
* - Numerical integration helpers (log-space Gauss-Hermite, Pareto IS)
  - `methane_pod.fitting.integrate`
  - ☐
* - Hawkes / self-exciting kernel
  - `methane_pod.intensity.hawkes`
  - ☐ — beyond the existing kernels
* - Spatial extension (Cox process)
  - `methane_pod.spatial`
  - ☐ — v2; ties to Tier III's $S(\mathbf{x},t)$
```

---

(vb-validation)=
## Validation strategy

- **Likelihood gradient.** `jax.grad` matches finite differences within tolerance. Cheap unit test.
- **Synthetic recovery — Point regime.** Already in [`06_stationary_numpyro_mcmc`](../../../methane_pod/notebooks/06_stationary_numpyro_mcmc.ipynb) for power-law mark.
- **Synthetic recovery — Full regime with importance correction.** Generate per-event posteriors with a known $\pi_\text{per-event}$, fit population, recover $(\lambda^{*}, f^{*}, P_d^{*})$ within reported posterior. Mirrors the importance-correction round trip from [06a § Validation](06a_instantaneous.md#va-validation).
- **SBC — point.** Across 1000 simulated populations, per-parameter rank statistics uniform.
- **SBC — soft observation.** Same SBC but with the soft-observation layer (per-event posteriors as input). Validates the cross-tier inference end-to-end.
- **Identifiability stress test.** Generate data where $\lambda$ is high but $P_d$ is low (vs. the opposite). Quantitative target: posterior correlation $\operatorname{corr}(\lambda_0, Q_{50}) > 0.5$ when confounded; $< 0.2$ when well-separated. Confirms the model knows what it can't disentangle.
- **Log-space integration test.** Compare log-space Gauss–Hermite to Pareto importance sampling for $\int \mathrm{Hill}(Q) \cdot \mathrm{LN\text{-}Pareto}(Q)\, \mathrm{d}Q$ across $\alpha \in [1.5, 4]$. Linear quadrature should fail loudly past $\alpha > 2$. Catches the silent thinned-rate underflow bug.
- **Hierarchical POD coverage.** With known controlled-release calibration injected as $(Q_{50,\text{pub}}, \sigma_{Q_{50},\text{pub}})$, the hierarchical POD prior should produce $Q_{50}$ posteriors that contain the true value at 95% CI ~95% of the time across simulated basins.

---

(vb-open-questions)=
## Open questions

:::{attention} Cox vs. Hawkes for clustering
Both are v1.5 candidates. Pick by basin diagnostic: event-driven clustering (compressor cycles) → Hawkes; environmentally-driven (weather-window persistence) → LGCP.
:::

:::{attention} Mark / temporal coupling
TMTPP currently assumes marks are i.i.d. given the temporal process. Tier IV's $Q(t)$ stochastic process per source is the right object to lift here — a per-source $Q(t)$ becomes a per-source contribution to the spatial population intensity. Promote when Tier IV $Q(t)$ lands.
:::

:::{attention} Per-instrument POD prior depth
Hierarchical with calibration uncertainty (v1 default) vs. full joint inference (v2). When does the basin data warrant promotion? Probably when the calibration-campaign sample size is too small for the instrument-condition (e.g. high-AOD scenes for EMIT).
:::

:::{attention} Continuous time-varying POD
v1 uses time-of-day bins. When do diurnal cloud / glint patterns warrant continuous $P_d(Q, t)$? Likely needed for sun-glint-sensitive instruments over coastal scenes.
:::

:::{attention} $Q_\text{break}$ / $Q_\text{min}$ joint inference
Currently informative prior from detection floor. Open: when basin data warrants joint inference with the power-law $\alpha$, does identifiability hold? Likely fine when the catalog spans ≥ 1.5 decades in $Q$.
:::

:::{attention} Numerical integration choice
Log-space Gauss–Hermite is the v1 default. Open: when does adaptive quadrature pay off (highly-variable $f$ shapes)?
:::
