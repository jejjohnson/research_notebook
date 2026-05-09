---
title: "Tier V.C — Persistency"
short_title: "Tier V.C — Persistency"
subject: "plumax — operational forecasting layer"
authors:
  - name: J. Emmanuel Johnson
    affiliations: [UNEP, IMEO, MARS]
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: [persistency, wait time, occurrence probability, dispatch, LDAR, Hawkes, methane forecast]
---

# Tier V.C — Persistency

**Question:** Given the inverted intensity $\lambda(t)$ from Tier V.B, when will the next emission event happen, and what's the probability of an event during a specified window?

This is the **operational layer** — what an LDAR (Leak Detection and Repair) crew or a satellite-tasking dispatcher actually consumes. The full derivations of each metric live in [`methane_pod/notebooks/08_persistency`](../../../methane_pod/notebooks/08_persistency.md); this page summarises the metrics and how they slot into the `plumax` API.

---

(vc-four-metrics)=
## The four operational metrics

(vc-wait-time)=
### 1. Expected wait time $\mathbb{E}[\Delta t \mid t_0]$

How long after time $t_0$ until the next event?

```{math}
:label: eq-vc-wait-homogeneous
\mathbb{E}[\Delta t] \;=\; \frac{1}{\lambda_0} \qquad \text{(homogeneous Poisson; memoryless)}
```

```{math}
:label: eq-vc-wait-inhomogeneous
\mathbb{E}[\Delta t \mid t_0] \;=\; \int_{t_0}^{\infty} \exp\!\left(-\int_{t_0}^{t} \lambda(u)\, \mathrm{d}u\right) \mathrm{d}t \qquad \text{(inhomogeneous; depends on starting clock)}
```

For a diurnal source, vastly different at noon vs. midnight.

**Operational use.** Dispatch decisions: arrive during a high-$\lambda$ window and the next event is imminent (worth waiting); arrive during a low-$\lambda$ window and you'd waste hours. Drives MARS-style dispatch suppression during dormant cycles.

(vc-occurrence)=
### 2. Probability of occurrence $\mathbb{P}\!\bigl(N(t_1, t_2) \geq 1\bigr)$

What's the chance of at least one event in $[t_1, t_2]$?

```{math}
:label: eq-vc-occurrence-homogeneous
\mathbb{P}\!\bigl(N(t_1, t_2) \geq 1\bigr) \;=\; 1 - \exp\!\bigl(-\lambda_0\, (t_2 - t_1)\bigr) \qquad \text{(homogeneous)}
```

```{math}
:label: eq-vc-occurrence-inhomogeneous
\mathbb{P}\!\bigl(N(t_1, t_2) \geq 1\bigr) \;=\; 1 - \exp\!\left(-\int_{t_1}^{t_2} \lambda(t)\, \mathrm{d}t\right) \qquad \text{(inhomogeneous)}
```

**Operational use.** "Wrench-turning" probability. If a maintenance window is 4 hours, what's the chance the leak shows itself during that window? Drives whether to schedule the visit.

(vc-conditional-intensity)=
### 3. Conditional intensity given prior detection $\lambda(t \mid t_\text{prev})$

For a source with a known recent detection at $t_\text{prev}$, what's the posterior intensity going forward?

For Poisson processes (no memory): unchanged. For Hawkes / self-exciting processes: bumped —

```{math}
:label: eq-vc-hawkes-bump
\lambda(t \mid t_\text{prev}) \;=\; \mu + \alpha\, \exp\!\bigl(-\beta(t - t_\text{prev})\bigr)
```

— captures the empirical observation that super-emitters "cluster".

**Operational use.** Prioritisation: a source with a recent detection is *more* likely to repeat-emit in the next 24 h. Re-task a high-resolution satellite (GHGSat {cite:p}`ghgsat`, Carbon Mapper {cite:p}`carbon_mapper`) on top of a TROPOMI alert {cite:p}`s5p_tropomi`.

(vc-cumulative-count)=
### 4. Cumulative event count $\mathbb{E}[N(0, T)]$ and credible bounds

Expected number of events in $[0, T]$, with credible interval from the posterior on $\lambda$.

```{math}
:label: eq-vc-cumulative
\mathbb{E}[N(0, T)] \;=\; \Lambda(T) \;=\; \int_{0}^{T} \lambda(t)\, \mathrm{d}t
```

(Homogeneous: $\lambda_0 \cdot T$.)

**Operational use.** Annual reporting, regulatory compliance. "How many emission events should we expect this year at this facility class, with 95% credible interval?"

---

(vc-api)=
## API shape

A thin wrapper around `methane_pod.intensity`:

```python
from plume_simulation.population.persistency import (
    expected_wait_time,
    occurrence_probability,
    cumulative_count,
    next_event_quantile,
)

# Inputs: posterior samples of intensity parameters (from Tier V.B fit)
# Outputs: posterior samples of the operational metric

E_wait = expected_wait_time(intensity, t0=18.0, posterior_samples=mcmc.get_samples())
# → array of shape (n_samples,) in [hours]

P_occur = occurrence_probability(intensity, t1=8.0, t2=12.0,
                                 posterior_samples=mcmc.get_samples())
# → array of shape (n_samples,) in [0, 1]
```

The metric functions take an intensity callable (any of the 13 `equinox` modules from [`methane_pod.intensity`](../../../methane_pod/src/methane_pod/intensity.py)), a query window, and a posterior sample of the intensity's parameters. They return posterior samples of the metric — full UQ propagation, no point estimates.

---

(vc-modules)=
## Module layout

```{list-table} Tier V.C module layout — concern, target module, status.
:label: tbl-vc-modules
:header-rows: 1

* - Concern
  - Module
  - Status
* - Intensity functions
  - [`methane_pod.intensity`](../../../methane_pod/src/methane_pod/intensity.py)
  - ✓
* - Wait-time / occurrence / cumulative metrics
  - `plume_simulation.population.persistency`
  - ☐
* - Posterior-aware metric wrappers
  - same module
  - ☐
* - Operational dashboard / report templates
  - out of scope for `plumax`; lives in `plumax-deploy` (future)
  - —
```

The integral over $\lambda(t)$ in the wait-time formula is closed-form for a few intensity choices (constant, exponential decay) and otherwise needs `jax.scipy.integrate` or a fixed quadrature. Worth wrapping once and reusing across metrics.

---

(vc-validation)=
## Validation strategy

- **Homogeneous limit.** For constant $\lambda$, all four metrics have closed-form formulas; the implementation should match to machine precision.
- **MC self-consistency.** Sample $n$ event times from a known $\lambda(t)$ via thinning, compute the empirical wait time / occurrence frequency, compare to the closed-form metric. Tests both the metric implementation and the simulator.
- **Posterior coverage.** For a synthetic source with known $\lambda_\text{true}(t)$, the 95% credible interval on $\mathbb{E}[\Delta t]$ should contain the truth ~95% of the time across replicates.
- **Diurnal sanity.** A solar-heated tank with peak $\lambda$ at 14:00 should have $\mathbb{E}[\Delta t \mid 14{:}00] \ll \mathbb{E}[\Delta t \mid 02{:}00]$. Numerical sanity check, not a formal test, but catches sign errors.

---

(vc-open-questions)=
## Open questions

:::{attention} What's "the" intensity?
A point estimate (posterior mean) or the full posterior over $\lambda$ parameters? Operational dashboards may want the former; researchers want the latter. The API returns posterior samples by default; downstream summarisation is the caller's choice.
:::

:::{attention} Hawkes vs Poisson default
Hawkes is more physically faithful for super-emitters but doubles the parameter count and complicates the wait-time integral. Default to Poisson with a Hawkes opt-in?
:::

:::{attention} Cross-source independence
Persistency metrics are per-source. Aggregating up to "expected events across a basin in 24h" requires the spatial / population point process from Tier V.B's open questions. Out of scope for v1.
:::

:::{attention} Action thresholds
Wait-time and occurrence probability become operational only with a threshold (e.g. "dispatch if $\mathbb{P}(\text{occur}) > 0.7$"). Where do thresholds live? Probably in the dashboard, not in `plumax` core.
:::
