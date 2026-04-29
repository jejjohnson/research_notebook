# Tier V.C — Persistency

**Question:** Given the inverted intensity `λ(t)` from Tier V.B, when will the next emission event happen, and what's the probability of an event during a specified window?

This is the **operational layer** — what an LDAR (Leak Detection and Repair) crew or a satellite-tasking dispatcher actually consumes. The full derivations of each metric live in [`methane_pod/notebooks/08_persistency`](../../../methane_pod/notebooks/08_persistency.md); this page summarises the metrics and how they slot into the `plumax` API.

---

## The four operational metrics

### 1. Expected wait time `E[Δt | t₀]`

How long after time `t₀` until the next event?

- **Homogeneous** (`λ ≡ λ₀`): `E[Δt] = 1 / λ₀` — memoryless; doesn't depend on when you start.
- **Inhomogeneous**: `E[Δt | t₀] = ∫_{t₀}^∞ exp(−∫_{t₀}^t λ(u) du) dt` — depends on starting clock; for a diurnal source, vastly different at noon vs. midnight.

**Operational use.** Dispatch decisions: arrive during a high-`λ` window and the next event is imminent (worth waiting); arrive during a low-`λ` window and you'd waste hours. Drives MARS-style dispatch suppression during dormant cycles.

### 2. Probability of occurrence `P(N(t₁, t₂) ≥ 1)`

What's the chance of at least one event in `[t₁, t₂]`?

- **Homogeneous**: `1 − exp(−λ₀ · (t₂ − t₁))`.
- **Inhomogeneous**: `1 − exp(−∫_{t₁}^{t₂} λ(t) dt)`.

**Operational use.** "Wrench-turning" probability. If a maintenance window is 4 hours, what's the chance the leak shows itself during that window? Drives whether to schedule the visit.

### 3. Conditional intensity given prior detection `λ(t | last detect)`

For a source with a known recent detection at `t_prev`, what's the posterior intensity going forward?

For Poisson processes (no memory): unchanged. For Hawkes / self-exciting processes: bumped — `λ(t | t_prev) = μ + α · exp(−β·(t − t_prev))` — captures the empirical observation that super-emitters "cluster".

**Operational use.** Prioritisation: a source with a recent detection is *more* likely to repeat-emit in the next 24 h. Re-task a high-resolution satellite (GHGSat, Carbon Mapper) on top of a TROPOMI alert.

### 4. Cumulative event count `E[N(0, T)]` and credible bounds

Expected number of events in `[0, T]`, with credible interval from the posterior on `λ`.

- **Homogeneous**: `λ₀ · T`.
- **Inhomogeneous**: `Λ(T) = ∫₀^T λ(t) dt`.

**Operational use.** Annual reporting, regulatory compliance. "How many emission events should we expect this year at this facility class, with 95% credible interval?"

---

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

## Module layout

| Concern | Module | Status |
|---------|--------|--------|
| Intensity functions | [`methane_pod.intensity`](../../../methane_pod/src/methane_pod/intensity.py) | ✓ |
| Wait-time / occurrence / cumulative metrics | `plume_simulation.population.persistency` | ☐ |
| Posterior-aware metric wrappers | same module | ☐ |
| Operational dashboard / report templates | out of scope for `plumax`; lives in `plumax-deploy` (future) | — |

The integral over `λ(t)` in the wait-time formula is closed-form for a few intensity choices (constant, exponential decay) and otherwise needs `jax.scipy.integrate` or a fixed quadrature. Worth wrapping once and reusing across metrics.

---

## Validation strategy

- **Homogeneous limit.** For constant `λ`, all four metrics have closed-form formulas; the implementation should match to machine precision.
- **MC self-consistency.** Sample `n` event times from a known `λ(t)` via thinning, compute the empirical wait time / occurrence frequency, compare to the closed-form metric. Tests both the metric implementation and the simulator.
- **Posterior coverage.** For a synthetic source with known `λ_true(t)`, the 95% credible interval on `E[Δt]` should contain the truth ~95% of the time across replicates.
- **Diurnal sanity.** A solar-heated tank with peak `λ` at 14:00 should have `E[Δt | 14:00] ≪ E[Δt | 02:00]`. Numerical sanity check, not a formal test, but catches sign errors.

---

## Open questions

- **What's "the" intensity?** A point estimate (posterior mean) or the full posterior over `λ` parameters? Operational dashboards may want the former; researchers want the latter. The API returns posterior samples by default; downstream summarisation is the caller's choice.
- **Hawkes vs Poisson default.** Hawkes is more physically faithful for super-emitters but doubles the parameter count and complicates the wait-time integral. Default to Poisson with a Hawkes opt-in?
- **Cross-source independence.** Persistency metrics are per-source. Aggregating up to "expected events across a basin in 24h" requires the spatial / population point process from Tier V.B's open questions. Out of scope for v1.
- **Action thresholds.** Wait-time and occurrence probability become operational only with a threshold (e.g. "dispatch if `P(occur) > 0.7`"). Where do thresholds live? Probably in the dashboard, not in `plumax` core.
