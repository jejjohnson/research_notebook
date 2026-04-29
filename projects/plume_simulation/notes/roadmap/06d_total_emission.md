# Tier V.D — Total emission estimation

**Question:** Given a population of detected plumes (and the ones we missed), what is the true total emitted mass over a region and time window?

This is the **inventory-grade output** of `plumax` — the number that gets reported into national greenhouse-gas inventories, climate models, and policy dashboards. It also requires the most care, because the satellite catalog you start from is **systematically biased**: detection thinning means the very things you can't see (small, frequent leaks) are exactly the things that matter for total mass.

---

## The missing-mass paradox

The full Monte Carlo proof is in [`methane_pod/notebooks/03_missing_mass_paradox`](projects/methane_pod/notebooks/03_missing_mass_paradox.ipynb). The result, in one sentence:

> A POD-thinned plume catalog simultaneously **overestimates the average emission rate** (because it oversamples big leaks) and **underestimates the total emitted mass** (because it misses many small leaks).

These two biases pull in opposite directions, but they don't cancel — averaging the wrong thing over the wrong sample size gives you the wrong total. The corrected estimator has to model the thinning explicitly.

---

## The corrected total-mass estimator

Given a TMTPP fit (Tier V.B) with posterior `(λ, f, P_d)`:

```
M_total(T) = E[N_true(T)] · E[Q]
           = (∫₀^T λ(t) dt) · (∫ Q · f(Q) dQ)
```

This is the **un-thinned** total — what would be emitted regardless of detection. Compare to the naive estimator:

```
M_naive(T) = Σ_{i: detected} Q_i
           ≈ (∫₀^T λ(t) · ∫ P_d(Q) f(Q) dQ dt) · E[Q | detected]
           = E[N_detected(T)] · E[Q | detected]
```

`M_naive` is biased low because:

- `E[N_detected] < E[N_true]` (some events missed).
- `E[Q | detected] > E[Q]` (detected events are systematically bigger — heavy tail of `f`).

The two errors *compound* rather than cancel: the regional total is undercounted, and the per-event mean is inflated. Inverting the POD model is the **only** way to recover an unbiased total.

---

## Posterior over total mass

With NUTS samples `(λ^{(s)}, f^{(s)}, P_d^{(s)})`, the posterior over `M_total(T)` is:

```
M_total^{(s)}(T) = (∫₀^T λ^{(s)}(t) dt) · (∫ Q · f^{(s)}(Q) dQ)
```

Reported as posterior median + 95% credible interval. Both integrals are tractable for the standard intensity / mark choices (closed-form for constant `λ` + lognormal `f`; quadrature otherwise).

---

## Validation strategy

This is the most important validation set in the whole tier — without it, the estimator is just a number.

- **MC ground truth (bias direction).** Reproduce the qualitative result of the paradox notebook: simulate a known `(λ*, f*, P_d*)`, compute `M_true` exactly, and check that the corrected estimator recovers `M_true` while `M_naive` is biased low.
- **MC ground truth (calibration).** Across 1000 replicates of the previous test, the 95% credible interval on `M_total` should contain `M_true` ~95% of the time.
- **Per-satellite sensitivity.** Same population, two different `P_d` (e.g. GHGSat-floor vs. TROPOMI-floor) → corrected estimator should give the same `M_total` posterior. The naive estimator gives wildly different `M_naive`. This is the test that *proves* the correction is doing its job.
- **Real-data benchmark.** Once [`07_pod_fitting_mcmc`](projects/methane_pod/notebooks/07_pod_fitting_mcmc.md) lands with IMEO + Tanager data, compare the corrected total for a well-studied basin (Permian) to published bottom-up inventories (EPA GHGI, GHGRP) and to top-down inverse-modelling estimates (Lu et al., Sherwin et al.). They will disagree; the question is whether the corrected estimator is *closer* to the top-down number than the naive one.

---

## Module layout

| Concern | Module | Status |
|---------|--------|--------|
| Missing-mass MC simulator | [`methane_pod.paradox`](projects/methane_pod/src/methane_pod/paradox.py) | ✓ (NumPy) |
| Posterior fit | [`methane_pod.fitting`](projects/methane_pod/src/methane_pod/fitting.py) | ✓ (synthetic); 🚧 (real data) |
| `M_total` estimator + uncertainty | `plume_simulation.population.totals` | ☐ |
| Per-satellite calibration loader | `plume_simulation.population.satellite_pod` | ☐ |
| Multi-satellite fusion | `plume_simulation.population.fusion` | ☐ |

---

## Multi-satellite fusion (Tier V.D extension)

For a region observed by `K` satellites, each with its own POD, the unified detection probability is:

```
P_d^union(Q) = 1 − ∏_k (1 − P_d^k(Q))
```

This is the "any satellite saw it" probability. Folds into the TMTPP likelihood as a single replacement of `P_d` with `P_d^union`. Adds one strong assumption: detections by different satellites are conditionally independent given the leak size — defensible at the population level, possibly violated for clustered super-emitters.

Open: whether to model **which** satellite detected each event (categorical mark) or just the union. The first gives more information per event but doubles the number of POD parameters.

---

## Open questions

- **Mass vs. mass-rate.** `M_total(T)` is mass. Most published inventories report mass-rate (Tg / yr). The conversion is `M_total / T`, but `T` for a satellite catalog is fuzzy — what's the effective observing time when satellites overpass intermittently? Document the convention.
- **Spatial aggregation.** Currently temporal-only. Aggregating `M_total` over a basin requires either a spatial point process (cleaner) or stratifying the sources by facility class and combining (operational shortcut). v1: stratification; v2: spatial CGS / Cox process.
- **POD parameter sources.** Per-satellite POD parameters can come from (a) fits in [`methane_pod`](projects/methane_pod/) on a held-out catalog, (b) published values from Varon et al. / Cusworth et al., or (c) joint inference with the population. Each has trade-offs around identifiability.
- **Reporting cadence.** Inventories are annual; satellites are daily-ish. How do we smooth the `M_total` time series? Rolling 30-day window? Bayesian time-series prior on `λ(t)`?
