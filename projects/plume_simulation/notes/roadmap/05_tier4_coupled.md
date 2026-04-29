# Tier IV — End-to-end coupled system

**Forward model:** transport + RTM + multi-instrument fusion, from source parameters all the way to simulated radiances across multiple satellites simultaneously. This is the full operational pipeline.

```
Source params (Q_{1:K}(t), x₀_{1:K}, t₀_{1:K},  ū, θ_wind, c_bg, α_BC, …)
       ↓  [Tier I/II/III transport]
Concentration field  c(x,t)
       ↓  [RTM / AK operator,  per instrument]
Simulated observations  {y_inst}_{inst ∈ {TROPOMI, EMIT, Tanager, GHGSat, …}}
       ↑
Cross-instrument bias correction  bias_inst
```

Tier IV is **assembly + multi-instrument fusion**, not new modelling: it composes transport (any of Tiers I–III) with the [RTM stack](04_rtm_stack.md) and joins observations from multiple satellites into a single coherent posterior. The contribution at this tier is the joint multi-instrument inference, the operational predictor, and the cross-instrument calibration.

---

## (1) Simple model — composed forward over multiple instruments

### Per-instrument forward

For a single instrument `inst`:

```
y_inst = AK_inst · column_z(transport(Q(t), x₀, met) + c_bg) + bias_inst + ε_inst
ε_inst ~ N(0, R_retr,inst + R_repr,inst + R_align,inst)
```

- `transport(Q(t), x₀, met)` — Tier I, II, or III. **Q(t) is time-resolved**, not a static rate (see §3 below).
- `c_bg` — regional background, with prior from the [emission inventory loader](00_prerequisites.md#background-emission-inventory-q_a).
- `AK_inst`, `R_retr,inst` — per-instrument from the [RTM stack](04_rtm_stack.md) or directly from the L2 product.
- `bias_inst ~ N(0, σ_inst²)` — per-instrument additive bias, **first-class state element**. Documented inter-instrument biases are O(±10 ppb); ignoring them double-counts agreement.
- `R_repr,inst` — representation error (model-vs-pixel-footprint mismatch); rises with terrain complexity.
- `R_align,inst` — temporal misalignment error (overpass at `t_inst` vs. modelled state at `t`).

### State vector — full enumeration

Single-overpass coupled inference works in a state space far larger than just `(Q, x₀, t₀)`:

```
x = ( Q_{1:K}(t),                          source rates over time, K unknown
      x₀_{1:K},                            source locations, 2D each
      ū, θ_wind,                           wind, tight prior from met
      c_bg,                                regional background, prior from inventory
      α_BC,                                lateral-BC scaling (Tier III only)
      bias_{inst},                         per-instrument additive bias
      A_surf, AOD,                         shared across the scene (RTM stack)
      … )
```

With trans-dimensional `K = n_sources` (basin case). Single-source single-instrument is the sanity-check special case, not the operational target.

### Multi-instrument fusion

The joint observation operator is a **list-of-forwards** keyed on `instrument_id`, not a single forward:

```
y = [y_TROPOMI, y_EMIT, y_Tanager, y_GHGSat, …]
H(x) = [H_inst(x) for inst in instruments]
```

Each `H_inst` carries its own AK, footprint, native resolution, observation time, and quality-flag schema. **Don't pre-regrid to a common resolution** — loses information; do the AK + footprint averaging at native resolution per instrument, fuse at the likelihood level.

### Spatiotemporal alignment — Q(t) as a stochastic process

Different satellites overpass at different times. With a static `Q` the coupled forward implies the same source state at every overpass — wrong for intermittent/leak emissions and wrong over multi-day windows. Default: **Q(t) ~ Ornstein-Uhlenbeck** with a basin-typical correlation timescale (hours to days), or **Gaussian process** prior with Matern-3/2 covariance. Captures intermittent / burst emissions naturally.

### Build order

Start with the cheapest combination that is still physically coherent, with multi-instrument fusion enabled from day 1:

1. **Tier I + AK + L2 fusion across {TROPOMI, GHGSat, EMIT}** for static Q. This is the v1 target — Methane Alert and Response System (MARS, UNEP-IMEO) style attribution with multi-satellite cross-validation.
2. Lagrangian (Tier II) + AK + L2 fusion → handles wind-driven plumes; same fusion harness.
3. FV (Tier III) + neural RTM + L1 fusion → full L1-radiance inversion with end-to-end gradients.
4. Q(t) stochastic-process upgrade once multi-day events appear in the catalog.

The point: **don't try to ship the most complex tier first.** Each upgrade replaces a single block in the diagram; the multi-instrument fusion harness, likelihood structure, and observational comparison stay the same.

---

## (2) Model-based inference

### End-to-end gradient — honest cost

```
∇_x J = ∇_x [Σ_inst (½ ‖H_inst(x) − y_inst‖²_{R_inst}) + ½ ‖x − x_b‖²_B]
```

JAX autodiff propagates through transport + RTM jointly — no chain rule by hand. **Cost is non-trivial:** each gradient call runs transport + RTM for every instrument's `H_inst`. For Tier III + HAPI that's seconds-to-minutes per call; emulator-based inference (Step 4) is the operational path.

The "JAX gives you gradients for free" framing is a half-truth — the gradient is automatic, but the *forward cost per gradient step* dominates wall time. Budget accordingly.

### Cost function

Three terms:

```
J(x) = Σ_inst  ½ ‖H_inst(x) − y_inst‖²_{R_inst}   observations, per-instrument
     + ½ ‖x − x_b‖²_B                               prior on full state
     + ½ ‖Q(t) − μ_Q(t)‖²_K_Q                       Q(t) stochastic-process prior
```

`B` carries the structured priors from §1 (lognormal `Q`, met-tight `ū`, `θ_wind`, GP/OU on `Q(t)`, etc.); `K_Q` is the OU/GP kernel. `R_inst` includes representation, retrieval, and temporal-alignment terms.

### Quality-flag handling

Per-instrument quality flags from the RTM stack flow into the coupled forward. **Default policy:** flagged pixels contribute zero log-likelihood (mask multiplier in `R^{-1}`). **Don't drop them silently** — keep masks visible in diagnostics so the effective per-instrument observation count is auditable.

### Posterior covariance

Three paths, mirroring Tier III:

- **Laplace around MAP** — cheapest, default for Tier I+L2 fusion.
- **Gauss-Newton Hessian** via Krylov / [`gaussx`](/home/azureuser/localfiles/gaussx/) — used when posterior is approximately Gaussian and tractable.
- **Ensemble (En-EKI / En-4D-Var)** via [`filterax`](/home/azureuser/localfiles/filterax/) — required when the posterior is non-Gaussian (multi-modal across `n_sources`, heavy-tailed `Q`).

Posterior export to Tier V.A is via the same adapter pattern as Tiers I/II/III.

### Trans-dimensional `n_sources`

`K = n_sources` is itself unknown. Three options:

- **Reversible-jump MCMC (RJMCMC).** Birth/death proposals on source count. Exact but slow.
- **Max-K with masking.** Fix `K_max`, infer activity probabilities `p_k` per slot. Tractable, biased toward `K_max`.
- **Hierarchical Dirichlet-process prior.** Variable `K` with a non-parametric prior. Middle ground.

v1: max-K with masking (`K_max = 10` per basin tile). Promote to RJMCMC when basin events exceed `K_max` regularly.

---

## (3) Model emulator — coupled vs. stacked

Two architectural choices:

### Stacked emulators (tier-modular)

Compose Tier-N transport emulator + RTM emulator at runtime. **Pros:** any emulator can be swapped independently; intermediate `c(x,t)` is materialised for diagnostics; modular validation chains directly into Steps 3/4 of each parent tier. **Cons:** two emulator calls per forward; no joint training signal.

### Coupled emulator (single network)

```
(met fields, source params, instrument metadata) → simulated multi-instrument overpass tensor
```

**Pros:** one network call replaces transport + RTM; massive speedup; potentially learns coupling biases the stack misses. **Cons:** intermediate `c(x,t)` is no longer materialised; cross-tier diagnostics break; retraining required when any block changes.

### Decision rule

- **Development / interpretability / cross-tier diagnostics:** stacked.
- **Operational latency-bound deployment (e.g. real-time alerting):** coupled.

Both should exist; the coupled emulator is validated against the stacked composition before deployment.

### Training-data budget

"Millions of pairs" naively needs `O(10⁶)` transport+RTM simulations. For Tier III + HAPI that's CPU-years on a single machine. **Active learning is mandatory:** sample sequentially, prioritise loss-residual hot spots and operationally important tails (sun-glint, high AOD, low PBL, multi-source basins). Reaches operational accuracy with `O(10⁵)` or fewer simulations.

### Domain randomization

Sample the joint `(met regime, source configuration, scene class, viewing geometry, instrument, n_sources)` distribution with **stratified sampling**, not uniform. Naive uniform under-represents the tail regimes that actually drive operational failures.

---

## (4) Emulator-based inference

Use the coupled (or stacked) emulator in EKI ([`filterax`](/home/azureuser/localfiles/filterax/)) or gradient-based inversion. Real-time capable.

- **Adjoint validation:** emulator-autodiff gradient ≈ physics-stack gradient on a held-out set. Same hard test as Tiers III and RTM — failure means the inversion is biased even when forward predictions look fine.
- **Posterior validation:** posterior from coupled-emulator inversion ≈ posterior from end-to-end physics inversion (Step 2).

---

## (5) Amortized inference (predictor)

```
f_θ: ( {(instrument_id, y_inst, AK_inst, mask_inst, footprint_inst)},
       met_reanalysis, transport_tier_id ) → p( Q_{1:K}(t), x₀_{1:K}, K | observations, met )
```

This is **the operational product**: a multi-instrument satellite-overpass list goes in, source-parameter posterior comes out.

### Multi-instrument list input

Input is a *list* of per-instrument observation tuples — same pattern as Tier II/III, generalised to a heterogeneous list. Each element keeps native resolution, AK, mask, and footprint. **No pre-regridding.**

### Per-instrument heads, tier-conditioned

- **Per-instrument summary networks** (TROPOMI 5 km vs. EMIT 60 m vs. Tanager 30 m vs. GHGSat hyperspectral need different encoders).
- **Transport tier as categorical context.** `transport_tier_id ∈ {I, II, III}` conditions the posterior head — the predictor is one model that handles all tiers, not three separate models.
- **Met + tier conditioning** wired in via FiLM / hypernet primitives in [`pyrox.nn`](/home/azureuser/localfiles/pyrox/) — same pattern as Tiers I/II/III.

### Trans-dimensional output

`K = n_sources` varies. Default architecture: **max-K masked output**, predicting `(K, {Q_k(t), x₀_k}_{k=1}^{K_max}, activity_mask)` jointly. Activity mask is a Bernoulli per slot. Promote to RJMCMC predictor head only if max-K masking shows systematic basin saturation.

### Posterior representation

- `Q_k(t)` posterior is a 1D function — conditional flow over time-axis (`gauss_flows` 1D handles natively).
- `x₀_k` posterior is 2D Gaussian per source.
- `K` posterior is categorical.
- Joint via factorised flow + categorical head.

### Training data

Simulate millions of `(source config, multi-instrument overpass)` pairs spanning the realistic met regime distribution + scene-class distribution + instrument coverage distribution. Active learning over the training schedule (§3) is mandatory at this scale.

---

## (6) Improve

- **Active learning loop.** Flag high-uncertainty scenes for targeted follow-up (e.g. trigger a GHGSat tasking based on a TROPOMI alert). Posterior entropy from the predictor is the natural trigger metric.
- **Joint met + source posterior.** Currently we condition on met with tight priors; instead infer `p(Q, ū, θ_wind, ... | y)` jointly with a looser met prior. WRF becomes informative prior, not hard constraint. Critical for remote regions where reanalysis is poor.
- **Hierarchical bias correction.** Bias `bias_inst` per instrument is currently a flat Gaussian. Promote to hierarchical: per-(instrument, basin, season) — captures known seasonal/regional structure in inter-instrument biases.
- **Multi-species coupling.** When CO + CH₄ are observed jointly (e.g. TROPOMI), use the source-ratio prior to constrain attribution (fossil vs. agricultural).
- **Q(t) hierarchical prior.** Promote OU correlation timescale and amplitude to hyperparameters with their own posterior.

---

## Module layout (proposed)

| Step | Concern | Module | Status |
|------|---------|--------|--------|
| 1 | Coupled forward (Tier I + AK + multi-inst) | `plume_simulation.coupled.gaussian_ak` | ☐ |
| 1 | Coupled forward (Tier II + AK + multi-inst) | `plume_simulation.coupled.lagrangian_ak` | ☐ |
| 1 | Coupled forward (Tier III + RTM + multi-inst) | `plume_simulation.coupled.fv_rtm` | ☐ |
| 1 | Multi-instrument fusion harness | `plume_simulation.coupled.fusion` | ☐ |
| 1 | Cross-instrument bias model | `plume_simulation.coupled.bias` | ☐ |
| 1 | Quality-flag aggregator | `plume_simulation.coupled.quality` | ☐ |
| 1 | Q(t) stochastic-process model (OU / GP) | `plume_simulation.coupled.q_dynamics` | ☐ |
| 1 | Trans-dimensional source-count handling | `plume_simulation.coupled.k_sources` | ☐ |
| 2 | End-to-end inversion | reuse [`assimilation/`](src/plume_simulation/assimilation/) with composed `forward` | ☐ |
| 2 | Posterior covariance (Laplace / Hessian / EnKF) | reuse Tier III's posterior modules | ☐ |
| 2 | Posterior export → Tier V | `plume_simulation.coupled.posterior_export` | ☐ |
| 3 | Stacked emulator runtime | `plume_simulation.coupled.stacked_emulator` | ☐ |
| 3 | Coupled emulator (end-to-end) | `plume_simulation.coupled.emulator` | ☐ |
| 3 | Active-learning training scheduler | `plume_simulation.coupled.active_learning` | ☐ |
| 5 | Operational predictor (per-instrument, tier-conditioned) | `plume_simulation.coupled.predictor` | ☐ |
| 6 | Joint met + source inversion | `plume_simulation.coupled.joint_met` | ☐ |

The `coupled` subpackage doesn't exist yet; this is the proposed shape. It's the only tier where new top-level modules are still needed once Tiers I–III and RTM are done.

---

## Validation strategy

- **Composition correctness.** Apply identity AK, identity RTM, single-instrument list → coupled forward should equal the bare transport forward. Cheap, catches plumbing bugs.
- **Linear-conditional-Gaussian limit.** Tier I + linear AK + Gaussian noise → conditional posterior `p(Q | x₀, ū, θ_wind, c_bg)` is closed-form via [`gaussx`](/home/azureuser/localfiles/gaussx/). Compare end-to-end JAX inversion to the closed-form result. (Note: only `Q` is linear; the joint over `(x₀, ū, …)` is nonlinear — phrase the test as conditional, not joint.)
- **Synthetic-truth recovery.** Simulate a known multi-instrument overpass through the full pipeline → run inversion → recover within reported posterior uncertainty. Stratify by instrument count (1, 2, 3+) — fusion benefit should be quantifiable.
- **Cross-instrument hold-out.** Invert with `N − 1` instruments, predict the held-out instrument's observations from the source posterior + coupled forward, compare predicted to actual. Catches multi-instrument fusion bugs and exposes the value of fusion vs. single-instrument inversion.
- **Cross-tier consistency.** Run inversion with Tier I, II, III transports on the same observations under stationary met conditions. Posteriors should overlap within stated uncertainty. Catches systematic biases between transport tiers and confirms the fusion harness is tier-agnostic.
- **Real-data multi-pass benchmark.** **The validation that proves Tier IV works.** Invert a documented event using `(TROPOMI + GHGSat + EMIT)` simultaneously (e.g. published Permian super-emitter from Sherwin et al. 2024 cross-comparison campaigns); compare posterior to the reported emission. Without this the tier is unvalidated for its intended use.
- **Bias-correction calibration.** With known cross-instrument biases (Sherwin et al.'s controlled-release flights), the inferred `bias_inst` posterior should recover the published values within 95% CI.
- **Predictor calibration.** SBC stratified by instrument count, met regime, scene class. Standard-deviation calibration vs. empirical RMSE on a held-out set.

---

## Open questions

- **Tiering at inference time.** Should the user choose the transport tier, or should the predictor figure it out (e.g. choose Tier I for stationary winds, Tier II for turbulent regimes)? Probably the former for v1 — explicit is safer; predictor-side dispatch as v2.
- **Trans-dimensional posterior — RJMCMC vs. masked-K.** v1 commitment is masked-K (`K_max = 10`). When does basin saturation force the upgrade to RJMCMC? Operationally: when >5% of overpasses saturate `K_max`.
- **Q(t) parameterisation.** OU process is the simplest non-trivial choice; GP with Matern-3/2 is more flexible but slower. v1: OU; v2: hierarchical kernel choice.
- **Cross-instrument bias structure.** Flat per-instrument Gaussian (v1), per-(instrument, basin) (v1.5), per-(instrument, basin, season) (v2). Promotion driven by residual diagnostics on real-data benchmark.
- **Training-data budget.** Coupled-emulator training cost is operationally prohibitive without active learning. Open: target `O(10⁵)` simulations with sequential design vs. `O(10⁶)` uniform — confirm the active-learning factor in pilot.
- **Coupled vs. stacked emulator default.** Stacked for development. Coupled for deployment. Open: at what predictor latency budget does coupled become required? Likely when overall pipeline must respond <1 s to an alert trigger.
- **What goes into "operational"?** This is the line between research artefact and product. Need to decide: SLA on inference latency, supported scene types, failure modes, monitoring. Probably outside scope of `plumax` itself — that's `plumax-deploy` or similar.
- **Posterior summaries.** A full distribution over all state elements is unwieldy for downstream consumers. Canonical 1-page summary: per-source mode + 68/95% credible region + entropy budget + per-instrument bias posterior + activity flag. Pin the schema before Tier V.A starts consuming.
- **Quality-flag aggregation.** Mask flagged pixels (default), or down-weight via inflated `R`? Open: empirical comparison on real data.
