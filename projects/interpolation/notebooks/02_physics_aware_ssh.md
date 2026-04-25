---
title: "Physics-aware SSH reconstruction — where physics enters the pathwise GP"
---

# Physics-aware SSH reconstruction

The pipeline in [00_ssh_pathwise_sampling.md](00_ssh_pathwise_sampling.md) and [01_efficient_machinery.md](01_efficient_machinery.md) is a **physics-agnostic** Gaussian process: a single Matérn × OU kernel encodes "the field is smoothly varying in space and time." That is a strong baseline — essentially what classical DUACS optimal interpolation does[^taburet2019] [^ballarotta2019] — but it leaves on the table a half-century of ocean-dynamics knowledge.

This note organises *how* that knowledge gets injected. To frame it, recall the pathwise posterior formula from 00:

$$
\boxed{\;
\underbrace{f(\mathcal{X}^*) \mid y}_{\text{posterior}}
\;\overset{d}{=}\;
\underbrace{\tilde f(\mathcal{X}^*)}_{\text{prior path}}
\;+\;
\underbrace{K_{\mathcal{X}^*\mathcal{X}}}_{\text{cross-cov}}
\,\underbrace{\bigl(K_{\mathcal{X}\mathcal{X}} + \sigma_{obs}^2 I\bigr)^{-1}}_{\text{inversion}\,=\,C^{-1}}
\,\bigl(\,
\underbrace{y}_{\text{data}}
\;-\;
\underbrace{\tilde f(\mathcal{X})}_{\text{prior path at obs}}
\bigr).
\;}
$$

Every term in this expression is a knob:

| Symbol | What it controls | Where physics enters | Treated in |
|---|---|---|---|
| $y,\, \sigma_{obs}^2$ | The observations and their noise | **DATA** — pre-process them, augment with new modalities (§2) | **§2 below** |
| $K_{\mathcal{X}\mathcal{X}},\, K_{\mathcal{X}^*\mathcal{X}}$ | The prior covariance — the kernel | **KERNEL** — choose a physically motivated kernel; structure it as a sum of physically meaningful components (§3) | **§3 below** |
| $C^{-1}$ | The linear inversion | **COMPUTE** — Woodbury, PCG, low-rank approximations | [01](01_efficient_machinery.md) |
| $\tilde f$ | The prior sample (RFF) | **COMPUTE** — RFF basis, Bochner | [01](01_efficient_machinery.md) |
| $\mathcal{X}^*$ | The prediction inputs (the grid) | **GEOMETRY** — patch decomposition, localisation | [03](03_global_scaling_patches.md) |

This note is about the first two columns: **DATA** (§2) and **KERNEL** (§3). A final §4 places geostrophic-current diagnostics, which are post-hoc and touch nothing in the pathwise formula above.

Two concrete axes summarise the menu:

> 1. **Modify the data** — subtract physics that is already understood (tides, atmospheric loading, mean dynamic topography); add velocity observations from drifters / HF radar / Doppler altimetry as *gradient* constraints on $\eta$.
> 2. **Modify the kernel** — pick a kernel whose spectral density encodes ocean dynamics (SQG); decompose the kernel as a sum of physically meaningful components on multiscale localised bases (MIOST/MASSH Gabor wavelets); keep low-rank structure so the gaussx Woodbury machinery from 01 still applies.

The grouping is deliberate: **the GP machinery from 00/01/03 stays the same throughout** — only the inputs $y$ and the kernel $K$ change. Methods that abandon the static-prior GP framework — variational data assimilation with dynamical priors (3D-Var, 4D-Var, DYMOST, BFN-QG) and learned variational schemes (4DVarNet) — live in [04_variational_dynamical_priors.md](04_variational_dynamical_priors.md), since they swap out the entire $C^{-1}$ + $K$ block, not just the kernel.

---

## 2. The DATA knob — modify $y$ before the GP sees it

The observation vector $y$ in the pathwise formula is a passive input — anything you do to $y$ before passing it to the solver is honoured exactly. Two physical interventions:

### 2a. Subtract — pre-processing the obs

The numbers in 01 quietly assume that the observations have already been corrected for the standard altimetry noise sources. They have not, by default — CMEMS L3 products are minimally corrected. The standard pipeline:

| Correction | Source | Removes |
|---|---|---|
| Wet/dry tropospheric | Radiometer + ECMWF | Atmospheric refraction (~10 cm) |
| Ionospheric | Dual-frequency altimeter | Plasma path delay (~5 cm at low latitudes) |
| Sea-state bias | Empirical (SWH, U10 dependence) | EM bias from wave shape (~few cm) |
| Solid Earth + pole tide | Cartwright–Tayler model | Earth-body tides (~30 cm) |
| Ocean tides | FES2014 / FES2022[^lyard2021] | M2, S2, K1, O1, … (~1 m peak) |
| Dynamic Atmospheric Correction (DAC) | MOG2D barotropic + IB[^carrere2003] | High-frequency wind/pressure response (~10 cm at mid-latitudes) |
| Mean Dynamic Topography (MDT) | CNES-CLS-22[^jousset2025] | Time-mean SSH so you work with anomaly $\eta_a$ |

After all corrections, what remains is the **sea-level anomaly** $\eta_a = \eta - \eta_{\text{MDT}}$ — the field the GP should be modelling. Three things to know:

- **All corrections are pre-computed and provided in the L3 product.** Reading the right CMEMS variable (`sla_unfiltered` or `sla_filtered`) gets you the already-corrected anomaly.
- **DAC is essential when pooling across short time windows.** The 5–10 cm barotropic response to a passing storm is *highly correlated across all observations within a few days* and will break the GP's stationarity assumption if not removed.
- **Tide residuals at coastal points are the dominant error budget.** FES tidal models have ~3 cm RMS on the open ocean but degrade to >10 cm in narrow shelf seas (English Channel, Patagonian shelf). For coastal SSH work, expect to filter or down-weight these.

In code: this is a one-shot upstream step. The GP machinery is unchanged.

### 2b. Augment — derivative observations from velocities

Lagrangian drifter trajectories give direct estimates of surface velocity (after removing Ekman, Stokes, and inertial-oscillation components)[^rio2014] [^mulet2021]. HF-radar gives the same on shelf seas. SKIM[^ardhuin2019] and the recently selected ESA Harmony / future Doppler-altimetry missions give it from space. **All of these constrain *gradients* of $\eta$ via geostrophic balance:**

$$
u_g(x, y) \;=\; -\frac{g}{f(y)}\,\frac{\partial \eta}{\partial y}, \qquad
v_g(x, y) \;=\; \frac{g}{f(y)}\,\frac{\partial \eta}{\partial x},
$$

with $g \approx 9.81\,\text{m s}^{-2}$ and Coriolis $f = 2 \Omega \sin\phi$ at latitude $\phi$. These are *linear* functionals of $\eta$ — partial derivatives with a latitude-dependent multiplier. Gaussian processes are closed under linear operators, so derivative observations slot into the GP as additional rows of the Gram matrix:

$$
\mathrm{Cov}\bigl(\partial_y \eta(x),\, \eta(x')\bigr) = \partial_{y} k_\theta(x, x'), \qquad
\mathrm{Cov}\bigl(\partial_y \eta(x),\, \partial_{y'} \eta(x')\bigr) = \partial_{y}\partial_{y'} k_\theta(x, x').
$$

For Matérn-3/2 these have closed form; in JAX, **`jax.grad` of the kernel function gives you all the entries automatically** — no analytical work needed. Recipe:

1. Pre-process drifter velocities to remove Ekman + inertial + Stokes (standard CMEMS workflow).
2. Multiply by $-f/g$ and $f/g$ to obtain "SSH-gradient observations."
3. Stack them onto the SLA observation vector $y$.
4. Define a kernel function that branches on observation type — `eta×eta`, `eta×∂η`, `∂η×∂η` — using `jax.grad(k)` for the differentiated entries.
5. Pass the augmented kernel to `gx.ImplicitKernelOperator` and the augmented $y$ to `gx.solve`. Everything else from 00/01 is unchanged.

This is mathematically identical to how the CMEMS Mean Dynamic Topography is constructed[^rio2014] [^mulet2021] [^jousset2025], where steady-state geostrophic balance jointly constrains altimetry residuals and drifter velocities.

**Cyclostrophic / equatorial caveats.** Geostrophy degenerates as $f \to 0$ at the equator and inside tight eddy cores where centrifugal acceleration competes with Coriolis. Standard fixes are $\beta$-plane geostrophy near the equator[^bonjean2002] and gradient-wind balance for cyclostrophic eddies — both add a quadratic-in-$\nabla\eta$ correction term, and the GP is no longer linear. Skip unless you specifically care about the equatorial band or eddy-core diagnostics.

---

## 3. The KERNEL knob — physics-informed priors

The kernel $k_\theta$ is the entire prior — it determines the smoothness, anisotropy, multi-scale content, and spectral slope of every reconstruction. The Matérn × OU baseline says only "smooth, isotropic, one length scale per axis." Two physical refinements, in increasing order of structural complexity:

### 3a. Clever spectral density — the SQG kernel

In a uniformly stratified ocean with zero interior potential vorticity, the dynamics reduce to a single equation for surface buoyancy[^held1995]:

$$
\partial_t b_s + J(\psi_s, b_s) \;=\; 0, \qquad b_s = -f \partial_z \psi |_{z=0},
$$

with surface streamfunction $\psi_s$ recovered from $b_s$ by a **half-Laplacian inversion**:

$$
\psi_s = (-\nabla^2)^{-1/2} \, b_s / N,
$$

where $N$ is the upper-ocean buoyancy frequency. Sea level $\eta = f \psi_s / g$ inherits the same spectral relation: $\hat\eta(k) \propto k^{-1} \hat b_s(k)$. Combined with the canonical SQG buoyancy spectrum $\widehat{|b_s|^2}(k) \propto k^{-5/3}$, this gives an **SSH spectral slope of $k^{-11/3}$** in the SQG regime, vs $k^{-5}$ for interior QG turbulence[^lapeyre2006] [^isern2008].

**Use as a GP kernel.** The "SQG kernel" is a stationary kernel whose spectral density follows the $k^{-11/3}$ slope, with a low-wavenumber cutoff at the Rossby deformation radius and a high-wavenumber cutoff at the altimeter noise floor. From the GP point of view this is just a different choice of $S(\omega)$ in Bochner's theorem — feed it into [`pyrox._basis._rff._draw_spectral_frequencies`](file:///home/azureuser/localfiles/pyrox/src/pyrox/_basis/_rff.py) and the rest of the pathwise machinery from 01 stays unchanged. Physically, the prior says "fields with energy concentrated at wavelengths near the deformation radius and a known fall-off rate are *a priori* more plausible" — much sharper than the Matérn fall-off, and it concentrates statistical strength where the ocean actually has variance.

**Where SQG is and isn't valid.** SQG works well[^isern2008] [^gonzalez2014] in the upper ocean of energetic mid-latitude regions, in winter when deep mixed layers maintain near-zero PV, and at mesoscale wavelengths (~50–300 km). It breaks at submesoscale (<50 km) where mixed-layer instabilities dominate[^callies2013], in summer when stratification is shallow, and below the surface mixed layer (use eSQG / interior + surface combinations instead). Pragmatically: use SQG as the *spatial* part of a multi-component kernel (next subsection) and let other components absorb what SQG misses.

### 3b. Structured multi-scale kernels — MIOST / MASSH Gabor wavelets

DUACS uses one global Matérn-like spatial covariance per region with hyperparameters tuned by latitude band[^taburet2019]. **MIOST** (Multiscale Inversion of Ocean Surface Topography, CNES/CLS) replaces this with a *sum of independent components*, each expanded on a redundant tight frame of localised wavelet atoms[^ubelmann2021] [^ballarotta2023]:

$$
\eta(x, t) \;=\; \sum_{c \in \text{components}} \sum_{i \in \text{atoms}_c} \alpha_i^{(c)} \, \psi_i^{(c)}(x, t).
$$

Components in operational MIOST: large-scale geostrophy, mesoscale geostrophy, equatorial waves, barotropic motion, internal tides[^ubelmann2021] [^dibarboure2025]. Each component carries its own covariance, diagonal in its wavelet basis (one variance per atom). The global covariance is then a *sum of structured operators* — one block per component.

#### The wavelet atoms

The reference open implementation is the Le Guillou **MASSH** code, mirrored locally at [`jej_vc_snippets/quasigeostrophic_model/massh/basis.py`](file:///home/azureuser/localfiles/jej_vc_snippets/quasigeostrophic_model/massh/basis.py). Each atom is a Gaspari–Cohn windowed plane wave (Gabor / Morlet wavelet):

$$
\psi_i(x, y, t) \;=\; w\!\left(\tfrac{x - x_0}{\Delta_x}\right) w\!\left(\tfrac{y - y_0}{\Delta_x}\right) w\!\left(\tfrac{t - t_0}{\Delta_t}\right) \cdot \sqrt{2}\, \cos\bigl(k_x (x - x_0) + k_y (y - y_0) - \omega (t - t_0) + \varphi\bigr),
$$

with $w$ a Gaspari–Cohn compact-support window (so atoms genuinely vanish outside $\Delta_x$, unlike Gaussians). The atom is parameterised by

| Symbol | Meaning |
|---|---|
| $(x_0, y_0, t_0)$ | atom centre — placed on a hierarchy of grids, one grid per scale octave |
| $(k_x, k_y, \omega)$ | central wavenumbers (multiples of $2\pi / \lambda$ at the atom's scale) |
| $\Delta_x, \Delta_t$ | spatial / temporal support radii — typically $\Delta_x = \lambda \cdot n_{\text{psp}} / 2$, with $n_{\text{psp}}=3$ wavelengths per atom |
| $\varphi$ | $\{0, \pi/2\}$ — gives both cosine and sine atoms (real wavelet pair) |

Each scale octave gets its own grid spacing and its own band of wavenumbers; the user picks the number of octaves to span the resolved range (e.g. $20\,\text{km}$ to $500\,\text{km}$, log-spaced).

#### Why this fits naturally into the pathwise GP

The wavelet expansion is *exactly* a finite-rank, structured prior on $\eta$. Stack the basis evaluations into a feature matrix $\Psi$ and the per-atom variances into a diagonal $\mathbf{D}$:

$$
\eta = \Psi \alpha, \qquad \alpha \sim \mathcal{N}(0, \mathbf{D}), \qquad \mathbf{D} = \mathrm{diag}(\sigma_i^2)
\quad\Longrightarrow\quad K_{\mathcal{X}\mathcal{X}} \;=\; \Psi \mathbf{D} \Psi^\top.
$$

This is a **low-rank + diagonal** kernel — structurally identical to the RFF Woodbury route from [01](01_efficient_machinery.md#3.-route-b-rff-woodbury-approximate-but-very-cheap-when-r-m), with three concrete differences that make it strictly better as an SSH prior:

- **Atoms are localised and bandpass**, not global plane waves. $\Psi$ is *sparse* in space and frequency: each pixel is touched by $\mathcal{O}(\log \lambda_{\max} / \lambda_{\min})$ atoms (one per octave that overlaps it), and each atom touches $\mathcal{O}(\Delta_x^2)$ pixels. Matrix-vector products with $\Psi$ are sparse and embarrassingly parallel — much cheaper than the dense $r(n+m)$ of RFF.
- **The diagonal $\mathbf{D}$ encodes the energy spectrum directly.** Set $\sigma_i^2$ from a target spectral slope (e.g. $k^{-11/3}$ for SQG, $k^{-3}$ for QG, observed slope from in-region spectra). No kernel-fitting needed.
- **Components are additive and independent.** $\Psi = [\Psi_{\text{geo}}, \Psi_{\text{eq}}, \Psi_{\text{bt}}, \Psi_{\text{IT}}]$ block-stacks per-component bases; $\mathbf{D}$ is block-diagonal. Each component lives in its own wavelet sub-frame with its own physical interpretation.

In gaussx vocabulary: $\Psi \mathbf{D} \Psi^\top$ is a [`gaussx.LowRankUpdate`](https://github.com/jejjohnson/gaussx/blob/main/src/gaussx/_operators/_low_rank_update.py) with sparse $\Psi$. The noisy Gram $C = \Psi \mathbf{D} \Psi^\top + \sigma_{obs}^2 I$ goes through Woodbury *exactly* as in 01's Route B, but with $\Psi$ implemented as a sparse-matvec instead of a dense feature matrix. **You get the per-day cost numbers from 01's "fully-RFF" column with a physically motivated basis**, plus the multi-component decomposition that DUACS lacks.

Resolution comparison (operational, mid-latitudes): DUACS recovers down to $\sim 150{-}200\,\text{km}$ wavelengths, MIOST down to $\sim 100{-}150\,\text{km}$[^ballarotta2019]. With SWOT KaRIn ingested, MIOST resolves $\sim 50{-}80\,\text{km}$[^dibarboure2025].

#### MIOST vs inducing points — same family, different basis

Both MIOST wavelets and Nyström-style inducing points (`gaussx.nystrom_operator` from [01](01_efficient_machinery.md#3.-route-b-rff-woodbury-approximate-but-very-cheap-when-r-m)) yield a low-rank approximation $K \approx \Psi \mathbf{D} \Psi^\top$. The difference is *what $\Psi$ contains*:

| Method | $\Psi$ columns | $\mathbf{D}$ | Picks up... |
|---|---|---|---|
| RFF (01) | random plane waves at frequencies $\omega_j \sim S(\omega)$ | identity | matches the kernel spectral density on average |
| Nyström inducing points | $K_{\mathcal{X}, \mathcal{Z}}$ with $\mathcal{Z}$ uniformly subsampled obs | $K_{\mathcal{Z}\mathcal{Z}}^{-1}$ | matches the kernel exactly at the inducing locations |
| MIOST Gabor wavelets | localised bandpass atoms on per-octave grids | diagonal physical variances per scale | matches the *desired* energy spectrum at every scale |

MIOST is the natural choice when (a) you want a per-scale interpretation and (b) you want to add or remove components without retraining. Nyström is the natural choice when you have a kernel you trust and want the cheapest possible approximation. RFF is the natural choice when neither structure helps.

#### Code pointers

- [`massh/basis.py`](file:///home/azureuser/localfiles/jej_vc_snippets/quasigeostrophic_model/massh/basis.py) — `WaveletBasis`, `WaveletParameters`, `WaveletGrid` classes; the canonical reference.
- [`massh/inversion.py`](file:///home/azureuser/localfiles/jej_vc_snippets/quasigeostrophic_model/massh/inversion.py) — variational inversion for the wavelet coefficients.
- [Ocean Data Challenges](https://github.com/ocean-data-challenges) — open MIOST baselines for the 2021/2023/2024 SWOT mapping benchmarks.

---

## 4. Diagnostics — geostrophic currents from $\hat\eta$ (post-hoc)

This is *not* a physics-aware reconstruction step — it is a one-line post-processing of the GP mean using the geostrophic relation in §2b. Compute $\hat\eta$ from the GP, then take finite differences on the grid:

$$
\hat u_g = -\frac{g}{f}\,\partial_y \hat\eta, \qquad \hat v_g = \frac{g}{f}\,\partial_x \hat\eta.
$$

This is what DUACS, MIOST, and OSCAR[^bonjean2002] all publish as their L4 surface-current product. **Zero impact on the GP inversion.** Use the user's local [`derivatives/finite_difference_geo.py`](file:///home/azureuser/localfiles/jej_vc_snippets/derivatives/finite_difference_geo.py) — implements $\partial_x, \partial_y$ on a lat–lon grid with the spherical metric $\partial_y \to \partial_y / R$, $\partial_x \to \partial_x / (R \cos\phi)$ — drop the GP mean field through it and you have operational currents.

If you want *uncertainty* on the currents, take finite differences of each posterior sample $\eta^*_s$ from the pathwise loop and compute the empirical std of the resulting $\{u_{g,s}, v_{g,s}\}$ — same pathwise machinery from 01, no extra cost.

---

## Recap — where each physics knob plugs in

| Physics axis | Knob | Touches in the pathwise formula | Cost vs baseline | gaussx primitive |
|---|---|---|---|---|
| **DATA — subtract** | Pre-process: tides, DAC, MDT | $y$ (one-shot upstream) | Free | None — pure data step |
| **DATA — augment** | Drifter / HF radar / Doppler-altimetry velocities as derivative obs | adds rows to $y$ and $K_{\mathcal{X}\mathcal{X}}$; uses `jax.grad(k)` for the derivative-row entries | Linear in number of velocity obs; usually $\ll m_{\text{altimetry}}$ | `ImplicitKernelOperator` with a typed kernel; `jax.grad` |
| **KERNEL — clever choice** | SQG spectral prior | $S(\omega)$ inside RFF / Bochner | Free — same RFF cost | `pyrox._basis._rff._draw_spectral_frequencies` |
| **KERNEL — multi-scale basis** | MIOST/MASSH Gabor wavelets | $K = \Psi \mathbf{D} \Psi^\top$ via sparse $\Psi$ | Cheaper — same as 01's "fully-RFF" column with sparse $\Psi$ | `gaussx.LowRankUpdate` + Woodbury solve |
| **KERNEL — multi-component** | Block-stacked MIOST components (geostrophic + equatorial + barotropic + IT) | $\Psi$ becomes block-stacked, $\mathbf{D}$ block-diagonal | Linear in number of components (free) | `gaussx.LowRankUpdate` (one per component, summed) |
| **KERNEL — inducing points** | Nyström approximation | $K \approx K_{\mathcal{X}\mathcal{Z}} K_{\mathcal{Z}\mathcal{Z}}^{-1} K_{\mathcal{Z}\mathcal{X}}$ | Same as 01 Route B | `gaussx.nystrom_operator` |
| **DIAGNOSTIC** | Geostrophic currents from $\hat\eta$ | Finite-difference of GP output | Free (post-hoc) | `jej_vc_snippets/derivatives/finite_difference_geo.py` |
| *(separate framework)* | 3D-Var / 4D-Var / DYMOST / BFN-QG / 4DVarNet | Replaces the GP prior with a dynamical or learned regulariser | Different framework — see [04](04_variational_dynamical_priors.md) | Outside the gaussx pathwise stack |

### Recommended sequence

For the next concrete step in this project, taking the menu in increasing order of effort and decreasing order of leverage:

1. **DATA — subtract** (a half-day fix). Confirm that you read CMEMS `sla_filtered`, with DAC and MDT already removed. This is the lowest-effort, highest-impact change — running on uncorrected η will be all wrong regardless of what kernel you use.
2. **KERNEL — multi-scale wavelet basis** on top of `gaussx.LowRankUpdate`. Highest-leverage extension that stays inside the pathwise framework — well-documented, locally-mirrored reference code (MASSH), gives a $\sim 2\times$ resolution improvement over a single-scale Matérn, costs the same as 01's fully-RFF column.
3. **KERNEL — SQG spectral prior** as the spatial component of one of the wavelet sub-frames. One-line variation on (2); especially helpful in mid-latitude winter.
4. **DATA — augment with drifter velocities**. Once the kernel is right, derivative obs add cheap, physically meaningful constraints. Plug into the same `gx.ImplicitKernelOperator` with a typed kernel.

For variational baselines (3D-Var, 4D-Var, DYMOST, BFN-QG, 4DVarNet) that replace the GP prior with a dynamical or learned regulariser, see [04_variational_dynamical_priors.md](04_variational_dynamical_priors.md).

---

[^taburet2019]: Taburet, G., Sanchez-Roman, A., Ballarotta, M., Pujol, M.-I., Legeais, J.-F., Fournier, F., Faugere, Y., Dibarboure, G. (2019). "DUACS DT2018: 25 years of reprocessed sea level altimetry products." *Ocean Science* 15, 1207–1224.

[^ballarotta2019]: Ballarotta, M., Ubelmann, C., Pujol, M.-I., Taburet, G., Fournier, F., Legeais, J.-F., Faugère, Y., Delepoulle, A., Chelton, D., Dibarboure, G., Picot, N. (2019). "On the resolutions of ocean altimetry maps." *Ocean Science* 15, 1091–1109.

[^ubelmann2021]: Ubelmann, C., Carrere, L., Pascual, A., Le Guillou, F., Pujol, M.-I., Taburet, G., Picot, N. (2021). "Reconstructing ocean surface current combining altimetry and future spaceborne Doppler data." *J. Geophys. Res. Oceans* 126, e2020JC016560.

[^ballarotta2023]: Ballarotta, M., et al. (2023). "Improved global sea surface height and current maps from remote sensing and in situ observations." *Earth Syst. Sci. Data* 15, 295–315.

[^dibarboure2025]: Dibarboure, G., et al. (2025). "Integrating wide-swath altimetry data into Level-4 multi-mission maps." *Ocean Science* 21, 63–.

[^held1995]: Held, I., Pierrehumbert, R., Garner, S., Swanson, K. (1995). "Surface quasi-geostrophic dynamics." *J. Fluid Mech.* 282, 1–20.

[^lapeyre2006]: Lapeyre, G., Klein, P. (2006). "Dynamics of the upper oceanic layers in terms of surface quasigeostrophy theory." *J. Phys. Oceanogr.* 36, 165–176.

[^isern2008]: Isern-Fontanet, J., Lapeyre, G., Klein, P., Chapron, B., Hecht, M. (2008). "Three-dimensional reconstruction of oceanic mesoscale currents from surface information." *J. Geophys. Res. Oceans* 113, C09005.

[^gonzalez2014]: González-Haro, C., Isern-Fontanet, J. (2014). "Global ocean current reconstruction from altimetric and microwave SST measurements." *J. Geophys. Res. Oceans* 119, 3378–3391.

[^callies2013]: Callies, J., Ferrari, R. (2013). "Interpreting energy and tracer spectra of upper-ocean turbulence in the submesoscale range (1–200 km)." *J. Phys. Oceanogr.* 43, 2456–2474.

[^rio2014]: Rio, M.-H., Mulet, S., Picot, N. (2014). "Beyond GOCE for the ocean circulation estimate: synergetic use of altimetry, gravimetry, and in situ data provides new insight into geostrophic and Ekman currents." *Geophys. Res. Lett.* 41, 8918–8925.

[^bonjean2002]: Bonjean, F., Lagerloef, G. (2002). "Diagnostic model and analysis of the surface currents in the tropical Pacific Ocean." *J. Phys. Oceanogr.* 32, 2938–2954.

[^mulet2021]: Mulet, S., Rio, M.-H., et al. (2021). "The new CNES-CLS18 global mean dynamic topography." *Ocean Science* 17, 789–808.

[^jousset2025]: Jousset, S., Mulet, S., Rio, M.-H., et al. (2025). "New global MDT CNES-CLS-22." *Earth Syst. Sci. Data* 18, 2285– (in press).

[^ardhuin2019]: Ardhuin, F., et al. (2019). "SKIM, a candidate satellite mission exploring global ocean currents and waves." *Frontiers in Marine Science* 6, 209.

[^lyard2021]: Lyard, F. H., Allain, D. J., Cancet, M., Carrère, L., Picot, N. (2021). "FES2014 global ocean tide atlas." *Ocean Science* 17, 615–649.

[^carrere2003]: Carrère, L., Lyard, F. (2003). "Modeling the barotropic response of the global ocean to atmospheric wind and pressure forcing — comparisons with observations." *Geophys. Res. Lett.* 30, 1275.
