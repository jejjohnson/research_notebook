---
title: "Beer–Lambert radiative transfer and the HAPI absorption LUT"
---

# Beer–Lambert radiative transfer and the HAPI absorption LUT

This note derives the forward model implemented by `plume_simulation.hapi_lut`: a line-by-line Voigt-profile sum of molecular absorption lines, packaged into a look-up table (LUT) of the absorption cross-section $\sigma(\nu; T, p)$, and applied to a single-layer Beer–Lambert transmittance calculation. The LUT is the precomputed object; the forward model is a $\mathcal{O}(\log N)$ interpolation followed by a single `exp`.

The theory follows Gordon et al. [^gordon2022] for the line-by-line physics (intensity, shape, and cross-section) and Rothman et al. [^rothman2013] for the HITRAN database conventions. LUT design follows the decoupled-VOD recipe used in operational SWIR retrievals (e.g. Frankenberg et al. [^frankenberg2005]).

[^gordon2022]: Gordon, I. E. et al. (2022). *The HITRAN2020 molecular spectroscopic database*. JQSRT **277**, 107949.
[^rothman2013]: Rothman, L. S. et al. (2013). *HITRAN: 40 years and counting*. JQSRT **130**, 4–50.
[^frankenberg2005]: Frankenberg, C. et al. (2005). *Iterative maximum a posteriori (IMAP)-DOAS for retrieval of strongly absorbing trace gases*. Atmos. Chem. Phys. **5**, 9–22.

## 1. Physical setup

Consider a pencil of sunlight traversing a plane-parallel atmospheric layer of vertical thickness $L_{\mathrm{vert}}$ containing a single absorbing gas at temperature $T$ [K], pressure $p$ [atm], and volume mixing ratio $\mathrm{VMR}$. We want the transmittance $\tau(\nu) \in [0, 1]$ — the fraction of photons at wavenumber $\nu$ that pass through untouched — as a function of the atmospheric state.

Two viewing angles enter: the solar zenith angle $\mathrm{SZA}$ (light in) and the viewing zenith angle $\mathrm{VZA}$ (light out to the sensor after ground reflection). The plane-parallel two-way *air-mass factor*

$$
\mathrm{AMF}(\mathrm{SZA}, \mathrm{VZA}) \;=\; \frac{1}{\cos \mathrm{SZA}} + \frac{1}{\cos \mathrm{VZA}}
$$

converts the vertical path length to the slant path length actually traversed by the photon; it is valid for zenith angles below $\sim 75^\circ$ and collapses to $\mathrm{AMF} = 2$ at nadir (SZA = VZA = 0).

## 2. Line-by-line absorption cross-section

The HITRAN database catalogues every rovibrational transition $i$ of every atmospherically relevant isotopologue by a central wavenumber $\nu_{0,i}$ [cm⁻¹], a reference line strength $S_i(T_{\mathrm{ref}})$ at $T_{\mathrm{ref}} = 296$ K [cm⁻¹/(molecule·cm⁻²)], and a lower-state energy $E_i''$ [cm⁻¹]. For a given $T$ each line's strength is the reference value scaled by the Boltzmann + stimulated-emission factors,

$$
S_i(T) \;=\; S_i(T_{\mathrm{ref}}) \cdot
\frac{Q(T_{\mathrm{ref}})}{Q(T)} \cdot
\frac{\exp(-c_2 E_i'' / T)}{\exp(-c_2 E_i'' / T_{\mathrm{ref}})} \cdot
\frac{1 - \exp(-c_2 \nu_{0,i} / T)}{1 - \exp(-c_2 \nu_{0,i} / T_{\mathrm{ref}})},
$$

where $Q(T)$ is the Total Internal Partition Sum (TIPS-2021 tables are bundled with HAPI) and $c_2 = hc/k_B \approx 1.4388$ K·cm is the second radiation constant.

Each line also has a shape function $f_i(\nu; T, p)$ determined by two broadening mechanisms:

- **Doppler broadening** — thermal translation gives a Gaussian shape with half-width $\gamma_D \propto \nu_{0,i} \sqrt{T}$.
- **Pressure broadening** — collisions give a Lorentzian shape with half-width $\gamma_L \propto p \cdot (T_{\mathrm{ref}}/T)^{n_i} \cdot (\mathrm{VMR}_{\mathrm{self}} \gamma_{\mathrm{self},i} + \mathrm{VMR}_{\mathrm{air}} \gamma_{\mathrm{air},i})$.

The real line shape is the convolution of the two — the **Voigt profile**,

$$
f_{V,i}(\nu; T, p) \;=\; \int_{-\infty}^{\infty} G_i(\nu'; T) \; L_i(\nu - \nu'; T, p) \; d\nu',
$$

evaluated by HAPI via the Humlíček algorithm. Summing line strength × shape over all lines within the requested wavenumber window gives the absorption cross-section

$$
\boxed{\;\sigma(\nu; T, p) \;=\; \sum_{i} S_i(T) \; f_{V,i}(\nu; T, p)\;}
\qquad [\mathrm{cm}^2 / \mathrm{molecule}].
$$

This is the core quantity HAPI computes via `absorptionCoefficient_Voigt` (name notwithstanding, the `HITRAN_units=True` option returns $\sigma$, not $\alpha$).

## 3. Absorption coefficient and Beer–Lambert

The bulk absorption coefficient follows from $\sigma$ and the local number density of the absorbing gas. From the ideal-gas law,

$$
N_{\mathrm{total}}(T, p) \;=\; \frac{p}{k_B\, T}, \qquad
N_{\mathrm{gas}} \;=\; \mathrm{VMR} \cdot N_{\mathrm{total}}
\qquad [\mathrm{molecules}/\mathrm{cm}^3],
$$

and the absorption coefficient is the product

$$
\alpha(\nu; T, p, \mathrm{VMR}) \;=\; \sigma(\nu; T, p) \cdot N_{\mathrm{gas}}
\qquad [\mathrm{cm}^{-1}],
$$

with units checking out: $(\mathrm{cm}^2/\mathrm{molecule}) \cdot (\mathrm{molecules}/\mathrm{cm}^3) = \mathrm{cm}^{-1}$.

The Beer–Lambert law states that each infinitesimal path $dl$ attenuates the intensity by a factor $\exp(-\alpha\, dl)$. Integrating over a homogeneous layer and applying the two-way AMF gives

$$
\boxed{\;\tau(\nu) \;=\; \exp\!\big[-\alpha(\nu; T, p, \mathrm{VMR}) \cdot L_{\mathrm{vert}} \cdot \mathrm{AMF}\big]\;}.
$$

This is the single-layer forward model in `plume_simulation.hapi_lut.beers.beers_law_from_lut`.

## 4. Why a look-up table

Evaluating $\sigma(\nu; T, p)$ at runtime for every pixel is prohibitively expensive: each call sums thousands of Voigt profiles across a dense $\nu$ grid. But $\sigma$ is a smooth function of $(T, p)$ at fixed $\nu$ — Doppler broadening scales as $\sqrt{T}$, pressure broadening as $p \cdot T^{-n}$, and line strength $S_i(T)$ is entire. **Precompute $\sigma$ on a coarse $(T, p)$ grid once, interpolate at runtime.**

The natural axes are:

- **Wavenumber $\nu$** — determined by the instrument's native resolution. HAPI works at $\Delta\nu \leq 0.01$ cm⁻¹ for hyperspectral applications (TROPOMI, EMIT); coarser is fine for tutorial work.
- **Temperature $T$** — spans the upper troposphere ($\sim 200$ K) to the surface ($\sim 320$ K), with $\sim 20$ K steps sufficient for bilinear-interpolation error below 1 %.
- **Pressure $p$** — spans the upper troposphere ($\sim 0.1$ atm) to the surface ($\sim 1.0$ atm), logarithmic or uniform depending on where accuracy matters most.

Viewing geometry ($\mathrm{SZA}, \mathrm{VZA}$) is **not** a LUT axis: AMF multiplies $\alpha \cdot L_{\mathrm{vert}}$ *after* the LUT lookup, so a single cross-section LUT serves every satellite geometry.

## 5. Differential Beer–Lambert — plume enhancement

For plume retrievals from point sources (landfills, oil-and-gas, industrial stacks) we care about the *enhancement* above the regional background, not the absolute column. Let $\mathrm{VMR}_{\mathrm{bg}}$ be the background and $\mathrm{VMR}_{\mathrm{tot}} = (1 + \varepsilon) \, \mathrm{VMR}_{\mathrm{bg}}$ the plume pixel, where $\varepsilon$ is the fractional enhancement.

The optical depth decomposes additively,

$$
\tau_{\mathrm{total}}(\nu) \;=\; \tau_{\mathrm{bg}}(\nu) \cdot \tau_{\mathrm{enh}}(\nu),
\qquad
\tau_{\mathrm{enh}}(\nu) \;=\; \frac{\tau_{\mathrm{total}}(\nu)}{\tau_{\mathrm{bg}}(\nu)},
$$

because Beer–Lambert turns a sum of absorption coefficients into a product of transmittances. On the **measurement** side, dividing the plume pixel radiance by the background pixel radiance cancels every factor that is common to both pixels:

$$
\frac{I_{\mathrm{plume}}(\nu)}{I_{\mathrm{bg}}(\nu)} \;=\;
\frac{I_0(\nu) \, \rho_{\mathrm{surf}}(\nu) \, \tau_{\mathrm{atm}}(\nu) \, \tau_{\mathrm{bg}}(\nu) \, \tau_{\mathrm{enh}}(\nu)}{I_0(\nu) \, \rho_{\mathrm{surf}}(\nu) \, \tau_{\mathrm{atm}}(\nu) \, \tau_{\mathrm{bg}}(\nu)}
\;=\; \tau_{\mathrm{enh}}(\nu).
$$

Solar irradiance $I_0$, broadband surface albedo $\rho_{\mathrm{surf}}$, aerosol / non-target atmospheric transmittance $\tau_{\mathrm{atm}}$, and the target-gas background all drop out of the ratio — leaving only the *narrow-band* absorption features of the plume enhancement. This cancellation is what makes plume retrievals from space possible even when the absolute column retrieval is dominated by nuisance parameters.

The retrieval inverts the LUT-derived simulated ratio against the measured ratio:

$$
\hat\varepsilon \;=\; \arg\min_\varepsilon \;
\big\| I_{\mathrm{plume}} / I_{\mathrm{bg}} \;-\; \tau_{\mathrm{enh}}(\varepsilon) \big\|^2.
$$

This is the forward-model kernel of matched-filter and optimal-estimation retrievals downstream (`jej_vc_snippets/methane_retrieval/matched_filter_beerslaw.py`, `lut_3dvar_beers.py`).

## 6. Assumptions and caveats

The forward model above inherits a stack of approximations, in roughly decreasing order of severity:

1. **Absorption-only radiative transfer.** Scattering by molecules (Rayleigh), aerosols, and clouds is ignored. For hazy or cloudy pixels a full scalar or vector RTM (e.g. VLIDORT, SASKTRAN) is needed.
2. **Single homogeneous layer.** Real atmospheres have $T(z)$, $p(z)$, $\mathrm{VMR}(z)$ profiles. Operational retrievals discretize into 20–40 layers and sum layer-wise vertical optical depths (VODs) — the same LUT machinery applies per layer.
3. **Plane-parallel geometry.** Breaks down past $\sim 75^\circ$ zenith; the sphericity correction is a $\sec$-replacement in the AMF.
4. **Local thermodynamic equilibrium (LTE).** Holds in the troposphere and lower stratosphere; fails at very high altitudes where radiative excitation competes with collisional excitation.
5. **Line-mixing neglected.** HAPI's `_Voigt` does not account for line mixing, which matters at very high pressure for some species (CO₂ 4.3 µm band, for instance). The `_HT` (Hartmann–Tran) profile is available but costlier.
6. **Diluent composition.** We pin `{'air': 1-VMR, 'self': VMR}`, using dry-air broadening coefficients. Humid conditions change $\gamma_L$ slightly via $\gamma_{\mathrm{H_2O}}$ — second-order for most retrievals.

## 7. Software surface

Three functions carry the physics:

- `generator.compute_absorption_lut(gas_config, grid_config)` — evaluates $\sigma(\nu; T, p)$ on the full LUT grid by calling `hapi.absorptionCoefficient_Voigt` for each $(T, p)$ knot.
- `beers.beers_law_from_lut(ds, vmr, T, p, L_vert, SZA, VZA)` — interpolates the LUT at $(T, p)$, multiplies by $N_{\mathrm{gas}}$ to get $\alpha$, applies AMF and exp, returns $\tau(\nu)$.
- `beers.plume_ratio_spectrum(ds, vmr_bg, vmr_tot, ...)` — the differential form $\tau_{\mathrm{tot}}(\nu) / \tau_{\mathrm{bg}}(\nu)$ used by the enhancement retrieval in [03_beers_law_with_lut.ipynb](03_beers_law_with_lut.ipynb).

The three subsequent notebooks walk through building the LUT ([01](01_hapi_lut_ch4.ipynb)), extending to multiple gases ([02](02_hapi_lut_multigas.ipynb)), and using it as the forward model of a plume-enhancement retrieval ([03](03_beers_law_with_lut.ipynb)).
