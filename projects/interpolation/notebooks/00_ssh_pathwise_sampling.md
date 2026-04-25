---
title: "SSH reconstruction via GP pathwise sampling"
---

# SSH Reconstruction via GP Pathwise Sampling

## The Inverse Problem

**Goal:** Reconstruct the SSH anomaly field $\eta$ on a regular grid at a fixed target time $t$, using satellite altimeter tracks from a temporal window $[t - \tau, t + \tau]$. This is a **pure static inverse problem** — no dynamics, no time-stepping. We treat the SSH field as approximately stationary over the window and let the kernel handle temporal discounting automatically.

**Why pool $t \pm \tau$?** A single Jason-3 pass covers roughly 1–5% of the Mediterranean at any given time, with a 10-day repeat cycle. A single Sentinel-6 pass is similar. Pooling passes from nearby times dramatically increases spatial coverage. The cost is a temporal mismatch — observations at $t \pm \tau$ represent the field at slightly different states. Rather than manually inflating error variances (as 3D-Var does), the GP handles this gracefully: a spatiotemporal kernel naturally assigns lower covariance to observation pairs that are far apart in time, so temporally distant tracks are automatically downweighted during inference.

---

## Domain and State

| Symbol | Shape | Meaning |
|---|---|---|
| $n_{lat}, n_{lon}$ | scalars | Number of grid cells in latitude and longitude |
| $n = n_{lat} \times n_{lon}$ | scalar | Total grid cells — e.g. $0.1°$ grid over Med $\approx 10^5$ |
| $\eta \in \mathbb{R}^n$ | $(n,)$ | SSH anomaly field at time $t$, vectorised row-major over the grid |

The field $\eta$ is what we want to recover. It is unobserved everywhere except along sparse satellite swaths.

---

## Observations

Satellite altimeters sample SSH along 1D ground tracks — narrow strips that cross the Mediterranean at oblique angles. We pool three groups of passes:

| Symbol | Shape | Meaning |
|---|---|---|
| $y_- \in \mathbb{R}^{m_-}$ | $(m_-,)$ | Along-track SSH anomaly measurements at time $t - \tau$ |
| $y_0 \in \mathbb{R}^{m_0}$ | $(m_0,)$ | Along-track SSH anomaly measurements at time $t$ |
| $y_+ \in \mathbb{R}^{m_+}$ | $(m_+,)$ | Along-track SSH anomaly measurements at time $t + \tau$ |
| $y = [y_-;\, y_0;\, y_+]$ | $(m,)$ | Full concatenated observation vector, $m = m_- + m_0 + m_+$ |

Typical values for the Mediterranean: $m_\pm \sim 500$–$2000$ per pass group, $m \sim 3000$–$8000$ total. Each observation $y_i$ has an associated position $(lon_i, lat_i)$ and time $t_i \in \{t-\tau, t, t+\tau\}$.

---

## The GP Prior

A Gaussian process is a distribution over functions. Writing $f: \mathcal{X} \to \mathbb{R}$ where $\mathcal{X}$ is the joint (space, time) domain:

$$f \sim \mathrm{GP}(0, k_\theta)$$

This means: for any finite collection of inputs $\{(x_i, t_i)\}_{i=1}^m$, the vector of function values $[f(x_1, t_1), \ldots, f(x_m, t_m)]$ is jointly Gaussian with zero mean and covariance matrix whose $(i,j)$ entry is $k_\theta((x_i, t_i), (x_j, t_j))$.

The **kernel** $k_\theta$ is the sole prior specification — it encodes everything we believe about the SSH field before seeing data: how smooth it is, how quickly correlations decay in space and time, and what its overall amplitude is.

Zero mean is appropriate here because we work with SSH **anomalies** (mean SSH removed), so the prior mean is zero by construction.

---

## Training and Prediction Inputs

The GP operates directly on (lon, lat, time) coordinates — no grid required at training time. Define:

$$\mathcal{X} = \begin{bmatrix} lon_1^- & lat_1^- & t-\tau \\ \vdots & \vdots & \vdots \\ lon_{m_-}^- & lat_{m_-}^- & t-\tau \\ lon_1^0 & lat_1^0 & t \\ \vdots & \vdots & \vdots \\ lon_{m_+}^+ & lat_{m_+}^+ & t+\tau \end{bmatrix} \in \mathbb{R}^{m \times 3}$$

Each row is one satellite observation's (lon, lat, time) coordinate. The prediction inputs are all grid cells at time $t$:

$$\mathcal{X}^* = \begin{bmatrix} lon_1^* & lat_1^* & t \\ \vdots & \vdots & \vdots \\ lon_n^* & lat_n^* & t \end{bmatrix} \in \mathbb{R}^{n \times 3}$$

All prediction points share the same time $t$ — we want the reconstructed field at the target time only.

---

## Kernel — Encoding Prior Physics

Use a **separable spatiotemporal kernel** — separable means the spatial and temporal contributions multiply:

$$k_\theta\!\left((x, t),\, (x', t')\right) = \sigma_\eta^2 \cdot k_s(x, x';\, L_s) \cdot k_t(t, t';\, L_t)$$

### Spatial Kernel — Matérn-3/2

The Matérn-3/2 kernel on the sphere (using great-circle distance $d$):

$$k_s(x, x';\, L_s) = \left(1 + \frac{\sqrt{3}\,d(x,x')}{L_s}\right)\exp\!\left(-\frac{\sqrt{3}\,d(x,x')}{L_s}\right)$$

**Why Matérn-3/2 and not Gaussian?** The squared-exponential (Gaussian) kernel produces infinitely differentiable sample paths — unrealistically smooth for SSH. Matérn-3/2 produces once mean-square differentiable paths, which is physically appropriate for mesoscale SSH fields that have sharp fronts and eddy boundaries. Matérn-5/2 (twice differentiable) is also a reasonable choice.

**Why great-circle distance?** The Mediterranean spans enough longitude (~40°) that planar Euclidean distance is inaccurate. Great-circle distance respects the spherical geometry.

### Temporal Kernel — Ornstein-Uhlenbeck

$$k_t(t, t';\, L_t) = \exp\!\left(-\frac{|t - t'|}{L_t}\right)$$

The OU kernel (Matérn-1/2) produces exponentially decaying temporal correlations. **The key physical consequence:** observations at $t \pm \tau$ contribute to the reconstruction at time $t$ with a weight of $\exp(-\tau / L_t)$ — automatically. If $L_t = 10$ days and $\tau = 3$ days, this factor is $\exp(-0.3) \approx 0.74$, meaning observations 3 days away carry about 74% of the weight of contemporaneous observations. No manual tuning of temporal weights needed.

### Hyperparameters

| Parameter | Meaning | Typical value (Med) |
|---|---|---|
| $\sigma_\eta^2$ | SSH anomaly marginal variance | $\sim 100\,\text{cm}^2$ |
| $L_s$ | Spatial decorrelation length | $\sim 100\,\text{km}$ |
| $L_t$ | Temporal decorrelation timescale | $\sim 10\,\text{days}$ |
| $\sigma_{obs}^2$ | Altimeter noise variance | $\sim 4\,\text{cm}^2$ |

---

## Key Matrices

These are the four objects that define all subsequent computation.

**Gram matrix at training points:**

$$K_{\mathcal{X}\mathcal{X}} \in \mathbb{R}^{m \times m}, \qquad [K_{\mathcal{X}\mathcal{X}}]_{ij} = k_\theta(\mathcal{X}_i, \mathcal{X}_j)$$

This is a **dense symmetric PSD matrix**. Entry $(i,j)$ is the prior covariance between observations $i$ and $j$ — it encodes how similar the SSH field is expected to be at those two (location, time) pairs. Observations close in space and time will have high covariance; observations far apart will have near-zero covariance. The diagonal $[K_{\mathcal{X}\mathcal{X}}]_{ii} = \sigma_\eta^2$ (the full prior variance at each point).

**Noisy Gram matrix:**

$$C = K_{\mathcal{X}\mathcal{X}} + \sigma_{obs}^2 I_m \in \mathbb{R}^{m \times m}$$

Adding $\sigma_{obs}^2 I_m$ accounts for altimeter instrument noise — it regularises the Gram matrix (ensuring it is strictly PD) and prevents overfitting to noisy observations. $C$ is the object we invert — it represents the total observation-space covariance including noise.

**Cross-covariance matrix:**

$$K_{\mathcal{X}^*\mathcal{X}} \in \mathbb{R}^{n \times m}, \qquad [K_{\mathcal{X}^*\mathcal{X}}]_{ij} = k_\theta(\mathcal{X}^*_i, \mathcal{X}_j)$$

Row $i$ of this matrix is the prior covariance between grid cell $i$ (at time $t$) and observation $j$ (at its (location, time)). This is the **gain numerator** — it determines how strongly each observation influences each grid cell. A grid cell far from all tracks will have small entries across its entire row, meaning observations barely update it. A grid cell directly under a track will have large entries for nearby observations.

**Prior variance at prediction points (diagonal only):**

$$[K_{\mathcal{X}^*\mathcal{X}^*}]_{ii} = k_\theta(\mathcal{X}^*_i, \mathcal{X}^*_i) = \sigma_\eta^2 \quad \forall i$$

Since all prediction points share time $t$ and the spatial kernel equals 1 at zero distance, the prior variance is $\sigma_\eta^2$ uniformly. The full matrix $K_{\mathcal{X}^*\mathcal{X}^*} \in \mathbb{R}^{n \times n}$ is never formed.

---

## Posterior Derivation

The GP prior and the observation model:

$$y = f(\mathcal{X}) + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \sigma_{obs}^2 I_m)$$

together define a joint Gaussian over $(f(\mathcal{X}^*), y)$:

$$\begin{bmatrix} f(\mathcal{X}^*) \\ y \end{bmatrix} \sim \mathcal{N}\!\left( \begin{bmatrix} 0 \\ 0 \end{bmatrix},\ \begin{bmatrix} K_{\mathcal{X}^*\mathcal{X}^*} & K_{\mathcal{X}^*\mathcal{X}} \\ K_{\mathcal{X}\mathcal{X}^*} & C \end{bmatrix} \right)$$

Applying the Gaussian conditioning formula from the unifying framework with:

$$\Sigma_{xx} = K_{\mathcal{X}^*\mathcal{X}^*}, \quad \Sigma_{xz} = K_{\mathcal{X}^*\mathcal{X}}, \quad \Sigma_{zz} = C$$

the posterior is:

$$f(\mathcal{X}^*) \mid y \;\sim\; \mathcal{N}\!\left(\mu_{f|y},\; \Sigma_{f|y}\right)$$

$$\mu_{f|y} = K_{\mathcal{X}^*\mathcal{X}}\, C^{-1}\, y \qquad (n,)$$

$$\Sigma_{f|y} = K_{\mathcal{X}^*\mathcal{X}^*} - K_{\mathcal{X}^*\mathcal{X}}\, C^{-1}\, K_{\mathcal{X}\mathcal{X}^*} \qquad (n, n)$$

The posterior mean $\mu_{f|y}$ is the **minimum mean-square error estimate** of the SSH field — the best single map you can produce. The posterior covariance $\Sigma_{f|y}$ describes the joint uncertainty over all grid cells — but at $(n \times n)$ it is completely intractable to store or compute for $n = 10^5$.

**Practical posterior quantities:** We never compute $\Sigma_{f|y}$ directly. Instead:

- **Pointwise variance** (diagonal only): $[\Sigma_{f|y}]_{ii} = \sigma_\eta^2 - [K_{\mathcal{X}^*\mathcal{X}}\, C^{-1}\, K_{\mathcal{X}\mathcal{X}^*}]_{ii}$ — computed row by row, cost $\mathcal{O}(nm^2)$.
- **Posterior samples** via pathwise update — this is the subject of the next section.

---

## The Pathwise Update — Correcting Prior Samples

The standard posterior mean formula above requires $C^{-1}y$ — a single linear solve. That gives you a point estimate. To get **samples from the posterior**, the naive approach would be to Cholesky-factorise the $(n \times n)$ posterior covariance $\Sigma_{f|y}$ — completely intractable.

**Matheron's rule** (Wilson et al., 2020) provides an exact alternative. The key insight is that for Gaussian random variables, the conditional distribution can be written as a **linear correction to a joint prior sample**:

$$f(\mathcal{X}^*) \mid y \;\overset{d}{=}\; \tilde{f}(\mathcal{X}^*) + K_{\mathcal{X}^*\mathcal{X}}\, C^{-1}\,\bigl(y - \tilde{f}(\mathcal{X})\bigr)$$

where $(\tilde{f}(\mathcal{X}^*), \tilde{f}(\mathcal{X}))$ is a **joint prior sample** drawn consistently from $\mathrm{GP}(0, k_\theta)$ at both the prediction and training inputs. This equality is in distribution — the right-hand side is an exact draw from the posterior $p(f \mid y)$.

**Why does this work?** Because the correction term is the posterior mean applied not to $y$ but to the **innovation** $y - \tilde{f}(\mathcal{X})$. Since $(\tilde{f}(\mathcal{X}^*), \tilde{f}(\mathcal{X}))$ are jointly Gaussian, the correction is linear, and the result has exactly the right mean and covariance.

**Why is this better than sampling from $\Sigma_{f|y}$ directly?**

- Sampling from $\Sigma_{f|y}$ requires an $(n \times n)$ Cholesky: $\mathcal{O}(n^3)$ compute, $\mathcal{O}(n^2)$ storage — completely intractable for $n = 10^5$.
- The pathwise update only requires an $(m \times m)$ Cholesky (computed once) plus an $\mathcal{O}(r(m+n))$ prior sample generation and an $\mathcal{O}(nm)$ mat-vec. All operations scale in $m$ and $n$ independently.

---

## Prior Sample via Random Fourier Features

To apply Matheron's rule we need a joint prior sample $(\tilde{f}(\mathcal{X}^*), \tilde{f}(\mathcal{X}))$ drawn consistently from $\mathrm{GP}(0, k_\theta)$.

Bochner's theorem guarantees that a stationary kernel $k_\theta(x, x') = k_\theta(x - x')$ can be written as the Fourier transform of a spectral density $p(\omega)$:

$$k_\theta(x - x') = \int p(\omega)\, e^{i\omega^\top(x - x')}\, d\omega$$

This motivates the **Random Fourier Features (RFF)** approximation. Sample $r$ frequency vectors $\{\omega_j\}_{j=1}^r$ from $p(\omega)$ and draw uniform phase offsets $\{b_j\}_{j=1}^r \sim \mathrm{Uniform}[0, 2\pi]$. Define the feature map:

$$\phi_j(x) = \sqrt{\frac{2\sigma_\eta^2}{r}}\,\cos(\omega_j^\top x + b_j)$$

Then the approximate prior sample is:

$$\tilde{f}(x, t) = \sum_{j=1}^{r} w_j\, \phi_j^s(x)\, \phi_j^t(t), \qquad w_j \overset{\text{iid}}{\sim} \mathcal{N}(0, 1)$$

where $\phi_j^s$ and $\phi_j^t$ are RFF features for the spatial and temporal kernels respectively (drawn once from their spectral densities). The weights $\{w_j\}$ are **fixed for a given sample draw** — evaluating $\tilde{f}$ at any point is then a deterministic computation.

**Consistency:** because $\{w_j\}$ are fixed, evaluating $\tilde{f}$ at $\mathcal{X}$ and $\mathcal{X}^*$ uses the **same weights** — this is what makes the prior sample joint and consistent, which is required for Matheron's rule to produce exact posterior draws.

**How many features $r$?** The RFF approximation introduces a bias of $\mathcal{O}(1/\sqrt{r})$ in the kernel approximation. For SSH reconstruction, $r \sim 1000$–$4000$ is typically sufficient. The approximation error in the posterior sample decreases as $r$ increases; the compute cost grows as $\mathcal{O}(r(m+n))$ per sample.

---

## Full Algorithm

### Offline (once per time window)

**Step 1 — Build the Gram matrix:**

$$[K_{\mathcal{X}\mathcal{X}}]_{ij} = \sigma_\eta^2 \cdot k_s\!\left(x_i, x_j;\, L_s\right) \cdot k_t\!\left(t_i, t_j;\, L_t\right) \qquad (m, m)$$

Evaluate $m^2$ kernel calls. Each call computes a great-circle distance and evaluates Matérn-3/2 × OU — $\mathcal{O}(1)$ per pair.

**Step 2 — Form and Cholesky-factorise $C$:**

$$C = K_{\mathcal{X}\mathcal{X}} + \sigma_{obs}^2 I_m \qquad (m, m)$$

$$C = LL^\top, \qquad L \in \mathbb{R}^{m \times m} \text{ lower triangular}$$

This is the **dominant computation**. $L$ is stored and reused for all subsequent solves and for every posterior sample.

**Step 3 — Compute the posterior mean coefficient vector:**

$$\alpha = C^{-1}y = L^{-\top}L^{-1}y \qquad (m,)$$

Two triangular solves. $\alpha$ is the vector of **dual weights** — entry $\alpha_i$ is the weight assigned to observation $i$ in the posterior mean. Observations with high $\alpha_i$ are highly informative; those in dense clusters will have smaller $\alpha_i$ due to redundancy.

**Step 4 — Build the cross-covariance matrix:**

$$[K_{\mathcal{X}^*\mathcal{X}}]_{ij} = \sigma_\eta^2 \cdot k_s\!\left(x_i^*, x_j;\, L_s\right) \cdot k_t\!\left(t, t_j;\, L_t\right) \qquad (n, m)$$

Evaluate $nm$ kernel calls. Since all prediction points share time $t$, the temporal factor $k_t(t, t_j; L_t)$ takes only three distinct values (one per time group $t-\tau, t, t+\tau$) — this allows factoring:

$$K_{\mathcal{X}^*\mathcal{X}} = \begin{bmatrix} e^{-\tau/L_t}\, K^*_s(-, L_s) & K^*_s(0, L_s) & e^{-\tau/L_t}\, K^*_s(+, L_s) \end{bmatrix}$$

where $K^*_s(\star, L_s) \in \mathbb{R}^{n \times m_\star}$ is the purely spatial cross-covariance matrix between all grid points and the $\star$-group track locations. This factorisation reduces the number of unique kernel evaluations to $n(m_- + m_0 + m_+)$ spatial evaluations plus $m$ cheap exponential evaluations.

**Step 5 — Compute the posterior mean field:**

$$\mu_{f|y} = K_{\mathcal{X}^*\mathcal{X}}\, \alpha \qquad (n,)$$

A dense mat-vec: $(n \times m)$ times $(m \times 1)$. This gives the **interpolated SSH field** on the grid at time $t$.

### Per Posterior Sample (repeat $S$ times)

**Step 6 — Draw RFF weights:**

$$w_j \overset{\text{iid}}{\sim} \mathcal{N}(0, 1), \quad j = 1, \ldots, r$$

Sample spatial frequencies $\omega_j^s$ from the Matérn-3/2 spectral density, temporal frequencies $\omega_j^t$ from the OU spectral density (both done once, reused across samples if desired, or redrawn for fresh samples).

**Step 7 — Evaluate prior sample at training and prediction inputs:**

$$\tilde{f}(\mathcal{X}) \in \mathbb{R}^m, \qquad \tilde{f}(\mathcal{X}^*) \in \mathbb{R}^n$$

Each evaluation: form the $(m+n) \times r$ feature matrix $\Phi$, then multiply by $w$. Cost $\mathcal{O}(r(m+n))$.

**Step 8 — Compute the innovation:**

$$\delta = y - \tilde{f}(\mathcal{X}) \qquad (m,)$$

This is the residual between what the prior sample predicts at the track locations and what was actually observed. If the prior sample happened to pass through the data exactly, $\delta = 0$ and no correction is needed — the prior sample is already a valid posterior draw (this never happens in practice).

**Step 9 — Solve the correction coefficient vector:**

$$\beta = C^{-1}\delta = L^{-\top}L^{-1}\delta \qquad (m,)$$

Two triangular solves using the already-computed $L$. This is $\mathcal{O}(m^2)$ — cheap because $L$ is already available.

**Step 10 — Apply the pathwise correction:**

$$\boxed{\eta^*_s = \tilde{f}(\mathcal{X}^*) + K_{\mathcal{X}^*\mathcal{X}}\,\beta \qquad (n,)}$$

One dense mat-vec: $(n \times m)$ times $(m \times 1)$. The result $\eta^*_s$ is one **exact posterior sample** of the SSH field on the grid at time $t$.

---

## What the Output Looks Like

After $S$ samples $\{\eta^*_s\}_{s=1}^S$:

**Posterior mean estimate** (best single map):

$$\hat{\eta} = \frac{1}{S}\sum_{s=1}^S \eta^*_s \approx \mu_{f|y} \qquad (n,)$$

For large $S$ this converges to the exact posterior mean $K_{\mathcal{X}^*\mathcal{X}}\alpha$ computed in Step 5 — but you may as well use Step 5 directly.

**Pointwise posterior standard deviation** (uncertainty map):

$$\hat{\sigma}(x_i^*) = \sqrt{\frac{1}{S-1}\sum_{s=1}^S (\eta^*_s(x_i^*) - \hat{\eta}(x_i^*))^2} \qquad (n,)$$

Large $\hat{\sigma}$ where tracks are sparse; small $\hat{\sigma}$ under dense track coverage. This is physically interpretable — you know exactly where the reconstruction is trustworthy.

**Physically consistent realisations:** Each $\eta^*_s$ is a plausible SSH field — spatially smooth, consistent with the kernel's length scale, and passing through the observations within noise. These can be passed directly to downstream analyses (eddy detection, geostrophic current estimation) to propagate uncertainty.

---

## Storage Analysis

All costs are at float64 (8 bytes). Let $n = 10^5$, $m = 5 \times 10^3$, $r = 2 \times 10^3$, $S = 100$.

| Object | Shape | Dtype | Storage formula | Size |
|---|---|---|---|---|
| $\mathcal{X}$ | $(m, 3)$ | float64 | $24m$ bytes | $120\,\text{KB}$ |
| $y$ | $(m,)$ | float64 | $8m$ bytes | $40\,\text{KB}$ |
| $K_{\mathcal{X}\mathcal{X}}$ | $(m, m)$ | float64 | $8m^2$ bytes | $200\,\text{MB}$ |
| $C = K_{\mathcal{X}\mathcal{X}} + \sigma^2 I$ | $(m, m)$ | float64 | $8m^2$ bytes | $200\,\text{MB}$ — can overwrite $K_{\mathcal{X}\mathcal{X}}$ |
| $L$ (Cholesky factor) | $(m, m)$ | float64 | $4m^2$ bytes (triangular) | $100\,\text{MB}$ |
| $\alpha = C^{-1}y$ | $(m,)$ | float64 | $8m$ bytes | $40\,\text{KB}$ |
| $K_{\mathcal{X}^*\mathcal{X}}$ | $(n, m)$ | float64 | $8nm$ bytes | **$4\,\text{GB}$** |
| RFF spectral params | $(r, 3)$ | float64 | $24r$ bytes | $48\,\text{KB}$ |
| $\tilde{f}(\mathcal{X})$ | $(m,)$ | float64 | $8m$ bytes | $40\,\text{KB}$ |
| $\tilde{f}(\mathcal{X}^*)$ | $(n,)$ | float64 | $8n$ bytes | $800\,\text{KB}$ |
| $\beta = C^{-1}\delta$ | $(m,)$ | float64 | $8m$ bytes | $40\,\text{KB}$ |
| $\eta^*_s$ (one sample) | $(n,)$ | float64 | $8n$ bytes | $800\,\text{KB}$ |
| $\{\eta^*_s\}_{s=1}^S$ (all samples) | $(S, n)$ | float64 | $8Sn$ bytes | $80\,\text{MB}$ |

**Total storage:** dominated by $K_{\mathcal{X}^*\mathcal{X}}$ at $\mathcal{O}(nm)$ — approximately **4 GB** for these parameters.

$$\text{Storage} = \mathcal{O}(m^2 + nm)$$

The $m^2$ term ($\sim 400\,\text{MB}$ combined for $K_{\mathcal{X}\mathcal{X}}$ and $L$) is secondary to the $nm$ term ($4\,\text{GB}$ for $K_{\mathcal{X}^*\mathcal{X}}$). If memory is tight, $K_{\mathcal{X}^*\mathcal{X}}$ can be computed in row-blocks and the mat-vecs in Steps 5 and 10 done blockwise — reducing peak memory to $\mathcal{O}(m^2 + b \cdot m)$ where $b$ is the block size.

---

## Compute Analysis

Let $d_{ker}$ be the cost of a single kernel evaluation (great-circle distance + Matérn + OU): $\mathcal{O}(1)$ but with a non-trivial constant (trig functions for great-circle).

### Offline (once per time window)

| Step | Operation | Cost | Notes |
|---|---|---|---|
| Build $K_{\mathcal{X}\mathcal{X}}$ | $m^2$ kernel evaluations | $\mathcal{O}(m^2 d_{ker})$ | Symmetric: $m(m+1)/2$ unique evals |
| Cholesky of $C$ | Dense factorisation | $\mathcal{O}(m^3 / 3)$ | **Dominant offline cost** |
| Solve $\alpha = C^{-1}y$ | Two triangular solves | $\mathcal{O}(m^2)$ | Negligible after Cholesky |
| Build $K_{\mathcal{X}^*\mathcal{X}}$ | $nm$ kernel evaluations | $\mathcal{O}(nm\, d_{ker})$ | Embarrassingly parallel |
| Compute $\mu_{f|y} = K_{\mathcal{X}^*\mathcal{X}}\alpha$ | Dense mat-vec | $\mathcal{O}(nm)$ | BLAS-2, fast |

**Total offline:** $\mathcal{O}(m^3 + nm)$

For $m = 5\times10^3$: $m^3/3 \approx 4 \times 10^{10}$ flops. At $10^{12}$ flops/sec (modern CPU): $\sim 40$ seconds. $nm = 5 \times 10^8$ — sub-second.

### Per Posterior Sample (Steps 6–10)

| Step | Operation | Cost | Notes |
|---|---|---|---|
| Draw $w_j$ and spectral freqs | Sample $r$ Gaussians + $r$ uniforms | $\mathcal{O}(r)$ | Negligible |
| Evaluate $\tilde{f}(\mathcal{X})$ | Form $(m \times r)$ feature matrix, multiply by $w$ | $\mathcal{O}(mr)$ | |
| Evaluate $\tilde{f}(\mathcal{X}^*)$ | Form $(n \times r)$ feature matrix, multiply by $w$ | $\mathcal{O}(nr)$ | **Dominant per-sample cost if $nr > nm$** |
| Compute $\delta = y - \tilde{f}(\mathcal{X})$ | Vector subtract | $\mathcal{O}(m)$ | Negligible |
| Solve $\beta = C^{-1}\delta$ | Two triangular solves | $\mathcal{O}(m^2)$ | Fast — $L$ precomputed |
| Correction $K_{\mathcal{X}^*\mathcal{X}}\beta$ | Dense mat-vec | $\mathcal{O}(nm)$ | BLAS-2, fast |
| Add: $\eta^* = \tilde{f}(\mathcal{X}^*) + K_{\mathcal{X}^*\mathcal{X}}\beta$ | Vector add | $\mathcal{O}(n)$ | Negligible |

**Total per sample:** $\mathcal{O}(r(m + n) + m^2 + nm) = \mathcal{O}(nm + rn)$

For $n = 10^5$, $m = 5 \times 10^3$, $r = 2 \times 10^3$: $nm = 5 \times 10^8$, $nr = 2 \times 10^8$, $m^2 = 2.5 \times 10^7$. The dominant per-sample cost is the mat-vec $K_{\mathcal{X}^*\mathcal{X}}\beta$ at $\mathcal{O}(nm)$.

**For $S = 100$ samples:** $100 \times (nm + rn) \approx 7 \times 10^{10}$ flops — about 70 seconds on CPU, or seconds on GPU (all mat-vecs are highly parallelisable).

**Total end-to-end:**

$$\text{Compute} = \underbrace{\mathcal{O}(m^3)}_{\text{Cholesky}} + \underbrace{\mathcal{O}(nm)}_{\text{cross-cov + mean}} + \underbrace{S \cdot \mathcal{O}(nm + rn)}_{\text{posterior samples}}$$

---

## Complexity Summary

| Phase | Storage | Compute |
|---|---|---|
| Build $K_{\mathcal{X}\mathcal{X}}$, Cholesky | $\mathcal{O}(m^2)$ | $\mathcal{O}(m^3)$ |
| Build $K_{\mathcal{X}^*\mathcal{X}}$, posterior mean | $\mathcal{O}(nm)$ — **bottleneck** | $\mathcal{O}(nm)$ |
| $S$ posterior samples | $\mathcal{O}(Sn)$ | $S \cdot \mathcal{O}(nm + rn)$ |
| **Total** | $\mathcal{O}(m^2 + nm)$ | $\mathcal{O}(m^3 + nm(1 + S) + Srn)$ |

**Scaling regime:**

- When $m$ is small ($m \lesssim 10^3$): Cholesky is fast; $K_{\mathcal{X}^*\mathcal{X}}$ and mat-vecs dominate.
- When $m$ is large ($m \gtrsim 10^4$): Cholesky $\mathcal{O}(m^3)$ becomes intractable — need sparse GP / inducing points to bring $m$ back down.
- When $S$ is large: per-sample mat-vec $\mathcal{O}(nm)$ dominates; batch across samples using $(n \times S)$ output matrix.
- When $n$ is large ($n \gtrsim 10^6$, sub-$0.05°$ grid): $K_{\mathcal{X}^*\mathcal{X}}$ storage becomes intractable — compute in blocks of rows, never materialise the full matrix.
