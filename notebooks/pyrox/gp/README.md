---
title: Gaussian Processes — regression & classification
---

# Gaussian Processes — regression & classification

Gaussian processes (GPs) give us a nonparametric prior over functions and an analytically tractable posterior in the regression case — a small, well-conditioned linear-algebra problem at the heart of "Bayesian deep learning for the reasonable person." This section walks through the two canonical GP inference modes — **conjugate regression** and **non-conjugate latent-variable classification** — using pyrox's `gp` module. For the textbook treatment see {cite}`rasmussenWilliams2006`; for a broader probabilistic-modeling context see {cite}`murphy2012`.

## Model

Let $x \in \mathbb{R}^{D}$ be an input and $f : \mathbb{R}^{D} \to \mathbb{R}$ the latent function of interest. A Gaussian process prior

$$
f \sim \mathcal{GP}\bigl(m(\cdot),\, k_\theta(\cdot, \cdot)\bigr)
$$

is fully specified by a mean function $m : \mathbb{R}^D \to \mathbb{R}$ (typically zero) and a positive-definite kernel $k_\theta : \mathbb{R}^D \times \mathbb{R}^D \to \mathbb{R}$ parameterized by hyperparameters $\theta$ (length-scales, amplitude, kernel-specific extras). Evaluated at any finite collection $X = \{x_i\}_{i=1}^N$, the vector $\mathbf{f} = f(X)$ is jointly Gaussian with $\mathbf{f} \sim \mathcal{N}(\mathbf{m}_X,\, K_{XX})$ where $[K_{XX}]_{ij} = k_\theta(x_i, x_j)$.

### Conjugate regression

For Gaussian noise $y_i = f(x_i) + \varepsilon_i$, $\varepsilon_i \overset{\text{iid}}{\sim} \mathcal{N}(0, \sigma_n^2)$, the posterior predictive at new inputs $X_\star$ is closed-form:

$$
\begin{aligned}
\mu_\star &= \mathbf{m}_\star + K_{\star X}\,(K_{XX} + \sigma_n^2 I)^{-1}(\mathbf{y} - \mathbf{m}_X),\\
\Sigma_\star &= K_{\star\star} - K_{\star X}\,(K_{XX} + \sigma_n^2 I)^{-1}\,K_{X\star}.
\end{aligned}
$$

Hyperparameters are learned by maximizing the log marginal likelihood

$$
\log p(\mathbf{y} \mid \theta, \sigma_n) = -\tfrac{1}{2}(\mathbf{y} - \mathbf{m}_X)^\top (K_{XX} + \sigma_n^2 I)^{-1}(\mathbf{y} - \mathbf{m}_X) - \tfrac{1}{2}\log\lvert K_{XX} + \sigma_n^2 I\rvert - \tfrac{N}{2}\log 2\pi.
$$

### Non-conjugate classification

For binary outputs $y_i \in \{0, 1\}$ with Bernoulli likelihood $y_i \mid f(x_i) \sim \text{Bernoulli}\bigl(\sigma(f(x_i))\bigr)$ (where $\sigma$ is the logistic link), the posterior $p(\mathbf{f} \mid \mathbf{y})$ is no longer Gaussian. Classical options are a Laplace approximation {cite}`mackay1992laplace` — a Gaussian approximation centered at the MAP with precision equal to the Hessian of the negative log posterior — or stochastic variational inference {cite}`hensman2013`. In pyrox the latent-GP model is the same equinox module as in the regression case; only the likelihood changes.

## Numerical considerations

- **Cholesky factorization** is the workhorse. Factor $K_{XX} + \sigma_n^2 I = L L^\top$ once, then solve with $L$ via two triangular systems; this is $O(N^3)$ time / $O(N^2)$ memory and dominates both training and prediction.
- **Jitter / nugget.** $K_{XX}$ is positive definite in exact arithmetic but loses rank near the data limit. Adding $\epsilon I$ with $\epsilon \in [10^{-6}, 10^{-4}]$ is standard. Pyrox threads this through its kernel call.
- **Parameterization of positives.** Length-scales, amplitudes, and noise variances are constrained to $\mathbb{R}_{>0}$. Train in the unconstrained space — typically $\log$ or $\operatorname{softplus}^{-1}$ — so gradients stay finite near zero. The masterclass notebooks show three ways pyrox wires this up.
- **Inducing points.** For $N \gtrsim 10^4$, exact GPs are impractical. Sparse variational GPs {cite}`titsias2009,hensman2013` pick $M \ll N$ inducing locations and give a $O(N M^2)$ algorithm; this isn't covered in the tutorials below but is worth knowing when they feel slow.
- **Prediction variance conditioning.** Computing $\Sigma_\star$ via $K_{\star X} (K + \sigma_n^2 I)^{-1} K_{X\star}$ is numerically safer when done as $K_{\star\star} - V^\top V$ with $V = L^{-1} K_{X\star}$, rather than inverting the kernel explicitly.

## Notebooks

- [`exact_gp_regression`](exact_gp_regression.ipynb) — conjugate regression end-to-end: prior / marginal likelihood / posterior predictive on a 1D synthetic dataset, three patterns for wiring hyperparameters (see the masterclass sub-section).
- [`latent_gp_classification`](latent_gp_classification.ipynb) — latent-GP + Bernoulli likelihood via NumPyro's `NUTS` sampler, with the same three patterns for parameter handling.

## References

```{bibliography}
:filter: docname in docnames
```
