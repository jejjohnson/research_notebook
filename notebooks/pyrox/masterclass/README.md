---
title: Regression masterclass — three patterns for parameter handling
---

# Regression masterclass — three patterns for parameter handling

A recurring friction when mixing [Equinox](https://github.com/patrick-kidger/equinox) modules {cite}`kidger2021equinox` with [NumPyro](https://github.com/pyro-ppl/numpyro) {cite}`phan2019numpyro` is *where hyperparameters live*. Equinox wants a PyTree of leaves; NumPyro wants named random sites. Pyrox picks three compatible idioms and the same regression problem is solved end-to-end in each, so you can A/B them on ergonomics.

The three patterns live on a spectrum:

| Pattern | Where the params live | What wraps them | Boilerplate |
|---|---|---|---|
| **Pattern 1** — `eqx.tree_at` + raw NumPyro | equinox module leaves | user does the substitution by hand | most |
| **Pattern 2** — `PyroxModule` + `pyrox_sample` | module leaves + a NumPyro plate | pyrox wires sampling | medium |
| **Pattern 3** — `Parameterized` + `PyroxParam` + native `pyrox.gp` | leaves carry prior metadata | pyrox resolves everything | least |

All three are mathematically equivalent — they sample the same posterior. The choice is about how much machinery you want pyrox to do for you.

## Model

All three notebooks target the same regression problem: an exact GP prior $f \sim \mathcal{GP}(0, k_\theta)$ with Gaussian noise $y_i = f(x_i) + \varepsilon_i$, $\varepsilon_i \sim \mathcal{N}(0, \sigma_n^2)$, and hyperparameters $\theta = (\ell, \alpha, \sigma_n)$ (length-scale, signal amplitude, noise scale). We place hyperpriors on each and infer $p(\theta \mid \mathbf{y})$.

The conjugate marginal likelihood (after integrating out $f$) is

$$
\log p(\mathbf{y} \mid \theta) = -\tfrac{1}{2} \mathbf{y}^\top (K_{XX}(\theta) + \sigma_n^2 I)^{-1} \mathbf{y} - \tfrac{1}{2} \log\lvert K_{XX}(\theta) + \sigma_n^2 I\rvert - \tfrac{N}{2}\log 2\pi,
$$

and the joint density evaluated during NUTS is

$$
\log p(\theta, \mathbf{y}) = \log p(\mathbf{y} \mid \theta) + \log p(\theta).
$$

Each pattern differs only in *how* $\theta$ gets into the Equinox kernel module that evaluates $K_{XX}(\theta)$.

## Numerical considerations

- **Constrained hyperparameters.** $\ell, \alpha, \sigma_n$ are all positive. We sample on the unconstrained real line using NumPyro's `TransformedDistribution` (typically $\log$ or `SoftPlus` bijectors) so HMC/NUTS gradients are finite at the origin.
- **Reparameterization.** Heavy-tailed priors on scales (e.g. `LogNormal`) interact poorly with poorly-conditioned likelihoods. When posteriors pile up near zero, a non-centered reparameterization ({cite}`kingma2014vae`-style whitening) dramatically improves the NUTS step-size adaptation.
- **Leapfrog / NUTS cost.** The dominant cost per log-posterior gradient is the Cholesky of $K_{XX}(\theta) + \sigma_n^2 I$ — $O(N^3)$ floating-point ops. Pre-computing the kernel once at evaluation time and re-using its Cholesky is how `pyrox.gp` stays fast.
- **Three patterns, same posterior.** All three code patterns produce the same $\log p(\theta, \mathbf{y})$ up to numerical noise; pyrox's CI pins this via a regression test. So pattern choice is *only* about code ergonomics and not about sampling efficiency.
- **Black-box VI as an alternative.** For bigger $N$ or when NUTS is slow, swap `NUTS` for `SVI` with an `AutoNormal` guide — a mean-field Gaussian approximation fit by maximizing the ELBO {cite}`blei2017vi,ranganath2014bbvi,kucukelbir2017advi`.

## Notebooks

- [`regression_masterclass_treeat`](regression_masterclass_treeat.ipynb) — **Pattern 1**: `eqx.tree_at` substitutes sampled scalars into the equinox kernel by pointer. Most transparent, most boilerplate.
- [`regression_masterclass_pyrox_sample`](regression_masterclass_pyrox_sample.ipynb) — **Pattern 2**: `PyroxModule` + `pyrox_sample` register the hyperparameter sites on behalf of the module. Mid-stack.
- [`regression_masterclass_parameterized`](regression_masterclass_parameterized.ipynb) — **Pattern 3**: `Parameterized` + `PyroxParam` attach prior metadata directly to the Equinox leaves, and `pyrox.gp` resolves sampling + constrained transforms automatically. Fewest user-facing moving parts.

## References

```{bibliography}
:filter: docname in docnames
```
