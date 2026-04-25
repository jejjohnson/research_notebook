---
title: SurVAE flows, Gaussianization, and likelihood accounting
---

# SurVAE flows, Gaussianization, and likelihood accounting

This note gives a deliberately slow proof of the likelihood rules behind
SurVAE flows from the perspective of Gaussianization. The goal is to explain
why ordinary normalizing flows, surjective transformations, and stochastic
VAE-like transformations can all be treated as composable layers with local
likelihood contributions.

The main references are SurVAE Flows {cite:p}`nielsen2020survae`,
Gaussianization Flows {cite:p}`meng2020gaussianization`, iterative
Gaussianization {cite:p}`laparra2011iterative`, and the standard normalizing
flow / VAE literature {cite:p}`rezende2015variational,dinh2017density,kingma2014vae`.

```{admonition} Big picture
:class: tip

A normalizing flow Gaussianizes data by an invertible transport map. A SurVAE
flow generalizes this idea: it Gaussianizes using invertible maps, information
losing maps, and stochastic maps, while keeping track of the likelihood or a
variational lower bound layer by layer.
```

## 1. Gaussianization as density estimation

Let

$$
x \in \mathcal X,
\qquad
z \in \mathcal Z,
\qquad
p_Z(z)=\mathcal N(z;0,I).
$$

In Gaussianization, we learn a map

$$
T:\mathcal X\to\mathcal Z,
\qquad
z=T(x),
$$

so that the transformed data look approximately standard Gaussian:

$$
z=T(x)\sim \mathcal N(0,I).
$$

If $T$ is bijective and differentiable, the likelihood follows from the
ordinary change-of-variables formula:

$$
p_X(x)=p_Z(T(x))\left|\det J_T(x)\right|.
$$

Equivalently,

$$
\log p_X(x)=\log p_Z(T(x)) + \log\left|\det J_T(x)\right|.
$$

```{admonition} Gaussianization convention
:class: note

There are two common directions. The **analysis** or **Gaussianization**
direction maps data to latent variables, $z=T(x)$. The **generative** direction
maps latent variables to data, $x=T^{-1}(z)$. The log determinant changes sign
depending on which direction is used.
```

For a composition of bijective Gaussianization layers,

$$
x=x_0 \mapsto x_1 \mapsto \cdots \mapsto x_K=z,
$$

we get

$$
\log p_X(x)
=
\log p_Z(x_K)
+
\sum_{k=1}^K
\log\left|\det J_{T_k}(x_{k-1})\right|.
$$

This is the classical normalizing-flow story.

SurVAE flows ask: what if some useful transformations are not bijections?

Examples include sorting, absolute value, max pooling, slicing, augmentation,
dequantization, periodic wrapping, and VAE-style stochastic maps. These are
natural in image modeling, representation learning, and geoscience, where many
forward operators lose information.

## 2. The universal latent-variable identity

Start from the marginal likelihood identity

$$
p_X(x)=\int p_{X,Z}(x,z)\,dz.
$$

Factor the joint distribution generatively:

$$
p_{X,Z}(x,z)=p_Z(z)p_{X\mid Z}(x\mid z).
$$

Then

$$
p_X(x)=\int p_Z(z)p_{X\mid Z}(x\mid z)\,dz.
$$

Now introduce any auxiliary inverse or inference density

$$
q_{Z\mid X}(z\mid x),
$$

assuming it is positive wherever the integrand is positive. Then

$$
p_X(x)
=
\int q_{Z\mid X}(z\mid x)
\frac{p_Z(z)p_{X\mid Z}(x\mid z)}{q_{Z\mid X}(z\mid x)}\,dz.
$$

Taking logs gives

$$
\log p_X(x)
=
\log
\mathbb E_{q(z\mid x)}
\left[
\frac{p_Z(z)p_{X\mid Z}(x\mid z)}{q_{Z\mid X}(z\mid x)}
\right].
$$

By Jensen's inequality,

$$
\log p_X(x)
\ge
\mathbb E_{q(z\mid x)}
\left[
\log p_Z(z)
+
\log p_{X\mid Z}(x\mid z)
-
\log q_{Z\mid X}(z\mid x)
\right].
$$

```{admonition} The master equation
:class: important

SurVAE flows are easiest to understand from this identity. Every layer asks:
what is the forward generative density $p(x\mid z)$, what is the inverse density
$q(z\mid x)$, and which terms are exact versus variational?
```

A VAE uses this lower bound directly. A bijective normalizing flow is a special
case where the bound is exact because the inverse is deterministic and unique.
SurVAE flows organize many transformation types under this same accounting
system.

## 3. Bijective transformations

Assume

$$
z=T(x),
\qquad
x=T^{-1}(z),
$$

where $T:\mathbb R^D\to\mathbb R^D$ is a differentiable bijection.

### 3.1 Volume-element proof

For a small region $A\subset \mathcal X$,

$$
\mathbb P(x\in A)=\mathbb P(z\in T(A)).
$$

Locally,

$$
dz=\left|\det J_T(x)\right|dx.
$$

Therefore,

$$
p_X(x)dx=p_Z(z)dz.
$$

Substitute $z=T(x)$:

$$
p_X(x)dx=p_Z(T(x))\left|\det J_T(x)\right|dx.
$$

Cancel $dx$:

$$
p_X(x)=p_Z(T(x))\left|\det J_T(x)\right|.
$$

Thus,

$$
\log p_X(x)=\log p_Z(T(x))+\log\left|\det J_T(x)\right|.
$$

```{admonition} What the determinant means
:class: tip

The Jacobian determinant is a local volume correction. If $T$ expands a small
volume around $x$, then the latent density must be pulled back with a larger
factor. If $T$ contracts volume, the correction is smaller.
```

### 3.2 Dirac-delta proof

Now write the generative direction as

$$
z\sim p_Z(z),
\qquad
x=f(z),
$$

where $f=T^{-1}$. Since $x$ is deterministic given $z$, the conditional density
is a Dirac delta:

$$
p_{X\mid Z}(x\mid z)=\delta(x-f(z)).
$$

Hence

$$
p_X(x)=\int p_Z(z)\delta(x-f(z))\,dz.
$$

Because $f$ is bijective, the equation

$$
x=f(z)
$$

has exactly one solution

$$
z=f^{-1}(x)=T(x).
$$

The multivariate delta identity gives

$$
\delta(x-f(z))
=
\frac{\delta(z-f^{-1}(x))}
{\left|\det J_f(f^{-1}(x))\right|}.
$$

Therefore,

$$
p_X(x)
=
\int p_Z(z)
\frac{\delta(z-f^{-1}(x))}
{\left|\det J_f(f^{-1}(x))\right|}
\,dz.
$$

The denominator is constant with respect to $z$, so

$$
p_X(x)
=
\frac{1}
{\left|\det J_f(f^{-1}(x))\right|}
\int p_Z(z)\delta(z-f^{-1}(x))\,dz.
$$

Using the sifting property of the delta function,

$$
\int p_Z(z)\delta(z-f^{-1}(x))\,dz
=
p_Z(f^{-1}(x)).
$$

Thus

$$
p_X(x)
=
\frac{p_Z(f^{-1}(x))}
{\left|\det J_f(f^{-1}(x))\right|}.
$$

Since $T=f^{-1}$,

$$
\left|\det J_T(x)\right|
=
\frac{1}
{\left|\det J_f(f^{-1}(x))\right|},
$$

so

$$
p_X(x)=p_Z(T(x))\left|\det J_T(x)\right|.
$$

```{admonition} The delta function is not magic
:class: note

The Dirac delta enforces the deterministic constraint $x=f(z)$. The Jacobian
appears because a point constraint in $x$-space corresponds to a differently
scaled point constraint in $z$-space.
```

## 4. Bijections as degenerate VAEs

A bijective flow can be written as a latent-variable model with deterministic
encoder and decoder:

$$
q_{Z\mid X}(z\mid x)=\delta(z-T(x)),
$$

and

$$
p_{X\mid Z}(x\mid z)=\delta(x-T^{-1}(z)).
$$

There is no posterior uncertainty because each $x$ corresponds to exactly one
$z$. Therefore, the variational lower bound is tight. This is why normalizing
flows give exact likelihoods.

## 5. Surjective transformations

A map

$$
f:\mathcal Z\to\mathcal X
$$

is surjective if every $x\in\mathcal X$ has at least one preimage, but possibly
many:

$$
f^{-1}(x)=\{z:f(z)=x\}.
$$

Generatively,

$$
z\sim p_Z(z),
\qquad
x=f(z).
$$

The forward map is deterministic, but the inverse is ambiguous.

Examples:

$$
x=|z|,
$$

where $z=x$ and $z=-x$ both map to the same value;

$$
x=\operatorname{sort}(z),
$$

where all permutations of $z$ map to the same sorted vector; and

$$
x=\operatorname{slice}(z),
$$

where some coordinates are discarded.

```{admonition} Surjection intuition
:class: important

A surjection is an information-losing deterministic map. The likelihood must
account for the missing information: either by summing/integrating over all
preimages exactly or by introducing a stochastic inverse distribution.
```

## 6. Exact likelihood for finite-to-one surjections

Assume $f:\mathbb R^D\to\mathbb R^D$ is many-to-one but locally invertible on
branches. Let the domain decompose into branches

$$
\mathcal Z=\bigcup_k \mathcal Z_k,
$$

and let

$$
f_k:\mathcal Z_k\to\mathcal X
$$

be bijective on each branch. For a given $x$, define

$$
z_k=f_k^{-1}(x).
$$

Start again from the delta representation:

$$
p_X(x)=\int p_Z(z)\delta(x-f(z))\,dz.
$$

Split the integral over branches:

$$
p_X(x)=
\sum_k
\int_{\mathcal Z_k}p_Z(z)\delta(x-f_k(z))\,dz.
$$

On each branch,

$$
\delta(x-f_k(z))
=
\frac{\delta(z-z_k)}{\left|\det J_{f_k}(z_k)\right|}.
$$

Therefore,

$$
p_X(x)=
\sum_k
\frac{p_Z(z_k)}{\left|\det J_{f_k}(z_k)\right|}.
$$

Equivalently,

$$
p_X(x)=
\sum_{z\in f^{-1}(x)}
p_Z(z)
\left|\det J_{f^{-1}_{\text{branch}}}(x)\right|.
$$

This is exact, but the sum may be expensive. Sorting has up to $D!$ branches,
for example.

## 7. Worked example: absolute value

Let

$$
x=|z|,
\qquad z\in\mathbb R,
\qquad x\in[0,\infty).
$$

For $x>0$,

$$
f^{-1}(x)=\{x,-x\}.
$$

The derivative magnitude is $1$ on both branches, so

$$
p_X(x)=p_Z(x)+p_Z(-x).
$$

Now introduce a stochastic inverse:

$$
q(z=x\mid x)=q_+(x),
\qquad
q(z=-x\mid x)=q_-(x),
$$

with

$$
q_+(x)+q_-(x)=1.
$$

Then Jensen's inequality gives

$$
\log p_X(x)
\ge
\mathbb E_{q(z\mid x)}
\left[
\log p_Z(z)-\log q(z\mid x)
\right].
$$

Expanding the expectation,

$$
\mathcal L(x)=
q_+(x)\left[\log p_Z(x)-\log q_+(x)\right]
+
q_-(x)\left[\log p_Z(-x)-\log q_-(x)\right].
$$

The bound is tight when $q(z\mid x)$ equals the true posterior over branches:

$$
p(z=x\mid x)=
\frac{p_Z(x)}{p_Z(x)+p_Z(-x)},
$$

and

$$
p(z=-x\mid x)=
\frac{p_Z(-x)}{p_Z(x)+p_Z(-x)}.
$$

```{admonition} What was lost?
:class: tip

The absolute value map destroys the sign. The stochastic inverse samples a sign.
The term $-\log q(z\mid x)$ accounts for the information needed to reconstruct
which branch was chosen.
```

## 8. Worked example: slicing and augmentation

Let

$$
z=(x,u),
$$

and define a surjection that drops $u$:

$$
f(z)=x.
$$

The exact likelihood is

$$
p_X(x)=\int p_Z(x,u)\,du.
$$

This integral may be intractable. Introduce an inverse distribution

$$
u\sim q(u\mid x).
$$

Then

$$
p_X(x)
=
\int q(u\mid x)\frac{p_Z(x,u)}{q(u\mid x)}\,du.
$$

Thus

$$
\log p_X(x)
\ge
\mathbb E_{q(u\mid x)}
\left[
\log p_Z(x,u)-\log q(u\mid x)
\right].
$$

This is the same algebra as the VAE ELBO, but now interpreted as a SurVAE
surjection.

```{admonition} Slicing versus augmentation
:class: note

Slicing loses variables in the generative direction. Augmentation adds auxiliary
variables in the inference direction. These are dual views of the same idea:
modeling a density on one space using a density on a larger space.
```

## 9. Stochastic transformations

A fully stochastic transformation has both an inference density and a generative
density:

$$
z\sim q_{Z\mid X}(z\mid x),
\qquad
x\sim p_{X\mid Z}(x\mid z).
$$

The marginal likelihood is

$$
p_X(x)=\int p_Z(z)p_{X\mid Z}(x\mid z)\,dz.
$$

Usually this integral is intractable, giving the lower bound

$$
\log p_X(x)
\ge
\mathbb E_{q(z\mid x)}
\left[
\log p_Z(z)
+
\log p_{X\mid Z}(x\mid z)
-
\log q_{Z\mid X}(z\mid x)
\right].
$$

This is the VAE case. SurVAE's contribution is to treat this as one layer type
inside a larger compositional flow.

## 10. Layerwise likelihood bookkeeping

Consider a composition

$$
x=x_0\to x_1\to\cdots\to x_K=z.
$$

At the end, evaluate the base density

$$
\log p_Z(z).
$$

Each layer contributes a correction.

For a bijection,

$$
\Delta_k=
\log\left|\det J_{T_k}(x_{k-1})\right|.
$$

For a stochastic or variational inverse layer,

$$
\Delta_k=
\log p_k(x_{k-1}\mid x_k)-\log q_k(x_k\mid x_{k-1}),
$$

with deterministic delta/Jacobian terms handled analytically when present.

So the total exact likelihood or lower bound has the form

$$
\log p_X(x)
\gtrsim
\log p_Z(z)+\sum_{k=1}^K\Delta_k.
$$

The symbol $\gtrsim$ means exact equality for fully exact transformations and a
lower bound when stochastic inverses or variational approximations are used.

```{admonition} Practical implementation rule
:class: important

A SurVAE layer needs two things: a forward sample/evaluate rule and a local log
contribution. Bijections contribute log determinants. Surjections contribute
branch, inverse, or entropy corrections. Stochastic layers contribute
$\log p-\log q$ terms.
```

## 11. Connection back to Gaussianization

Classical Gaussianization says

$$
x\mapsto z\sim \mathcal N(0,I)
$$

using invertible transformations. SurVAE-style Gaussianization says the map may
include operations that are useful but not invertible.

Examples:

- periodic wrapping canonicalizes angles but loses winding number;
- sorting canonicalizes permutation symmetry but loses the original order;
- pooling summarizes local patches but loses sub-patch detail;
- slicing projects from a higher-dimensional latent representation to observed
  coordinates;
- dequantization maps discrete observations into continuous latent variables.

For geoscience, this is natural. Many observation operators are not bijections:

$$
\text{high-resolution field}\mapsto \text{coarse-resolution field},
$$

$$
\text{3D atmospheric state}\mapsto \text{2D column observation},
$$

$$
\text{radiance spectrum}\mapsto \text{retrieved methane column},
$$

$$
\text{continuous field}\mapsto \text{quantized satellite product}.
$$

These transformations lose information. SurVAE flows provide a density-estimation
language for this situation: keep exact likelihoods when possible, introduce
stochastic inverses when necessary, and track the resulting lower bound.

## 12. Summary

| Transformation | Forward behavior | Inverse behavior | Likelihood accounting |
| --- | --- | --- | --- |
| Bijection | one-to-one | deterministic | exact change of variables |
| Surjection | many-to-one | branch sum or stochastic inverse | exact if summed; ELBO if sampled |
| Stochastic | random | stochastic | variational lower bound |

The shortest useful mental model is

$$
\boxed{\text{normalizing flows} = \text{Gaussianization by invertible transport}}
$$

and

$$
\boxed{\text{SurVAE flows} = \text{Gaussianization by transport plus controlled information loss/addition}.}
$$

The Dirac delta proof is the bridge: it shows how deterministic transformations
can be written as conditional densities, and how their likelihood corrections
come from enforcing constraints and correcting volume.

## References

```{bibliography}
:filter: docname in docnames
```
