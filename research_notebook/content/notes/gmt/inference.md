---
title: Inference
subject: Modern 4DVar
subtitle: How to think about modern 4DVar formulations
short_title: Inference
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - CNRS
      - MEOM
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: data-assimilation, open-science
abbreviations:
    GP: Gaussian Process
---


To solve the inference problem of a single instance, of an observation, we can optimize using the population log-loss

$$
q^*(\cdot|y) =
\underset{q(\cdot|y)}{\text{argmin}} \hspace{2mm}
\mathbb{E}_{z\sim p(z|y)}
\left[ -\log q(z|y)\right]
$$

which is the cross entropy between the true posterior and the optimal soft predictor.
This is a measure of "regret" or excess loss. whereby the optimal soft predictor is when this is equal, i.e. $q^*(z|y)=p(z|y)$.

We can also reformulate this as the minimization between the KL divergence.

$$
q^*(\cdot|y) =
\underset{q(\cdot|y)}{\text{argmin}} \hspace{2mm}
D_{KL}\left[ p(\cdot|y) || q(\cdot|y) \right]
$$

This is an asymmetric measure of distance between two distributions.

:::{prf:theorem}
:class: dropdown

We want to find the optimal soft predictor by solving the problem of minimizing the population log-loss.
However,

$$
q^*(\cdot|\cdot) =
\underset{q(\cdot|\cdot)}{\text{argmin}} \hspace{2mm}
\mathbb{E}_{(z,y)\sim p(z,y)}
\left[ -\log q(z|y)\right]
$$

Looking at the Bayes theorem, we con deconstruct this population loss as a conditional distribution.

$$
p(z,y) = p(z|y)p(y)
$$

We can use the law of iterated expectations to use this as follows



:::
