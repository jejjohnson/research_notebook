# Conditional Normalizing Flows

## Model

$$
\mathbf{y} = \boldsymbol{f}(\mathbf{x};\boldsymbol{\theta})
$$

$$
\tilde{\mathbf{x}} = \text{NN}_{\boldsymbol \theta}(\mathbf{x})
$$

---
## Transform

$$
\mathbf{z} = \boldsymbol{T}_{\boldsymbol{\theta}}(\mathbf{y};\mathbf{x})
$$

We can also add a type of encoder function for the inputs, $\mathbf{x}$.

$$
\tilde{\mathbf{x}} = \text{NN}_{\boldsymbol \theta}(\mathbf{x}) 
$$

So this alters the formulation slightly:

$$
\mathbf{z} = \boldsymbol{T}_{\boldsymbol{\theta}}(\mathbf{y}; \tilde{\mathbf{x}})
$$

or more compactly:

$$
\mathbf{z} = \boldsymbol{T}_{\boldsymbol{\theta}}(\mathbf{y}; \text{NN}_{\boldsymbol \theta}(\mathbf{x}))
$$




---
## Prior

$$
p(\mathbf{z}|\mathbf{x}) = \mathcal{N}\left(\mathbf{z}; \boldsymbol{\mu}(\mathbf{x}), \boldsymbol{\sigma}^2(\mathbf{x}) \right)
$$


---
## Split Prior


$$
p(\mathbf{z}_{\ell+1}|\mathbf{z}_\ell, \mathbf{x}) = \mathcal{N}(\mathbf{z}_{\ell+1}; \boldsymbol{\mu}_{\boldsymbol\theta}(\mathbf{z}_\ell, \mathbf{x}), \boldsymbol{\sigma}^2_{\boldsymbol\theta}(\mathbf{z}_\ell,\mathbf{x}))
$$

---
## Coupling

This is arguably the most powerful method to incorporate prior knowledge.

$$
\mathbf{z}_\ell = \boldsymbol{T}(\mathbf{x}; \text{NN}_{\boldsymbol \theta}(\mathbf{z}_\ell,\mathbf{x}))
$$


---
## Sources

* Learning Likelihoods with Conditional Normalizing Flows - Winkler et al (2019)