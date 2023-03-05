# Conditional Variational Inference

---
## Motivations

![figure](https://cvws.icloud-content.com/B/AXJ9rUeugE62r5wOOJGZplFkqQ2AAciBENkJg7AR6H79V70Lrc3KTPH6/Motivation.jpg?o=ApLY0goJWL5NkZyyYTtwj8USHSjRYJLQZJqTVK3czpdn&v=1&x=3&a=CAogRHJLzZNCKG3vI1B4x3b7JcTBaU0HlUFRE9KjoKC6YkASbRDvk6TV2zAY7_D_1tswIgEAUgRkqQ2AWgTKTPH6aiaYk08JNOG61Pu_HMRe7MCQrI-mGrILCwleX8wCib2pfpjdDxfg_nImR8i0It6Sw5Co0EnCW0BSrDmW6t2xhv2n-8JkGJN6lQspVQ7h7eQ&e=1673877518&fl=&r=6b99f1f8-20fb-4a9f-a718-8de67620876f-1&k=KzYtz42KdChgNAei2IZb7g&ckc=com.apple.clouddocs&ckz=com.apple.CloudDocs&p=70&s=B5SVOB5laiQ7qAU_RI8bJZLaxCc&cd=i)

Variational inference is the most scalable inference method the machine learning community has (as of 2019).

Ultimately, we are interested in approximating the likelihood distribution of our observations, $\mathcal{Y}$, which we assume come from some $\mathcal{X}$.

$$
y \sim p(y|x)
$$

```{note}
With a tip!
```

We write some sort of approximation of the *true* (or best) underlying distribution via some parameterized form like so

$$
p_*(\mathbf{y}|\mathbf{x}) \approx p_{\boldsymbol \theta}(\mathbf{y}|\mathbf{x}).
$$

We can do the discriminative version which just means fitting a function to approximate the likelihood

$$
\mathbf{y} = \boldsymbol{f_\theta}(\mathbf{x}) + \epsilon
$$

However, we're often not happy with this approximation and would prefer a more probabilistic interpretation. Another method would be to assume some latent variable, $z$, also plays a pivotal role in order to obtain a probabilistic interpretation.

$$
\mathbf{y} = \boldsymbol{f_\theta}(\mathbf{x},\mathbf{z}) + \epsilon
$$

So how do we express this in a more bayesian way? We can use the same methodology for VIs but we will always be conditioned on, $\mathbf{x}$. So again, we need to assume some latent variable, $\mathbf{z}$, plays a role in estimating the underlying conditional density estimation of $\mathbf{y}$. In the simplest form, we assume a generative model for the joint distribution can be written as

$$
p_\theta(y,z|x) = p(y|z,x)p(z|x)
$$

When fitting a model, we are interested in maximizing the marginal likelihood

$$
p_\theta(y|x) = \int p_\theta(y|z,x)p_\theta(z|x)dz
$$

However, this quantity is intractable because we have a non-linear function thats within an integral.  So we use an variational distribution, $q_\phi(z|x,y)$.

$$
\log p_\theta(y|x) = \mathbb{E}_{q_\phi(z|x,y)}\left[ \log p_\theta(y|x) \right]
$$

Like the VI section, we have many types of conditional variational distributions we can impose. For example, we have some of the following:

* *Conditional Gaussian*, $q(z|x)$
* *Conditional Mixture Distribution*, $\sum_{k}^{K}\pi_k \mathbb{P}(z|x)$
* *Conditional Bijective Transform* (Flow), $q(z|\tilde{z},x)$
* *Conditional Stochastic Transform*, $q(z|y,x)$

We won't go through each of these like in the previous section because it is *relatively* straightforward to extend each of the previous section to include a conditional rv.


---
## ELBO (Encoder) - Derivation

> This derivation comes from the book by [paper]() and the blog post from [pyro-ppl/CVAE](). I find it to be a much better and intuitive derivation.

**Note**: I put the *encoder* tag in the title. This is because there are other ELBOs that have different purposes, for example, variational distributions without an encoder and also an encoder for conditional likelihoods. In this first one, we will like at the ELBO derivation

**Note**: This derivation is more or less *the exact same* as the standard VI however it will include the conditional term, $q(\cdot|\cdot,x),p(\cdot|\cdot,x)$.


As mentioned above, we are interested in expanding the expectation of the marginal likelihood wrt the encoder variational distribution

$$
\log p_\theta(y|x) = \mathbb{E}_{q_\phi(z|x,y)}\left[ \log p_\theta(y|x) \right]
$$

We will do a bit of mathematical manipulation to expand this expectation. Firstly, we will start with Bayes rule:

$$
p_\theta(y|x) = \frac{p_\theta(z,y|x)}{p_\theta(z|y,x)}
$$

Plugging this into our expectation gives us:

$$
\log p_\theta(y|x) = \mathbb{E}_{q_\phi(z|x)}\left[ \log \frac{p_\theta(z,y|x)}{p_\theta(z|y,x)} \right]
$$

Now we will do the identity trick (multiply by 1/1 :) ) within the log term to incorporate the variational distribution, $q_\phi$.

$$
\log p_\theta(y|x) = \mathbb{E}_{q_\phi(z|y,x)}\left[ \log \frac{p_\theta(z,y|x)q_\phi(z|y,x)}{p_\theta(z|y,x)q_\phi(z|y,x)} \right] = \mathbb{E}_{q_\phi(z|y,x)}\left[ \log \frac{p_\theta(z,y|x)q_\phi(z|y,x)}{q_\phi(z|y,x)p_\theta(z|y,x)} \right]
$$

Using the log rules, we can split this fraction into two fractions;

$$
\log p_\theta(y|x) =  \mathbb{E}_{q_\phi(z|x)}\left[ \log \frac{p_\theta(z,y|x)}{q_\phi(z|y,x)} + \log \frac{q_\phi(z|y,x)}{p_\theta(z|y,x)} \right]
$$

Now, we can expand the expectation term across the additive operator

$$
\log p_\theta(y|x) =  \mathbb{E}_{q_\phi(z|y,x)}\left[ \log \frac{p_\theta(z,y|x)}{q_\phi(z|y,x)} \right] + \mathbb{E}_{q_\phi(z|y,x)}\left[\log \frac{q_\phi(z|y,x)}{p_\theta(z|y,x)} \right]
$$

Here, we notice that the second term is actually the Kullback-Leibler divergence term.

$$
\text{D}_{\text{KL}} [Q||P] = \mathbb{E}_Q\left[\log \frac{Q}{P} \right] = - \mathbb{E}_Q\left[\log \frac{P}{Q} \right]
$$

so we can replace this with the more compact form.

$$
\log p_\theta(y|x) =  \mathbb{E}_{q_\phi(z|y,x)}\left[ \log \frac{p_\theta(z,y|x)}{q_\phi(z|y,x)} \right] + \text{D}_{\text{KL}} \left[q_\phi(z|y,x)||p_\theta(z|y,x) \right]
$$

We know from theory that the KL divergence term is always zero or positive. So this means that we can draw a bound on the first term in terms of the marginal log-likelihood.

$$
 \mathcal{L}_{\text{ELBO}}:=\mathbb{E}_{q_\phi(z|y,x)}\left[ \log \frac{p_\theta(z,y|x)}{q_\phi(z|y,x)} \right] \leq \log p_\theta(y|x)
$$

This term is called the Evidence Lower Bound (ELBO). So the objective is to *maximize* this term which will also minimize the KLD.

$$
 \mathcal{L}_{\text{ELBO}}=\mathbb{E}_{q_\phi(z|y,x)}\left[ \log \frac{p_\theta(z,y|x)}{q_\phi(z|y,x)} \right]
$$

So now, we can expand the joint distribution using Bayes rule, i.e. $p(y,z|x)=p(y|z,x)p(z|x)$, to give us.

$$
 \mathcal{L}_{\text{ELBO}}=\mathbb{E}_{q_\phi(z|y,x)}\left[ \log \frac{p(y|z,x)p(z|x)}{q_\phi(z|y,x)} \right]
$$

We can also expand this fraction using the log rules,

$$
 \mathcal{L}_{\text{ELBO}}=\mathbb{E}_{q_\phi(z|y,x)}\left[ \log p(y|z,x) + \log p(z|x) - \log q_\phi(z|y,x) \right].
$$

where:

* $q_\phi(z|y,x)$ - (conditional) prior network
* $p_\theta(y|z,x)$ - generation network
* $p_\theta(z|x)$ - prior network
* $p_\theta(y|x)$ - (pretrained) baseline network

Now, we have some options on how we can group the likelihood, the prior and the variational distribution together and each of them will offer a slightly different interpretation and application.

---
## Reconstruction Loss

If we group the prior probability and the variational distribution together, we get:

$$
 \mathcal{L}_{\text{ELBO}}=\mathbb{E}_{q_\phi(z|y,x)}\left[ \log p(y|z,x) \right] + \mathbb{E}_{q_\phi(z|y,x)}\left[ \log \frac{p(z|x)}{q_\phi(z|y,x)} \right].
$$

This is the same KLD term as before but in the reverse order. So with a slight of hand in terms of the signs, we can rearrange the term to be

$$
 \mathcal{L}_{\text{ELBO}}= \mathbb{E}_{q_\phi(z|y,x)}\left[ \log p(y|z,x) \right] - \mathbb{E}_{q_\phi(z|y,x)}\left[ \log \frac{q_\phi(z|y,x)} {p(z|x)}\right].
$$

---
**Proof**:

$$
\mathbb{E}_q[ \log p - \log q] = - \mathbb{E}_q[\log q - \log p] = - \mathbb{E}_q[\log\frac{q}{p}]
$$

**QED**.

---

So now, we have the exact same KLD term as before. Let's use the simplified form.

$$
 \mathcal{L}_{\text{ELBO}}={\color{blue}\mathbb{E}_{q_\phi(z|y,x)}\left[ \log p_\theta(y|z,x) \right]} - {\color{green}\text{D}_\text{KL}\left[q_\phi(z|y,x)||p_\theta(z|x)\right]}.
$$


where:
* ${\color{blue}\mathbb{E}_{q_\phi(z|y,x)}\left[ \log p_\theta(y|z,x) \right]}$ - is the $\color{blue}\text{reconstruction loss}$.
* ${\color{green}\text{D}_\text{KL}\left[q_\phi(z|y,x)||p_\theta(z|x)\right]}$ - is the complexity, i.e. the $\color{green}\text{KL divergence}$ (a distance metric) between the prior and the variational distribution.

This is easily the most common ELBO term especially with conditional Variational AutoEncoders (cVAEs). The first term is the expectation of the likelihood term wrt the variational distribution. The second term is the KLD between the prior and the variational distribution.


---
## Volume Correction

Another approach is more along the lines of the transform distribution. Assume we have our original data domain $\mathcal{X}$ and we have some stochastic transformation, p(z|y,x), which transforms the data from our original domain to a transform domain, $\mathcal{Z}$.

$$
z \sim p_\theta(z|y,x)
$$

To acquire this, let's look at the equation again

$$
 \mathcal{L}_{\text{ELBO}}=\mathbb{E}_{q_\phi(z|y,x)}\left[ \log p_\theta(y|z,x) + \log p_\theta(z|x) - \log q_\phi(z|y,x) \right].
$$

except this time we will isolate the prior and combine the likelihood and the variational distribution.

$$
 \mathcal{L}_{\text{ELBO}}={\color{blue}\mathbb{E}_{q_\phi(z|y,x)}\left[ \log p_\theta(z|x) \right]} + {\color{green}\mathbb{E}_{q_\phi(z|y,x)}\left[ \log \frac{p_\theta(y|z,x)}{q_\phi(z|y,x)} \right]}.
$$


where:

* ${\color{blue}\mathbb{E}_{q_\phi(z|y,x)}\left[ \log p_\theta(z|x) \right]}$ - is the expectation of the transformed distribution, aka the ${\color{blue}\text{reparameterized probability}}$.
* ${\color{green}\mathbb{E}_{q_\phi(z|y,x)}\left[ \log \frac{p_\theta(y|z,x)}{q_\phi(z|y,x)} \right]}$ - is the ratio between the inverse transform and the forward transform , i.e. ${\color{green}\text{Volume Correction Factor}}$ or *likelihood contribution*.


**Source**: I first saw this approach in the SurVAE Flows paper.

---
## Variational Free Energy (VFE)

There is one more main derivation that remains (that's often seen in the literature). Looking at the equation again

$$
 \mathcal{L}_{\text{ELBO}}=\mathbb{E}_{q_\phi(z|y,x)}\left[ \log p_\theta(y|z,x) + \log p(z|x) - \log q_\phi(z|y,x) \right],
$$

we now isolate the likelihood *and* the prior under the variational expectation. This gives us:

$$
 \mathcal{L}_{\text{ELBO}}={\color{blue}\mathbb{E}_{q_\phi(z|y,x)}\left[ \log _\theta(y|z,x) p_\theta(z|x)\right]} - {\color{green} \mathbb{E}_{q_\phi(z|y,x)}\left[ \log q_\phi(z|y,x) \right]}.
$$

where:

* ${\color{blue}\mathbb{E}_{q_\phi(z|y,x)}\left[ \log p_\theta(y|z,x) p_\theta(z|x)\right]}$ - is the ${\color{blue}\text{energy}}$ function
* ${\color{green} \mathbb{E}_{q_\phi(z|y,x)}\left[ \log q_\phi(z|y,x) \right]}$ - is the ${\color{green}\text{entropy}}$


**Source**: I see this approach a lot in the Gaussian process literature when they are deriving the Sparse Gaussian Process from Titsias.




## Prior Guess

We can also utilize an initial guess from a baseline model, $p_\theta(y|x)$. This can be fed as an input into the encoder, $q(z|x,y)$.


```python=
# =========
# MODEL
# =========
x, y = ..., Optional
# inputs into the baseline net, p(y|x)
y_hat = baseline_net(x)

# inputs to the prior net, q(z|y,x)
z_loc, z_scale = prior_net(x, y_hat)

# sample from distribution
z_samples = sample("z", Normal(z_loc, z_scale))

# output from the generation net, p(y|z)
loc = generation_net(z)

# w/ observations
# training, we can do masking tricks
mask_loc = ...
mask_scale = ...
mask_y = ...

# sample from dist
sample("y", Normal(mask_loc, mask_scale), obs=mask_y)

# w/o observations
deterministic("y", mask_loc.detach())

# =========
# GUIDE
# =========
x, ys = ..., Optional

# with observations
# use baseline net
y_hat = baseline_net(x)
# prior net, q(z|y,x)
loc, scale = prior_net(x, y_hat)
# sample from Normal
sample("z", Normal(loc, scale))

# w/o observations
loc, scale = recognition_net(x, None)
# sample from normal
sample("z", Normal(loc, scale))
```
