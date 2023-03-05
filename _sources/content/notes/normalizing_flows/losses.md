# Losses


## KL-Divergence vs Negative Log-Likelihood

Here we want to show that the KL-Divergence between the true distribution $p_\text{data}(\mathbf{x})$ and the estimated distribution
$p_\theta(\mathbf{x})$ is the same as maximizing the likelihood of our estimated distribution $p_\theta(\mathbf{x})$.

$$
\begin{equation}
    \text{D}_\text{KL}\left[p_\text{data}(\mathbf{x}) || p_\theta(\mathbf{x}) \right] = \mathbb{E}_{p_\text{data}(\mathbf{x})} \left[ \log p_\theta(\mathbf{x}) \right] + \text{constant}
\end{equation}
$$

````{admonition} Proof
:class: dropdown info

First we decompose the KL-Divergence into its log terms.

$$
\begin{align}\text{D}_\text{KL}\left[p_\text{data}(\mathbf{x}) || p_\theta(\mathbf{x}) \right] &=\mathbb{E}_{p_\text{data}(\mathbf{x})} \left[ \log \frac{p_\text{data}(\mathbf{x})}{p_\theta(\mathbf{x})} \right] \\
&=\mathbb{E}_{p_\text{data}(\mathbf{x})}\left[ \log p_\text{data}(\mathbf{x}) - \log p_\theta(\mathbf{x}) \right] \\
&=\mathbb{E}_{p_\text{data}(\mathbf{x})}\left[ \log p_\text{data}(\mathbf{x})\right] - \mathbb{E}_{ p_\text{data}(\mathbf{x})} \left[ \log p_\theta(\mathbf{x}) \right]
\end{align}
$$

%
The first term is the entropy of our data, $H\left(p_\text{data}(\mathbf{x}) \right)$. This term doesn't depend on our parameters $\theta$ which means it will be constant irregardless of how well we estimate $p_\theta(\mathbf{x})$. So we can simplify this function.

$$
\begin{align}
\text{D}_\text{KL}\left[p_\text{data}(\mathbf{x}) || p_\theta(\mathbf{x}) \right] &=- \mathbb{E}_{p_\text{data}(\mathbf{x})} \left[ \log p_\theta(\mathbf{x}) \right] + \text{constant} \\&=- \int p_\text{data}(\mathbf{x}) \log p_\theta(\mathbf{x}) d\mathbf{x} + C
\end{align}
$$

The remaining term is the cross-entropy; the expected amount of bits need to compress. This is optimal when $p_\text{data}(\mathbf{x}) = p_\theta(\mathbf{x})$ (cite: Shannon Source Coding Theorem). Let $p_\text{data}(\mathbf{x})$ be an empirical distribution described by a delta.

$$
\begin{equation}
    p_\text{data}(\mathbf{x}) = \frac{1}{N} \sum_{n=1}^N \delta (\mathbf{x} - \mathbf{x}_n)
\end{equation}
$$

We assume that it puts a probability on the observed data and zero everywhere else. Plugging this into our KL-Divergence function, we get:


$$
\begin{equation}
\text{D}_\text{KL}\left[p_\text{data}(\mathbf{x}) || p_\theta(\mathbf{x}) \right] = -  \int \left[ \frac{1}{N} \sum_{n=1}^N \delta (\mathbf{x} - \mathbf{x}_n) \right] \log p_\theta(\mathbf{x}) d\mathbf{x} + C
\end{equation}
$$

Then using the law of large numbers where given enough samples we can empirically estimate this integral, we can simplify this even further:

$$
\begin{equation}
\text{D}_\text{KL}\left[p_\text{data}(\mathbf{x}) || p_\theta(\mathbf{x}) \right] = -  \frac{1}{N} \sum_{n=1}^N \log p_\theta(\mathbf{x}) + C
\end{equation}
$$

We are left with the log-likelihood term. So maximizing the likelihood of our estimated distribution $p_\theta(\mathbf{x})$ is equivalent to minimizing the difference between the estimated distribution $p_\theta(\mathbf{x})$ and the real distribution $p_\text{data}(\mathbf{x})$. This is a proxy method allowing us to find the parameters $\theta$ without explicitly knowing the real distribution.

````

## Constructive-Destructive KL-Divergence



Let $\boldsymbol{f}_\theta$ be the invertible, bijective normalizing function which maps $\mathbf{x}$ to $\mathbf{z}$, i.e. $\boldsymbol{f}_\theta:\mathbf{x} \in \mathbb{R}^D \rightarrow \mathbf{z} \in \mathbb{R}^D$. Let $g_\theta$ be the inverse of $\boldsymbol{f}_\theta$ which is the generating function mapping $\mathbf{z}$ to $\mathbf{x}$, i.e. $\boldsymbol{g}_\theta := \boldsymbol{f}_\theta^{-1} :\mathbf{z} \in \mathbb{R}^D \rightarrow \mathbf{x} \in \mathbb{R}^D$. We can view $\boldsymbol{f}_\theta$ as a destructive density whereby we "destroy" the density of the original dataset $p_\text{data}(\mathbf{x})$ into a common base density $p_\mathbf{z}$. Conversely, we can view $g_\theta$ as a constructive density whereby we "construct" the density of the original dataset $p_\text{data}(\mathbf{x})$ from a base density $p_\mathbf{z}$.

$$
\begin{equation}
\mathbf{z} = \boldsymbol{f}_\theta(\mathbf{x}), \qquad \mathbf{x} = g_\theta(\mathbf{z})
\end{equation}
$$

We're assuming $\mathbf{z}\sim p_\mathbf{z}(\mathbf{z})$. Using the change of variables formula, we can express the probability of $p_\theta(\mathbf{x})$ in terms of $\mathbf{z}$ and the transform $\boldsymbol{f}_\theta$.

$$
\begin{equation}
p_\theta(\mathbf{x}) = p_\mathbf{z}(\boldsymbol{f}_\theta(\mathbf{x})) \left| \nabla_\mathbf{x} \boldsymbol{f}_\theta(\mathbf{x})\right|
\end{equation}
$$

This function $\boldsymbol{f}_\theta$ "normalizes" the complex density $\mathbf{x}$ into a simpler base distribution $\mathbf{z}$. We can also express this equation in terms of $g_\theta$ which is the standard found in the normalizing flow literature.

$$
\begin{equation}
p_\theta(\mathbf{x}) = p_\mathbf{z}(\mathbf{z}) \left| \nabla_\mathbf{z} g_\theta(\mathbf{z})\right|^{-1}
\end{equation}
$$

The function $g_\theta$ pushes forward the base density $\mathbf{z}$ to a more complex density $\mathbf{x}$.
%
In this demonstration, we want to show that the following is equivalent.

$$
\begin{equation}
\text{D}_\text{KL}\left[p_\text{data}(\mathbf{x}) || p_\mathbf{x}(\mathbf{x}; \theta) \right] = \text{D}_\text{KL} \left[ p_\text{target}(\mathbf{z}; \theta) || p_\mathbf{z}(\mathbf{z}) \right]
\end{equation}
$$

This says that the KL-Divergence between the data distribution $p_\text{data}(\mathbf{x})$ and the model $p_\mathbf{x}(\mathbf{x};\theta)$ is equivalent to the KL-Divergence between \textit{induced} distribution $p_\text{target}(\mathbf{z};\theta)$ from the transformation $\boldsymbol{f}_\theta(\mathbf{x})$ and the chosen base distribution $p_\mathbf{z}(\mathbf{z})$.

````{admonition} Proof
:class: dropdown info

First we deconstruct the KL-Divergence term into its log components.

$$
\begin{equation}
\text{D}_\text{KL} \left[ p_\text{data}(\mathbf{x}) || p_\mathbf{x}(\mathbf{x};\theta) \right]= \mathbb{E}_{p_\text{data}(\mathbf{x})} \left[ \log p_\text{data}(\mathbf{x}) - \log p_\mathbf{x}(\mathbf{x};\theta) \right]
\end{equation}
$$

If we expand $p_\mathbf{x}(\mathbf{x};\theta)$ with the change of variables formula.

$$
\begin{equation}
\text{D}_\text{KL} \left[ p_\text{data}(\mathbf{x}) || p_\theta(\mathbf{x}) \right]= \mathbb{E}_{p_\text{data}(\mathbf{x})} \left[ \log p_\text{data}(\mathbf{x}) - \log p_\mathbf{z}(\boldsymbol{f}_\theta(\mathbf{x})) - \log |\nabla_\mathbf{x} \boldsymbol{f}_\theta(\mathbf{x})| \right]
\end{equation}
$$

Now we do a change of variables from the data distribution $\mathbf{x}$ to the base distribution $\mathbf{z}$.

$$
\begin{equation}\text{D}_\text{KL} \left[ p_\text{data}(\mathbf{z}) || p_\mathbf{x}(\mathbf{x};\theta) \right]= \mathbb{E}_{p_\text{target}(\mathbf{z})} \left[ \log p_\text{data}(g_\theta(\mathbf{z})) - \log p_\mathbf{z}(\mathbf{z}) + \log |\nabla_\mathbf{z} g_\theta(\mathbf{z})| \right]
\end{equation}
$$

Recognize that we have changed the expectations from the data to the induced distribution and all terms are wrt to $\mathbf{z}$. So we can reduce this to:

$$
\begin{equation}
\text{D}_\text{KL} \left[ p_\text{data}(\mathbf{x}) || p_\mathbf{x}(\mathbf{x};\theta) \right]=
\mathbb{E}_{p_{\text{target}}(\mathbf{z})} \left[ \log p_{\text{target}}(\mathbf{z}) - \log p_\mathbf{z}(\mathbf{z}) \right]
\end{equation}
$$

where $p_{\text{target}}(\mathbf{x})$ is the distribution of $\mathbf{z}=\boldsymbol{f}_\theta(\mathbf{x})$ when $\mathbf{x}$ is sampled from $p_\text{data}(\mathbf{x})$. So this is simply the KL-Divergence between the transformed data in the latent space and the base distribution we choose:

$$
\begin{equation}
\text{D}_\text{KL} \left[ p_\text{data}(\mathbf{x}) || p_\theta(\mathbf{x}) \right]=\text{D}_\text{KL} \left[ p_{\boldsymbol{f}_\theta}(\mathbf{z}) || p_\mathbf{z}(\mathbf{z}) \right]
\end{equation}
$$

which completes the proof.

````
