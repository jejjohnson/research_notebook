# Gaussian Distributions



## Univariate Gaussian

$$\mathcal{P}(x|\mu, \sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}\text{exp}\left( -\frac{1}{2\sigma^2}(x - \mu)^2 \right)$$

## Multivariate Gaussian

$$\begin{aligned}
\mathcal{P}(x | \mu, \Sigma) &= \mathcal{N}(\mu, \Sigma) \\
&= \frac{1}{(2\pi)^{\frac{D}{2}}}\frac{1}{\sqrt{\text{det}|\Sigma|}}\text{exp}\left( -\frac{1}{2}(x-\mu)^{\top}\Sigma^{-1}(x-\mu) \right)
\end{aligned}$$


## Joint Gaussian Distribution


$$\begin{bmatrix}
    \mathbf{x} \\ \mathbf{y}
    \end{bmatrix}
    \sim
    \mathcal{N}
    \left(
    \begin{bmatrix}
    \boldsymbol{\mu}_\mathbf{x} \\ \boldsymbol{\mu}_\mathbf{y}
    \end{bmatrix},
    \begin{bmatrix}
    \boldsymbol{\Sigma}_\mathbf{x} &
    \boldsymbol{\Sigma}_\mathbf{xy}\\ 
    \boldsymbol{\Sigma}_\mathbf{xy}^\top &
    \boldsymbol{\Sigma}_\mathbf{y}
    \end{bmatrix}
    \right)
$$



### Lemma I - Conditional distribution of a Gaussian rv.

Let's define a joint Gaussian distribution for $\mathbf{x,y}$.

$$\begin{bmatrix}
\mathbf{x} \\ \mathbf{y}
\end{bmatrix}
\sim
\mathcal{N}
\left(
\begin{bmatrix}
\boldsymbol{\mu}_\mathbf{x} \\ \boldsymbol{\mu}_\mathbf{y}
\end{bmatrix},
\begin{bmatrix}
\boldsymbol{\Sigma}_\mathbf{x} &
\boldsymbol{\Sigma}_\mathbf{xy}\\ 
\boldsymbol{\Sigma}_\mathbf{xy}^\top &
\boldsymbol{\Sigma}_\mathbf{y}
\end{bmatrix}
\right)$$

We can write each of the marginal and conditional distributions just based on this joint distribution.

---

$$
p(\mathbf{x})	= \mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_\mathbf{x}, \boldsymbol{\Sigma}_\mathbf{x})
$$

$$
p(\mathbf{y})	= \mathcal{N}(\mathbf{y}|\boldsymbol{\mu}_\mathbf{y}, \boldsymbol{\Sigma}_\mathbf{y})
$$

$$
p(\mathbf{x}|\mathbf{y})= \mathcal{N}\left(\mathbf{x}|\boldsymbol{\mu}_\mathbf{x}+\boldsymbol{\Sigma}_\mathbf{xy}\boldsymbol{\Sigma}^{-1}_\mathbf{y}(\mathbf{y}-\boldsymbol{\mu}_\mathbf{y}), \boldsymbol{\Sigma}_\mathbf{x} - \boldsymbol{\Sigma}_\mathbf{xy}\boldsymbol{\Sigma}_\mathbf{y}^{-1}\boldsymbol{\Sigma}_\mathbf{xy}^\top \right)
$$

$$
p(\mathbf{y}|\mathbf{x})	=\mathcal{N}\left(\mathbf{y}|\boldsymbol{\mu}_\mathbf{y}+\boldsymbol{\Sigma}_\mathbf{xy}^\top\boldsymbol{\Sigma}^{-1}_\mathbf{x}(\mathbf{x}-\boldsymbol{\mu}_\mathbf{x}), \boldsymbol{\Sigma}_\mathbf{y} - \boldsymbol{\Sigma}_\mathbf{xy}^\top\boldsymbol{\Sigma}_\mathbf{x}^{-1}\boldsymbol{\Sigma}_\mathbf{xy}\right)
$$




---

### Lemma II - Linear Conditional Gaussian model.

Take a rv $\mathbf{x}$ which is Gaussian distributed

$$p(\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\mathbf{x},\boldsymbol{\Sigma}_\mathbf{x})$$

and take a rv $\mathbf{y}$ which is a linear transformation of $\mathbf{x}$ and is also Gaussian distributed. So we have

$$p(\mathbf{y}|\mathbf{x}) = \mathcal{N}(\mathbf{y}|\mathbf{Ax}+b, \mathbf{R})$$

Since both distributions are Gaussian, we can write the joint distribution of $p(\mathbf{x,y})$ which is also Gaussian.

$$
\begin{bmatrix}
\mathbf{x} \\ \mathbf{y}
\end{bmatrix}
\sim \mathcal{N}
\left(
\begin{bmatrix}
\boldsymbol{\mu}_\mathbf{x} \\ \mathbf{A}\boldsymbol{\mu}_\mathbf{x}+\mathbf{b}
\end{bmatrix},
\begin{bmatrix}
\boldsymbol{\Sigma}_\mathbf{x} & \boldsymbol{\Sigma}_\mathbf{x}\mathbf{A}^\top 
\\ 
\mathbf{A}\boldsymbol{\Sigma}_\mathbf{x} &
\mathbf{A}\boldsymbol{\Sigma}_\mathbf{x}\mathbf{A}^\top + \mathbf{R}
\end{bmatrix}
\right)
$$

This is Gaussian distributed, so we can write down the same equations using the above lemma. Let:

- $\boldsymbol{\Sigma}_\mathbf{x}=\boldsymbol{\Sigma}_\mathbf{x}$
- $\boldsymbol{\Sigma}_\mathbf{xy}=\boldsymbol{\Sigma}_\mathbf{x}\mathbf{A}^\top$
- $\boldsymbol{\Sigma}_\mathbf{y}=\mathbf{A}\boldsymbol{\Sigma}_\mathbf{x}\mathbf{A}^\top + \mathbf{R}$
- $\boldsymbol{\Sigma}_\mathbf{xy}=\mathbf{A}\boldsymbol{\Sigma}_\mathbf{x}$
- $\boldsymbol{\mu}_\mathbf{y}=\mathbf{A}\boldsymbol{\mu}_\mathbf{x}+\mathbf{b}$

---

$$
p(\mathbf{x})=\mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_\mathbf{x}, \boldsymbol{\Sigma}_\mathbf{x})
$$

$$
p(\mathbf{y})	= \mathcal{N}(\mathbf{y}|\mathbf{A}\boldsymbol{\mu}_\mathbf{x}+\mathbf{b}, \mathbf{A}\boldsymbol{\Sigma}_\mathbf{x}\mathbf{A}^\top + \mathbf{R})
$$

$$
p(\mathbf{x}|\mathbf{y}) = \mathcal{N}\left(\mathbf{x}|\boldsymbol{\mu}_\mathbf{x}+\boldsymbol{\Sigma}_\mathbf{x}\mathbf{A}^\top(\mathbf{A}\boldsymbol{\Sigma}_\mathbf{x}\mathbf{A}^\top + \mathbf{R})^{-1}(\mathbf{y}-\boldsymbol{\mu}_\mathbf{y}), \boldsymbol{\Sigma}_\mathbf{x} - \boldsymbol{\Sigma}_\mathbf{x}\mathbf{A}^\top(\mathbf{A}\boldsymbol{\Sigma}_\mathbf{x}\mathbf{A}^\top + \mathbf{R})^{-1}\mathbf{A}\boldsymbol{\Sigma}_\mathbf{x} \right)
$$

$$
p(\mathbf{y}|\mathbf{x}) = \mathcal{N}(\mathbf{y}|\mathbf{Ax}+b, \mathbf{R})
$$

---

#### Marginal

From the lemma we have:

$$p(\mathbf{x}) = \mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_\mathbf{x})v, A)$$

where:

- $a = \boldsymbol{\mu}_\mathbf{x}$
- $\mathbf{A}=\boldsymbol{\Sigma}_\mathbf{x}$

Fortunately, this is a simple plug-in-play with no reductions.

$$p(\mathbf{x})=\mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_\mathbf{x},\boldsymbol{\Sigma}_\mathbf{x})$$

---
## Likelihood

Take a Gaussian distribution with a full covariance matrix:



$$
\mathcal{N}(\mathbf{x};\boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}}\exp \left[ - \frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu}) \right]
$$

---

### Mahalanobis Distance

The Maholanobis Distance is given by:

$$
\Delta^2 = (\mathbf{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu}) 
$$

We can write a simplified version in terms of the Euclidean norm.

$$
(\mathbf{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})  = ||\mathbf{x}-\boldsymbol{\mu}||^2_{\boldsymbol{\Sigma}^{-1}} = ||\boldsymbol{\Sigma}^{-1/2}(\mathbf{x} - \boldsymbol{\mu})||_2^2
$$

**Note**: We often see this as a simplified representation of the covariance metric in the Gaussian likelihood function. This is even more apparent within the mean-squared loss functions as a simplified representation.

---
## Log likelihood

We can also write the log-likliehood of the Gaussian distribution. We simply take the $\log$ of the RHS.

$$
\log \mathcal{N}(\mathbf{x};\boldsymbol{\mu}, \boldsymbol{\Sigma}) = - \frac{d}{2} \log 2\pi - \frac{1}{2} \log |\boldsymbol{\Sigma}| - \frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})
$$

If we assume that $\mathbf{x}$ is iid, we can rewrite this as a summation

$$
\log \mathcal{N}(\mathbf{x};\boldsymbol{\mu}, \boldsymbol{\Sigma}) = - \frac{d}{2} \log 2\pi - \frac{N}{2} \log |\boldsymbol{\Sigma}| - \frac{1}{2} \sum_{i=1}^N(\mathbf{x}_i - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x}_i-\boldsymbol{\mu})
$$



#### Trace-Trick

We can rewrite the distance function using the trace-trick.

$$
\log \mathcal{N}(\mathbf{x};\boldsymbol{\mu}, \boldsymbol{\Sigma}) = - \frac{d}{2} \log 2\pi - \frac{N}{2} \log |\boldsymbol{\Sigma}| - \frac{1}{2} \text{trace} \left[ \boldsymbol{\Sigma}^{-1} \sum_{i=1}^N(\mathbf{x}_i - \boldsymbol{\mu})^\top (\mathbf{x}_i-\boldsymbol{\mu}) \right]
$$


---
## Optimization

### Positivity


#### Softplus

Source: [Ensembles](https://proceedings.neurips.cc/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf)

$$
\text{Softplus}(x) = \log ( 1 + \exp(x))
$$

```python
var_scaled = softplus(var) + 10e-6
```

#### Log variance

```python
var = log(0.5 * var) + 10e-6
```

---
### Log Likelihoods


* Quickly in Batches - [blog](https://gregorygundersen.com/blog/2020/12/12/group-multivariate-normal-pdf/)
* Floating Point Precision - [blog](https://gregorygundersen.com/blog/2019/01/18/log-likelihood/)



### Marginal Distribution $\mathcal{P}(\cdot)$

We have the marginal distribution of $x$

$$\mathcal{P}(x) \sim \mathcal{N}(a, A)$$

and in integral form:

$\mathcal{P}(x) = \int_y \mathcal{P}(x,y)dy$

and we have the marginal distribution of $y$

$$\mathcal{P}(y) \sim \mathcal{N}(b, B)$$

### Conditional Distribution $\mathcal{P}(\cdot | \cdot)$

We have the conditional distribution of $x$  given $y$.

$$\mathcal{P}(x|y) \sim \mathcal{N}(\mu_{a|b}, \Sigma_{a|b})$$

where:

* $\mu_{a|b} = a + BC^{-1}(y-b)$
* $\Sigma_{a|b} = A - BC^{-1}B^T$

and we have the marginal distribution of $y$ given $x$

$$\mathcal{P}(y|x) \sim \mathcal{N}(\mu_{b|a}, \Sigma_{b|a})$$

where:

* $\mu_{b|a} = b + AC^{-1}(x-a)$
* $\Sigma_{b|a} = B - AC^{-1}A^T$

basically mirror opposites of each other. But this might be useful to know later when we deal with trying to find the marginal distributions of Gaussian process functions.

**Source**:

* Sampling from a Normal Distribution - [blog](https://juanitorduz.github.io/multivariate_normal/)
  > A really nice blog with nice plots of joint distributions.
* Two was to derive the conditional distributions - [stack](https://stats.stackexchange.com/questions/30588/deriving-the-conditional-distributions-of-a-multivariate-normal-distribution?noredirect=1&lq=1)
* How to generate Gaussian samples = [blog](https://medium.com/mti-technology/how-to-generate-gaussian-samples-347c391b7959s)


Multivariate Gaussians and Detereminant - [Lecturee Notes](http://courses.washington.edu/b533/lect4.pdf)


---

### Bandwidth Selection


**Scotts**

```python
sigma = np.power(n_samples, -1.0 / (d_dimensions + 4))
```

**Silverman**

```python
sigma = np.power(n_samples * (d_dimensions + 2.0) / 4.0, -1.0 / (d_dimensions + 4)
```
# Gaussian Distribution



### **PDF**

$$f(X)=
\frac{1}{\sqrt{(2\pi)^D|\Sigma|}}
\text{exp}\left( -\frac{1}{2} (x-\mu)^\top \Sigma^{-1} (x-\mu)\right)$$

### **Likelihood**

$$- \ln L = \frac{1}{2}\ln|\Sigma| + \frac{1}{2}(x-\mu)^\top \Sigma^{-1} (x - \mu) + \frac{D}{2}\ln 2\pi $$

### Alternative Representation

$$X \sim \mathcal{N}(\mu, \Sigma)$$

where $\mu$ is the mean function and $\Sigma$ is the covariance. Let's decompose $\Sigma$ as with an eigendecomposition like so

$$\Sigma = U\Lambda U^\top = U \Lambda^{1/2}(U\Lambda^{-1/2})^\top$$

Now we can represent our Normal distribution as:

$$X \sim \mu + U\Lambda^{1/2}Z$$



where:

* $U$ is a rotation matrix
* $\Lambda^{-1/2}$ is a scale matrix
* $\mu$ is a translation matrix
* $Z \sim \mathcal{N}(0,I)$

or also

$$X \sim \mu + UZ$$

where:

* $U$ is a rotation matrix
* $\Lambda$ is a scale matrix
* $\mu$ is a translation matrix
* $Z_n \sim \mathcal{N}(0,\Lambda)$


#### Reparameterization

So often in deep learning we will learn this distribution by a reparameterization like so:

$$X = \mu + AZ $$

where:

* $\mu \in \mathbb{R}^{d}$
* $A \in \mathbb{R}^{d\times l}$
* $Z_n \sim \mathcal{N}(0, I)$
* $\Sigma=AA^\top$ - the cholesky decomposition



---
### **Entropy**

**1 dimensional**

$$H(X) = \frac{1}{2} \log(2\pi e \sigma^2)$$

**D dimensional**
$$H(X) = \frac{D}{2} + \frac{D}{2} \ln(2\pi) + \frac{1}{2}\ln|\Sigma|$$


### **KL-Divergence (Relative Entropy)**

$$
KLD(\mathcal{N}_0||\mathcal{N}_1) = \frac{1}{2}
 \left[ 
 \text{tr}(\Sigma_1^{-1}\Sigma_0) + 
 (\mu_1 - \mu_0)^\top \Sigma_1^{-1} (\mu_1 - \mu_0) -
D + \ln \frac{|\Sigma_1|}{\Sigma_0|}
\right]
$$

if $\mu_1=\mu_0$ then:

$$
KLD(\Sigma_0||\Sigma_1) = \frac{1}{2} \left[ 
\text{tr}(\Sigma_1^{-1} \Sigma_0)  - D  + \ln \frac{|\Sigma_1|}{|\Sigma_0|} \right]
$$

**Mutual Information**

$$I(X)= - \frac{1}{2} \ln | \rho_0 |$$

where $\rho_0$ is the correlation matrix from $\Sigma_0$.

$$I(X)$$
