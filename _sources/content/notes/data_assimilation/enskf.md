# Ensemble Kalman Filter

Notes for the Ensemble Kalman Filter (EnsKF).



---
## Model

$$
\begin{aligned}
\mathbf{x}(t) &= \mathbf{Mx}(t-1) + \boldsymbol{\eta}(t), &\boldsymbol{\eta} \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}) \\
\mathbf{y}(t) &= \mathbf{Hx}(t) + \boldsymbol{\epsilon}(t), &\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{R}) \\
\end{aligned}
$$

---
### Forecast Step

$$
\begin{aligned}
\mathbf{x}^f(t) &= \mathbf{Mx}^a(t-1) \\
\mathbf{P}^f(t) &= \mathbf{MP}^a(t-1) \mathbf{M}^\top + \mathbf{Q}
\end{aligned}
$$

---
### Analysis Step (Forward)

$$
\begin{aligned}
\mathbf{x}^a(t) &= \mathbf{x}^f(t) + \mathbf{K}^a[ \mathbf{y}(t) - \mathbf{H}\mathbf{x}^f(t)] \\
\mathbf{P}^a(t) &= [ \mathbf{I} - \mathbf{K}^a \mathbf{H}]\mathbf{P}^f(t)
\end{aligned}
$$

where:

$$
\mathbf{K}^a = \mathbf{P}^f(t) \mathbf{H}^\top [ \mathbf{HP}^f(t)\mathbf{H}^\top + \mathbf{R}]^{-1}
$$

---
### Analysis Step (Backwards)

$$
\begin{aligned}
\mathbf{x}^s(t) &= \mathbf{x}^a(t) + \mathbf{K}^s[ \mathbf{x}^s(t+1) - \mathbf{x}^f(t+1)] \\
\mathbf{P}^s(t) &= \mathbf{P}^a(t) + \mathbf{K}^s[ \mathbf{P}^s(t+1) - \mathbf{P}^f(t+1)](\mathbf{K}^s)^\top 
\end{aligned}
$$

where:

$$
\mathbf{K}^s = \mathbf{P}^a(t) \mathbf{M}^\top [ \mathbf{P}^f (t+1)]^{-1}
$$


---
## Model

In man applications, there are cases where we cannot write the transition model and/or the emission model in a simple matrix operation. Instead, we use non-linear functions.

$$
\begin{aligned}
\mathbf{x}(t) &= \boldsymbol{m}(\mathbf{x}, t-1) + \boldsymbol{\eta}(t) \\
\mathbf{y}(t) &= \boldsymbol{h}(\mathbf{x}, t) + \boldsymbol{\epsilon}(t)
\end{aligned}
$$

This gives us added flexibility and this also allows us to model more complexities found within the data. However, we cannot use the linear Kalman filter equations for inference.

**Solutions**: In general, you'll find 3 classes of algorithms to deal with this: Extended Kalman Filters (EKF), Ensemble Kalman Filters (EnKF) and Particle Filters (PF).

---
### Ensemble Kalman Filter

In the EnKF, we use an ensemble to track the state of the system. So $\mathbf{x}(t)$, is approximated by $N_e$ members. These members are used to track the evolution of the state. Then the empirical mean and covariance is calculated followed by the Kalman filter equations to update the state based on observations. This provides a robust and fast solution.

$$
\mathbf{x}(t) = (\mathbf{x}_1(t), \mathbf{x}_2(t), \ldots, \mathbf{x}_{N_e}(t))
$$

$$
\mathbf{x}_{(i)}^f(t) = \boldsymbol{m}(\mathbf{x}_{(i)}^a(t-1)) + \boldsymbol{\eta}_{(i)}(t), \hspace{5mm} \forall i=1,2, \ldots, N_e
$$

$$
\mathbf{K} = \mathbf{P}^f(t) \mathbf{H}^\top (\mathbf{H} \mathbf{P}^f(t) \mathbf{H}^\top + \mathbf{R})^{-1}
$$

---
#### Forecast Step

$$
\begin{aligned}
\mathbf{x}_{(i)}^f(t) &= \boldsymbol{m}(\mathbf{x}_{(i)}^a(t-1)) + \boldsymbol{\eta}_{(i)}(t), &\forall i=1,2, \ldots, N_e \\
\mathbf{x}^f(t) &= \frac{1}{N_e}\sum_{i=1}^{N_e} \mathbf{x}_{(i)}(t) \\
\mathbf{P}^f(t) &= \frac{1}{N_e - 1} \sum_{i=1}^{N_e} \left( \mathbf{x}_{(i)}(t) - \bar{\mathbf{x}}(t) \right)\left( \mathbf{x}_{(i)}(t) - \bar{\mathbf{x}}(t) \right)^\top \\
\mathbf{K} &= \mathbf{P}^f(t)\mathbf{H}^\top \left( \mathbf{HP}^f(t)\mathbf{H}^\top + \mathbf{R} \right)
\end{aligned}
$$


---
##### Deterministic Ensemble Kalman Filter

$$
\begin{aligned}
\mathbf{x}^a(t) &= \mathbf{x}^f(t) + \mathbf{K}(\mathbf{y}(t) - \boldsymbol{h}(\mathbf{x}^f(t))) \\
\mathbf{P}^a(t) &= \left( \mathbf{I} - \mathbf{KH} \right)\mathbf{P}^f(t)
\end{aligned}
$$


---
##### Stochastic Ensemble Kalman Filter

$$
\begin{aligned}
\mathbf{y}^f_{(i)}(t) &= \boldsymbol{h}(\mathbf{x}_{(i)}^f(t)) + \boldsymbol{\epsilon}_{(i)}(t)\\
\mathbf{x}^a_{(i)}(t) &= \mathbf{x}_{(i)}^f(t) + \mathbf{K}\left( \mathbf{y}(t) - \mathbf{y}^f_{(i)}(t) \right) \\
\mathbf{x}^a(t) &= \frac{1}{N_e}\sum_{i=1}^{N_e} \mathbf{x}_{(i)}^a(t) \\
\mathbf{P}^a(t) &= \frac{1}{N_e - 1} \sum_{i=1}^{N_e} \left( \mathbf{x}_{(i)}^a(t) - \mathbf{x}^a(t) \right)\left( \mathbf{x}_{(i)}^a(t) - \mathbf{x}^a(t) \right)^\top \\
\end{aligned} 
$$

---
### Computational Issues

**Sherman-Morrison Woodbury**

$$
(AB^\top (C + BAB^\top))^{-1} = (A^{-1} + B^\top CB)^{-1}B^\top C^{-1}
$$

**Kalman Gain** (Observation State)

$$
\mathbf{K} = \mathbf{PH}^\top (\mathbf{HPH}^\top + \mathbf{R})^{-1}
$$

**Reduced Space**

$$
\mathbf{K} = \mathbf{S}(\mathbf{I} + (\mathbf{HS})^\top \mathbf{R}^{-1}\mathbf{HS})^{-1}(\mathbf{HS})^\top \mathbf{R}^{-1}
$$

where $\mathbf{S} = \frac{\mathbf{x}}{\sqrt{N-1}}$.

#### Inverting Symmetric Definite Matrices

**Standard**

```python
C_inv = linalg.pinv(C, hermitian=True)
```

**SVD**

$$
\begin{aligned}
\mathbf{P} &= \mathbf{U\Lambda V}^\top \\
\mathbf{P}^{-1} &= \mathbf{V} {\Lambda}^{-1} \mathbf{U}^\top
\end{aligned}
$$

```python
U, D, VH = linalg.svd(C, full_matrices=False, hermitian=True)
C_inv = VH.T @ diag(1/D) @ U.T
```

*Note*: This is useful for data assimilation with square roots

**Cholesky**

$$
\begin{aligned}
\mathbf{P} &= \mathbf{LL}^\top \\
\mathbf{LU} &= \mathbf{I} \\
\mathbf{L}^\top \mathbf{P}^{-1} &= \mathbf{U}
\end{aligned}
$$

```python
L = linalg.cholesky(C)
U = solve_triangular(L, eye(L.shape[1]), lower=True)
C_inv = solve_triangular(L.transpose(1, 2), U, lower=False)
```


*Note*: This is useful for optimal interpolation.


#### Analysis Step (SVD)

**Forecast Ensemble Anomalies**

$$
\mathbf{S}^f = \frac{1}{N-1}\left( \mathbf{x}^f - \bar{\mathbf{x}}^f \right)
$$

**Forecast Ensemble Covariance**

$$
\mathbf{P}^f = \mathbf{S}^f\mathbf{S}^{f^\top}
$$

**Kalman Gain** (Ensemble Space)

$$
\mathbf{K} = \mathbf{S}^f\left( \mathbf{I} + (\mathbf{HS})^\top \mathbf{R}^{-1}\mathbf{HS} \right)^{-1}( \mathbf{HS}^f)^\top \mathbf{R}^{-1}
$$


```python
S @ inv(eye(Ne) + (H @ S).T @ inv(R) @ H @ S) @ (H @ S).T @ inv(R)
```

`[Ns x D] = [Ns x D] @ []`

**SVD** of Signal/Noise

$$
(\mathbf{HS})^\top\mathbf{R}^{-1}\mathbf{HS} = \mathbf{U\Lambda U}^\top
$$

**Kalman Gain with SVD**

$$
\mathbf{K} = \mathbf{S}^f \mathbf{U}(\mathbf{I}+\mathbf{\Lambda})^{-1/2}\mathbf{U}^\top
$$

**Analysis Ensemble of Anomalies**

$$
\mathbf{S}^a = \mathbf{S}^f \mathbf{U}(\mathbf{I} + \mathbf{\Lambda})^{-1/2} \mathbf{U}^\top
$$

**Analysis Ensemble of Covariances**

$$
\mathbf{P}^a = \mathbf{P}^f - \mathbf{KHP}^f = \mathbf{S}^a\mathbf{S}^{a\;\top}
$$

**Analysis Mean**

$$
\bar{\mathbf{x}}^a = \bar{\mathbf{x}}^f + \mathbf{K}(\mathbf{y}_\text{obs} - \mathbf{H}\bar{\mathbf{x}}^f)
$$

**Analysis Ensemble**

$$
\mathbf{x}^a = \bar{\mathbf{x}}^a + \sqrt{N - 1}\;\mathbf{S}^a
$$

---
#### Perturbed Observations

* Sequential Data Assimilation with Non-Linear Quasi-Geostrophic Model using Monte Carlo methods to forecast error statistics - Evensen (1994)
* Data Assimilation using Ensemble Kalman Filter Technique - Houtekamer & Mitchell (1998)

---
**Inputs**

We have some ensemble members.

* $\mathbf{X}_{e} \in \mathbb{R}^{N_e \times D_x}$
* $\mathbf{y} \in \mathbb{R}^{D_y}$


---
**Sample the Observations given the state**

Here, we propagate these samples through our emission function, $\boldsymbol{H}$, to give us some predictions for the observations.

$$
\begin{aligned}
\mathbf{Y}_e &= \boldsymbol{H}(\mathbf{X}_e; \boldsymbol{\theta})
\end{aligned}
$$

where $\mathbf{Y}_e \in \mathbb{R}^{N_e \times D_y}$.

```python
# emission function
Y_ens = H(X_ens, t, rng)
```

`[Ns x Dy]`


---
**Calculate 1st Moment**

Now we need to calculate the 1st moment (the mean) to approximate the Gaussian distribution. We need to do this for both the predicted state, $\mathbf{X}_e$, and the predicted observations, $\mathbf{Y}_e$.

$$
\begin{aligned}
\bar{\mathbf{x}} &= \frac{1}{N_e}\sum_{i}^{N_e} \mathbf{X}_{e\;(i)} \\
\bar{\mathbf{y}} &= \frac{1}{N_e}\sum_{i}^{N_e} \mathbf{Y}_{e\;(i)} \\
\end{aligned}
$$

where $\bar{\mathbf{x}} \in \mathbb{R}^{D}$


---

Now we need to calculate the 1st and 2nd moments (mean, covariance) to approximate the Gaussian distribution. We need to do this for both the predicted state, $\mathbf{X}_e$, and the predicted observations, $\mathbf{Y}_e$.



$$
\begin{aligned}
\bar{\mathbf{x}} = \mathbf{X}_e - 
\end{aligned}
$$

```python
# mean of state ensembles (Dx)
x_mu = mean(X_ens)
# mean of obs ensembles (Dy)
y_mu = mean(Y_ens)
```


---
**Calculate Residuals**

We calculate the errors (residuals) for our ensemble of predictions, $\mathbf{Y}_e$, wrt to the true observation value, $\mathbf{y}_\text{obs}$.


---
## Resources

* Auto-differentiable Ensemble Kalman Filters - Chen et al (2021) - [arxiv](https://arxiv.org/abs/2107.07687) | [PyTorch](https://github.com/ymchen0/torchEnKF)