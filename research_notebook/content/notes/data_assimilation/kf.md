# Kalman Filter

Implementation of the Kalman Filter




---
## Methods



### Filter


$$
p(\mathbf{z}_t|\mathbf{x}_{1:t}) = \mathcal{N}(\mathbf{z}_t|\boldsymbol{\mu}_{\mathbf{z}_{t|t-1}}, \boldsymbol{\Sigma}_{\mathbf{z}_{t|t-1}})
$$

This is a two step process:

1. Prediction Step for the Transition Model
2. Measurement Step for the Emission Model


---

#### Predict Step

$$
\begin{aligned}
\text{(Pred. Mean)} \hspace{10mm} \boldsymbol{\mu}_{\mathbf{z}_{t|t-1}} &=  \mathbf{F}\mathbf{z}_{t-1} \\
\text{(Pred. Cov.)} \hspace{10mm} \boldsymbol{\Sigma}_{\mathbf{z}_{t|t-1}} &= \mathbf{F} \boldsymbol{\Sigma}_{t-1}\mathbf{F}^\top+\mathbf{Q}
\end{aligned}
$$



```python
# predictive mean (state), t|t-1
mu_z_t_cond = F @ mu_t
# predictive covariance (state), t|t-1
Sigma_z_t_cond = F @ Sigma_t @ F.T + Q
```

---
#### Update Step

$$
p(\mathbf{z}_t|\mathbf{x}_{1:t}) = \mathcal{N}(\mathbf{z}_t|\boldsymbol{\mu}_{\mathbf{z}_t}, \boldsymbol{\Sigma}_{\mathbf{z}_t})
$$


where:
* $\boldsymbol{\mu}_{\mathbf{z}_t}$ is the estimation of the state mean given the observations
* $\boldsymbol{\Sigma}_{\mathbf{z}_t}$ is the estimation of the state cov given the observations.


$$
\begin{aligned}
\text{(Pred. Mean)} \hspace{10mm} \boldsymbol{\mu}_{\mathbf{x}_{t}} &=  \mathbf{H}\boldsymbol{\mu}_{\mathbf{z}_{t|t-1}} \\
\text{(Innovation)} \hspace{10mm} \boldsymbol{\Sigma}_{\mathbf{x}_{t}} &= \mathbf{H} \boldsymbol{\Sigma}_{\mathbf{z}_{t|t-1}} \mathbf{H}^\top + \mathbf{R}
\end{aligned}
$$

```python
# Pred Mean (obs)
mu_x_t_cond = H @ mu_z_t_cond
# Pred Cov (obs)
Sigma_x_t_cond = H @ Sigma_z_t_cond @ H.T + R
```



We then need to do a correction. This is done via a 2-step process


##### Innovation & Kalman Gain

$$
\begin{aligned}
\text{(Innovation)} \hspace{10mm} \boldsymbol{r}_{t} &= \mathbf{x}_t - \boldsymbol{\mu}_{\mathbf{x}_{t}} \\
\text{(Kalman Gain)} \hspace{10mm} \mathbf{K}_{t} &= \boldsymbol{\Sigma}_{\mathbf{z}_{t|t-1}} \mathbf{H}^{-1} \boldsymbol{\Sigma}_{\mathbf{x}_{t}}^{-1}
\end{aligned}
$$

```python
# innovation
r_t = x_t - mu_x_t_cond
# kalman gain
K_t = Sigma_z_t_cond @ H.T @ inv(Sigma_x_t_cond)
```


##### Correction

$$
\begin{aligned}
\text{(Est. Mean)} \hspace{10mm} \boldsymbol{\mu}_{\mathbf{z}_{t}} &=  \boldsymbol{\mu}_{\mathbf{z}_{t|t-1}} + \mathbf{K}_t\boldsymbol{r}_{t} \\
\text{(Est. Cov.)} \hspace{10mm} \boldsymbol{\Sigma}_{\mathbf{z}_{t}} &= (\mathbf{I} - \mathbf{K}_t\mathbf{H})\boldsymbol{\Sigma}_{\mathbf{z}_{t|t-1}}
\end{aligned}
$$


```python
# estimated state mean
mu_z_t = mu_z_t_cond + K_t @ r_t
# estimated state covariance 
Sigma_z_t = (I - K_t @ H) @ Sigma_z_t_cond
```
---
## Filtering



#### Psuedocode

**Inputs**

* $\mathbf{A} \in \mathbb{R}^{D_x \times D_x}$ - transition matrix
* $\mathbf{Q} \in \mathbb{R}^{D_x \times D_x}$ - transition noise
* $\mathbf{H} \in \mathbb{R}^{D_y \times D_x}$ - emission matrix
* $\boldsymbol{\mu}_0 \in \mathbb{R}^{D_x}$ - prior for state, $\mathbf{x}$, mean
* $\boldsymbol{\Sigma}_0 \in \mathbb{R}^{D_x \times D_x}$ - prior for state, $\mathbf{x}$, covariance
* 
**Parameters**

* $\mathbf{x} \in \mathbb{R}^{D_x}$ - transition matrix
* $\mathbf{y} \in \mathbb{R}^{D_y }$ - transition noise
* $\mathbf{H} \in \mathbb{R}^{D_y \times D_x}$ - emission matrix


**Function**


```python
def _sequential_kf(F, Rs, H, ys, Qs, m0, P0, masks, return_predict=False):
  """
  Parameters
  ----------
  F : np.ndarray, shape=(state_dim, state_dim)
    transition matrix for the transition function
  Rs : np.ndarray, shape=(n_time, state_dim, state_dim)
    noise matrices for the transition function
  H : np.ndarray, shape=(obs_dim, state_dim)
    the emission matrix for the emission function
  Qs: np.ndarray, shape=(n_time, obs_dim, obs_dim)
    the noises for the emission 
  ys : np.ndarray, shape=(batch, n_time, obs_dim)
    the observations
  m0 : np.ndarray, shape=(obs_dim)

  """

    def body(carry, inputs):
        # ==================
        # Unroll Inputs
        # ==================
        # extract constants
        y, R, Q, mask = inputs
        # extract next steps (mu, sigma, ll)
        m, P, ell = carry

        # ==================
        # Predict Step
        # ==================
        m_ = F @ m
        P_ = F @ P @ F.T + Q

        # ==================
        # Update Step
        # ==================

        # residuals
        obs_mean = H @ m_
        HP = H @ P_
        S = HP @ H.T + R

        # log likelihood
        ell_n = mvn_logpdf(y, obs_mean, S, mask)
        ell = ell + ell_n

        K = solve(S, HP).T

        # correction step
        m = m_ + K @ (y - obs_mean)
        P = P_ - K @ HP
        if return_predict:
            return (m, P, ell), (m_, P_)
        else:
            return (m, P, ell), (m, P)

    (_, _, loglik), (fms, fPs) = scan(
      f=body,
      init=(m0, P0, 0.),
      xs=(ys, Qs, Rs, masks)
    )

    return loglik, fms, fPs
```

```python
def kalman_filter(dt, kernel, y, noise_cov, mask=None, return_predict=False):
    """
    Run the Kalman filter to get p(fₙ|y₁,...,yₙ).
    Assumes a heteroscedastic Gaussian observation model, i.e. var is vector valued
    :param dt: step sizes [N, 1]
    :param kernel: an instantiation of the kernel class, used to determine the state space model
    :param y: observations [N, D, 1]
    :param noise_cov: observation noise covariances [N, D, D]
    :param mask: boolean mask for the observations (to indicate missing data locations) [N, D, 1]
    :param return_predict: flag whether to return predicted state, rather than updated state
    :return:
        ell: the log-marginal likelihood log p(y), for hyperparameter optimisation (learning) [scalar]
        means: intermediate filtering means [N, state_dim, 1]
        covs: intermediate filtering covariances [N, state_dim, state_dim]
    """
    if mask is None:
        mask = np.zeros_like(y, dtype=bool)
    Pinf = kernel.stationary_covariance()
    minf = np.zeros([Pinf.shape[0], 1])

    # get constant params
    F = ... # transition matrix
    H = ... # emission matrix

    # generate noise matrices
    Rs = ... # generate noise for transitions
    Qs = ... # generate noise for emissions

    ell, means, covs = _sequential_kf(As, Qs, H, y, noise_cov, minf, Pinf, mask, return_predict=return_predict)
    return ell, (means, covs)
```


---
## Smoothing


---
## Likelihoods


---
## Missing Data

<!-- ---
## Task List


**Components**

* [ ] Filter
  * [ ] Sequential
  * [ ] Parallel
* [ ] Smoothing
  * [ ] Sequential
  * [ ] Parallel
* [ ] Log Probability
* [ ] Posterior
  * [ ] Predictions, obs
  * [ ] Samples, state
* [ ] Masked Log (Missing Data)
* [ ] Augmentation 
  * [ ] EOF
  * [ ] AutoEncoder (AE)
  * [ ] Flow Model
  * [ ] SurVae Flow


**Fixed Param Examples**

* [ ] Random Walk
* [ ] Object Tracking


**Training Procedures**

* [ ] Gradient-Based
  * [ ] SGD
* [ ] Iterative
  * [ ] E/M
* [ ] PPL - Numpyro
* [ ] PPL - Pyro

**Demonstrations**

* [ ] Random Walk
  * [ ] Fixed Params
  * [ ] Free Params
* [ ] Object Tracking
  * [ ] Fixed Params
  * [ ] Free Params
* [ ] Lorenz-63
* [ ] Lorenz-84
* [ ] Lorenz-96
* [ ] 1D Spatio-Temporal
  * [ ] Missing Data 
* [ ] 2D Spatio-Temporal
  * [ ] Missing Data
* [ ] 3D Spatio-Temporal
  * [ ] Missing Data
 -->


---
## Resources


#### Courses

> *Sensor Fusion and Non-Linear Filtering* - [Youtube Playlist](https://youtube.com/playlist?list=PLTD_k0sZVYFqjFDkJV8GE2EwfxNK59fJY)


> Kalman Filter (ML TV) - [Video I](https://youtu.be/LioOvUZ1MiM) | [Video II](https://youtu.be/8oeg2fdV8jE)


> Kalman Filtering and Applications in Finance - [Youtube](https://youtu.be/R63dU5w_djQ)


#### Papers

* {cite:p}`ouala2018NNKF`


---
### Code

* Kalman Filter with Dask/XArray - [Repo](https://github.com/CedricTravelletti/Climate/tree/main)