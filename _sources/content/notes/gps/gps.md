# Gaussian Processes

Consider the regression setting where we assume the following model:
$$
 y = \boldsymbol{f}(\mathbf{x}) + \epsilon
$$

where $\mathbf{x}$ is a discriminate vector of inputs, $\mathbf{f}(\cdot)=\left[f_1, \ldots, f_N \right]$ is a latent GP function, and $\epsilon \sim \mathcal{N} (0, \sigma_y^2)$ is a independently, identically distributed (i.i.d.) Gaussian noise parameter. We place a GP $\textbf{prior}$ for $p(\boldsymbol{f})$ s.t.

$$
p (\boldsymbol{f}|\mathbf{X}, \boldsymbol{\theta}) \sim \mathcal{GP} \left(\mathbf{m}_{\boldsymbol\psi}, \mathbf{K}_{\boldsymbol\phi}\right),
$$

where $\mu_\mathcal{GP}$ and $\mathbf{K}_\mathcal{GP}$ are the mean and covariance matrix for the GP, $\boldsymbol{\theta} = \left\{ \boldsymbol{\psi,\phi}\right\}$ are the parameters of the model and $\mathbf{X}$ is the data. Combining this prior with the regression problem model from the previous equation, we assume a **likelihood** function:

$$
\begin{equation}
    p (y | \boldsymbol{f}, \mathbf{X}) \sim \mathcal{N} (y | \boldsymbol{f} (\mathbf{x}), \sigma_y^2 \mathbf{I})
\end{equation}
$$

We can invoke Bayes rule giving us the joint posterior distribution:
$$
\begin{equation}
    p (\boldsymbol{f}, \boldsymbol{f}_* | y) = \frac{p(\boldsymbol{f}, \boldsymbol{f}_*)p(y|\boldsymbol{f})}{p(y)}
\end{equation}
$$
where $p(y)$ is the marginal likelihood which we can obtain by integrating out the latent variables $\boldsymbol{f}$:
$$
\begin{aligned}
    p(y) &= \int_{\boldsymbol{f}} p(y,\boldsymbol{f})d\boldsymbol{f} \\
    &= \mathcal{N} (y | \boldsymbol{\mu}_\mathcal{GP}, \mathbf{K} + \sigma^2\mathbf{I}) \\
    &= \mathcal{N} (y | \boldsymbol{\mu}_\mathcal{GP}, \mathbf{K}_{\mathcal{GP}})
\end{aligned}
$$
In a regression setting, we are more interested in predictions; given some parameters and some data, what is the predictive function $\boldsymbol{f}$? This is known as  the {posterior} distribution:

$$
\begin{equation}
    p( \boldsymbol{f} |  y, \mathbf{X}, \boldsymbol{\theta}) \sim \mathcal{N} (\boldsymbol{\mu}_{\mathcal{GP}} , \boldsymbol{\nu}^2_{GP})
\end{equation}
$$

---


### Inference

First, given the joint distribution of $\boldsymbol{f}, \boldsymbol{f}_*$ conditioned on $\mathbf{X,X_*}$

$$
\begin{equation} 
p(\boldsymbol{f}, \boldsymbol{f}_*|\mathbf{x}, \mathbf{x}_*)=\mathcal{N}\left( 
    \begin{bmatrix}  
    \boldsymbol{f} \\ \boldsymbol{f}_*
    \end{bmatrix}; 
    \begin{bmatrix}
    \boldsymbol{m}(\mathbf{x}) \\ \boldsymbol{m}(\mathbf{x}_*)
    \end{bmatrix},
    \begin{bmatrix}
    \mathbf{K} & \mathbf{K}_* \\
    \mathbf{K}_* & \mathbf{K}_{**}
    \end{bmatrix} \right)
\end{equation}
$$

If we condition on our training inputs $D=(\mathbf{X}, y)$, we can come up with a **predictive distribution** for test points $\mathbf{x}_*$ via

$$
\begin{equation}
    p(\boldsymbol{f}_* | \boldsymbol{f}) = \mathcal{N} (\boldsymbol{\mu}_{\mathcal{GP}*} , \boldsymbol{\nu}^2_{\mathcal{GP}**})
\end{equation}
$$

and we can give the GP predictive mean and variance functions as

$$
\begin{aligned}
    \boldsymbol{\mu}_{\mathcal{GP}} &= \underbrace{m (\mathbf{x}_*)}_{\text{Prior Mean}} + \underbrace{\boldsymbol{k}_{*} \mathbf{K}^{-1}}_{\text{Kalman Gain}}\underbrace{(y- m (\mathbf{X}))}_{\text{Error}}\\
    \boldsymbol{\nu}^2_{\mathcal{GP}} &= k_{**} - \boldsymbol{k}_{*} \mathbf{K}^{-1}\boldsymbol{k}_{*}^{\top}.
\end{aligned}
$$

If we integrate out the $\boldsymbol{f}$ (or just take the conditional distribution of the joint PDF), then we get:
$$
\begin{aligned}
    p(\boldsymbol{f}_*|\mathbf{x}_*, \mathbf{x}, y) &= \int_{\boldsymbol{f}} p(\boldsymbol{f}|\mathbf{x}, y) p(\boldsymbol{f}_*|\mathbf{x}_*,y)d\boldsymbol{f} \\
    &= \mathcal{N}(\boldsymbol{f}_*|\mu_*, \Sigma_*)
\end{aligned}
$$
and the joint distribution of $\boldsymbol{f}_*$ and unobserved $y$:
$$
\begin{equation} 
p(y, \boldsymbol{f}_*|\mathbf{x}, \mathbf{x}_*)=\mathcal{N}\left( 
    \begin{bmatrix}  
    \boldsymbol{f} \\ \boldsymbol{f}_*
    \end{bmatrix}; 
    \begin{bmatrix}
    \mathcal{GP}M(\mathbf{x}) \\ \mathcal{GP}M(\mathbf{x}_*)
    \end{bmatrix},
    \begin{bmatrix}
    \mathcal{GP}K(\mathbf{x}, \mathbf{x}) + \sigma^2\mathbf{I} & \mathcal{GP}K(\mathbf{x}, \mathbf{x}_*) \\
    \mathcal{GP}K(\mathbf{x}, \mathbf{x}_*) & \mathcal{GP}K(\mathbf{x}_*, \mathbf{x}_*)
    \end{bmatrix} \right)
\end{equation}
$$
which gives us the mean predictions and the variance in our predictions:
$$
\begin{align}
    \boldsymbol{\mu}_{\mathcal{GP}} &= \underbrace{ \boldsymbol{m}(\mathbf{x}_*)}_{\text{Prior Mean}} + \underbrace{\mathbf{k}_{*} \mathbf{K}^{-1}}_{\text{Kalman Gain}}\underbrace{(y- \boldsymbol{m}(\mathbf{X}))}_{\text{Error}}= \boldsymbol{m}(\mathbf{x}_*) + \mathbf{K}_{*} \alpha \\
    \boldsymbol{\nu}^2_{\mathcal{GP}} &= \underbrace{k_{**}}_{\text{Prior Variance}} - \mathbf{k}_{*} \mathbf{K}_{\mathcal{GP}}^{-1}\mathbf{k}_{*}^{\top}
\end{align}
$$
where $\alpha = \mathbf{K}^{-1}(y-m(\mathbf{X}))$ and $\mathbf{K}_\mathcal{GP}=\mathbf{K}_\theta(\mathbf{X,X})+\sigma^2\mathbf{I}$. This is the typical formulation~\ref{fig:intro-probabilistic} which assumes that the output of $\mathbf{x}$ (and $\mathbf{x}_*$) is deterministic. In section~\ref{chapter3:egp}, we will look at the case where $\mathbf{x}_*$ is stochastic.

### GP Training

In GP model inference, one maximizes the likelihood of the data $D$ given the hyper-parameters $\boldsymbol{\theta}, \boldsymbol{\sigma}_y^2$. The marginal likelihood is given by:

$$
\begin{equation}
    p (y | \mathbf{X}, \theta) = \mathcal{N} \left( y | \mathcal{GP}M, \mathcal{GP}K + \sigma_y^2\mathbf{I} \right)
\end{equation}
$$

We can find the hyper-parameters $\boldsymbol{\theta}$ by maximizing the marginal log-likelihood. So fully expanding of the eq: \ref{eq:gp_ml}, we get:
$$
\begin{equation}
    \log p (y | \mathbf{X}, \theta) = -\underbrace{\frac{1}{2}(y - \mathcal{GP}M)^{\top} \mathbf{K}_{\mathcal{GP}}^{-1} (y - \mathcal{GP}M)}_{\text{Data-Fit}}  - \underbrace{\frac{1}{2} \log \left| \mathbf{K}_{GP} \right|}_{\text{Complexity}} - \frac{N}{2}\log 2\pi
\end{equation}
$$

This maximization automatically embodies Occam's razor which does a trade-off between model complexity and overfitting. This is closed form for all GPs but these days, we typically use automatic differentiation toolboxes to alleviate some of the burden.  Irregardless, the two most expensive calculations are within this procedure as the inversion of $\matinv{\mathbf{K}}_{\mathcal{GP}}$ and the $\Det{\mathbf{K}_{\mathcal{GP}}}$; since $\mathbf{K} \in \Real^{N \times N}$, then these calculations are $\bigO (N^3)$ in operations and $\bigO (N^2)$ in memory costs. The kernel function is one of the most important aspects within the GP training regime. Once the kernel has been chosen to best reflect the problem at hand it has been found in \cite{GPPRIOR2018} that any prior over the hyper-parameters does not provide significant improvements in GP predictions. However, note that the community is notorious for using the isotropic RBF kernel by default when conducting research. This kernel is the most flexible among the kernel family but not necessarily the most expressive~\cite{AUTOGP2017}. 



### Drawbacks

\begin{displayquote}
"\textit{It is important to keep in mind that Gaussian processes are not appropriate priors for all problems.}" \\
-- Neal, 1998
\end{displayquote}

It is important to note that although the GP algorithm is one of the most trusted and reliable algorithms, it is not always the best algorithm to use for all problems. Below we mention a few drawbacks that the standard GP algorithm has along with some of the standard approaches to overcoming these drawbacks.

\vspace{2mm} \noindent \textbf{Gaussian Marginals}. GPs have problems modeling heavy-tailed, asymmetric or multi-modal marginal distributions. There are some methods that change the likelihood so that it is heavy tailed~\citep{GPTSTUDENT2011,GPTSTUDENT2014} but this would remove the conjugacy of the likelihood term which would incur difficulties during fitting. Deep GPs and latent covariate models are an improvement to this limitation. A very popular approach is to construct a fully Bayesian model. This entails hyperpriors over the kernel parameters and Monte carlo sampling methods such as Gibbs sampling~\citep{GPGIBBS08}, slice sampling~\citep{GPSLICE2010}, Hamiltonian Monte Carlo~\citep{GPHMC2018}, and Sequential Monte Carlo~\citep{GPSMC15}. These techniques will capture more complex distributions. With the advent of better software~\citep{PYMC16,NUMPYRO2019} and more advanced sampling techniques like a differentiable iterative NUTS implementation~\citep{NUMPYRO2019}, the usefulness of MC schemes is resurfacing.

\vspace{2mm} \noindent \textbf{Limited Number of Moments}. This is related to the previous limitation: the idea that an entire function can be captured in terms of two moments: a mean and a covariance. There are some relationships which are difficult to capture without an adequate description, e.g. discontinuities~\citep{Neal96} and non-stationary processes, and thus is a limitation of the GP priors we choose. The advent of warping the inputs or outputs of a GP has becoming a very popular technique to deal with the limited expressivity of kernels. Input warping is popular in methods such as deep kernel learning whereby a Neural network is used to capture the features and are used as inputs to the kernel function output warping is common in chained~\citep{GPCHAINED2016} and heteroscedastic methods where the function output is warped by another GP to capture the noise model of the data. Deep Gaussian processes~\citep{Damianou2015} can be thought of input and output warping methods due the multi-layer composition of function inputs and outputs.

\vspace{2mm} \noindent \textbf{Linearity of Predictive Mean}. The predictive mean of a GP is linear to the observations, i.e. $\mu_{GP}=\mathbf{K}\alpha$. This essentially is a smoother which can be very powerful but also will miss key features. If there is some complex structured embedded within the dataset, then a GP model can never really capture this irregardless of the covariance function found.

\vspace{2mm} \noindent \textbf{Predictive Covariance}. The GP predictive variance is a function of the training inputs and it is independent of the observed inputs. This is important if the input data has some information which could be used to help determine the regions of uncertainty, e.g. the gradient. An example would be data on a spatial grid whereby some regions points would have more certainty than others which could be obtained by knowing the input location and not necessarily the expected output.

