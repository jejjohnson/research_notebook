# Notation


---
## Spaces

|      Notation      | Description                                        |
| :----------------: | :------------------------------------------------- |
| $N \in \mathbb{N}$ | the number of samples (natural number)             |
| $D \in \mathbb{N}$ | the number of features/covariates (natural number) |


---
## Variables

|                  Notation                  | Description                                                                                                                                   |
| :----------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------- |
|            $x,y \in \mathbb{R}$            | scalers (real numbers)                                                                                                                        |
| $\mathbf{x} \in \mathbb{R}^{D_\mathbf{x}}$ | a $D_\mathbf{x}$-dimensional column vector, usually the input.                                                                                |
| $\mathbf{y} \in \mathbb{R}^{D_\mathbf{y}}$ | a $D_\mathbf{y}$-dimensional column vector, usually the output.                                                                               |
|            $x^j \in \mathbb{R}$            | the $j$-th feature from a vector, $\mathbf{x} \in \mathbb{R}^{D}$, where $(x^j)_{1\leq j \leq D}$                                             |
|            $x_i \in \mathbb{R}$            | the $i$-th sample from a vector, $\mathbf{x} \in \mathbb{R}^{N}$, where $(x_i)_{1\leq j \leq D}$                                              |
|  $\mathbf{X} \in \mathbb{R}^{N \times D}$  | a collection of $N$ input vectors, $\mathbf{X}=[\mathbf{x}_1, \ldots, \mathbf{x}_N]^\top$, where $\mathbf{x} \in \mathbb{R}^{D}$              |
|  $\mathbf{Y} \in \mathbb{R}^{N \times P}$  | a collection of $N$ output vectors, $\mathbf{Y}=[\mathbf{y}_1, \ldots, \mathbf{y}_N]^\top$, where $\mathbf{y} \in \mathbb{R}^{P}$             |
|    $\mathbf{x}^{j} \in \mathbb{R}^{N}$     | the $j$-th feature from a collection of vectors, $\mathbf{X}$, where $(\mathbf{x}^{j} )_{1\leq j \leq D}$                                     |
|    $\mathbf{x}_{i} \in \mathbb{R}^{D}$     | the $i$-th sample from a collection of vectors, $\mathbf{X}$, where $(\mathbf{x}_{i})_{1\leq i \leq N}$                                       |
|          $x_{i}^j \in \mathbb{R}$          | the $i$-th sample and $j$-th feature from a collection of vectors, $\mathbf{X}$, where $(\mathbf{x}_{i}^{j})_{1\leq i \leq N,1\leq j \leq D}$ |

---
## Functions

|                        Notation                        | Description                                                                                          |
| :----------------------------------------------------: | :--------------------------------------------------------------------------------------------------- |
|       $f : \mathcal{X} \rightarrow \mathcal{Y}$        | a latent function that operates on a scaler and maps a space $\mathcal{X}$ to a space $\mathcal{Y}$  |
| $\boldsymbol{f} : \mathcal{X} \rightarrow \mathcal{Y}$ | a latent function  that operates on a vector and maps a space $\mathcal{X}$ to a space $\mathcal{Y}$ |
|             $\boldsymbol{k}(\cdot, \cdot)$             | kernel or covariance function                                                                        |


---
## Common Terms

|       Notation        | Description                                                                                                 |
| :-------------------: | :---------------------------------------------------------------------------------------------------------- |
|       $\theta$        | parameter or hyperparameter                                                                                 |
| $\boldsymbol{\theta}$ | a collection of parameters or hyperparameters, $\boldsymbol{\theta}=[\theta_1, \theta_2, \ldots, \theta_p]$ |


---
## Probability


|                        Notation                         | Description                                                                                                           |
| :-----------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------- |
|                     $\mathcal{X,Y}$                     | the space of data                                                                                                     |
|                          $P,Q$                          | the probability space of data                                                                                         |
|               $f_\mathcal{X}(\mathbf{x})$               | the probability density function (PDF) on $\mathbf{x}$                                                                |
|               $F_\mathcal{X}(\mathbf{x})$               | the cumulative density function (CDF) on $\mathbf{x}$                                                                 |
|            $F_\mathcal{X}^{-1}(\mathbf{x})$             | the  Quantile or Point Percentile Function (ppf) (i.e. inverse cumulative density function) on $\mathbf{x}$           |
|                      $p(x;\theta)$                      | A probability distribution, $p$, of the variable $x$, parameterized by $\theta$                                       |
|           $p(\mathbf{x};\boldsymbol{\theta})$           | A probability distribution, p, of the multidimensional variable, $\mathbf{x}$, parameterized by $\boldsymbol{\theta}$ |
|              $\mathcal{N}(x; \mu, \sigma)$              | A normal distribution for $x$ parameterized by $\mu$ and $\sigma$.                                                    |
| $\mathcal{N}(x; \boldsymbol{\mu}, \boldsymbol{\Sigma})$ | A multivariate normal distribution for $\mathbf{x}$ parameterized by $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$.    |
|         $\mathcal{N}(\mathbf{0}, \mathbf{I}_D)$         | A multivariate normal distribution with a zero mean and 1 variance.                                                   |

---
## Information Theory

|          Notation           | Description                                        |
| :-------------------------: | :------------------------------------------------- |
|           $I(X)$            | Self-Information for a rv $X$.                     |
|           $H(X)$            | Entropy of a rv $X$.                               |
|           $TC(X)$           | Total correlation (multi-information) of a rv $X$. |
|          $H(X,Y)$           | Joint entropy of rvs $X$ and $Y$.                  |
|          $I(X,Y)$           | Mutual information between two rvs $X$ and $Y$.    |
| $\text{D}_{\text{KL}}(X,Y)$ | Jullback-Leibler divergence between $X$ and $Y$.   |

---
## Gaussian Processes

|                  Notation                  | Description                                                                                                       |
| :----------------------------------------: | :---------------------------------------------------------------------------------------------------------------- |
|              $\boldsymbol{m}$              | mean function for a Gaussian process.                                                                             |
|                $\mathbf{K}$                | kernel function for a Gaussian process                                                                            |
| $\mathcal{GP}(\boldsymbol{m}, \mathbf{K})$ | Gaussian process distribution parameterized by a mean function, $\boldsymbol{m}$ and kernel matrix, $\mathbf{K}$. |
|      $\boldsymbol{\mu}_\mathcal{GP}$       | GP predictive mean function.                                                                                      |
|    $\boldsymbol{\sigma}^2_\mathcal{GP}$    | GP predictive variance function.                                                                                  |
|     $\boldsymbol{\Sigma}_\mathcal{GP}$     | GP predictive covariance function.                                                                                |

