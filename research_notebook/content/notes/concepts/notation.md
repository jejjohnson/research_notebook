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
|    $\boldsymbol{f}(\;\cdot\;;\boldsymbol{\theta})$     | a latent function parameterized by $\boldsymbol{\theta}$                                             |
|      $\boldsymbol{f}_{\boldsymbol \theta}(\cdot)$      | a latent function parameterized by $\boldsymbol{\theta}$ (succinct version)                          |
|             $\boldsymbol{k}(\cdot, \cdot)$             | kernel or covariance function                                                                        |


---

Below, we have some specificities for these functions and how they translate to real situations.

**Scalar Input - Scalar Output**

$$
f: \mathbb{R} \rightarrow \mathbb{R}
$$

---
**Vector Input - Scalar Output**

$$
\boldsymbol{f}: \mathbb{R}^D \rightarrow \mathbb{R}
$$

---
*Example*: 1D Spatio-Temporal Scalar Field

$$
y = \boldsymbol{f}(x_\phi, t)
$$

---
*Example*: 2D Spatial Scalar Field

We have a 2-dimensional scalar field. The coordinates, $\mathbf{x} \in \mathbb{R}^{D_\phi}$, are 2D, e.g. (lat,lon) coordinates $D_\phi = [\phi, \psi]$. Then each of these coordinates are represented by a scalar value, $y \in \mathbb{R}$. So we have a function, $\boldsymbol{f}$, maps each coordinate, $\mathbf{x}$, of the field to a scalar value, $y$, i.e. $\boldsymbol{f}: \mathbb{R}^{D_\phi} \rightarrow \mathbb{R}$. More explicitly, we can write this function as:

$$
y = \boldsymbol{f}(\mathbf{x}_\phi)
$$

if we stack a lot of samples together, $\mathcal{D} = \left\{ \mathbf{x}_n, y_n\right\}_{n=1}^N$, we get a matrix for the coordinates, $\mathbf{X}$, and a vector for the scalar values, $\mathbf{y}$. So we have $\mathcal{D} = \left\{ \mathbf{X}, \mathbf{y}\right\}$.

**Note**: For more consistent and aesthetically pleasing notation, we have $\mathbf{Y} = \mathbf{y}^\top$ so we can have the dataset, $\mathcal{D} = \left\{ \mathbf{X}, \mathbf{Y}\right\}$

---
*Example*: 2D Spatio-Temporal Scalar Field

$$
y = \boldsymbol{f}(\mathbf{x}_\phi, t)
$$

---
**Vector Input - Vector Output**

$$
\boldsymbol{f}: \mathbb{R}^D \rightarrow \mathbb{R}^P
$$

---
*Example*: 2D Vector Field

We have a 2-dimensional vector field (similar to the above example). The coordinates, $\mathbf{x} \in \mathbb{R}^{D_\phi}$, are 2D, e.g. (lat,lon) coordinates $D_\phi = [\phi, \psi]$. Then each of these coordinates are represented by a **vector** value, $\mathbf{y} \in \mathbb{R}^{P}$. In this case, let the dimensions be the (u,v) fields, i.e. $P=[u,v]$. So we have a function, $\boldsymbol{f}$, maps each coordinate, $\mathbf{x}$, of the field to a vector value, $y$, i.e. $\boldsymbol{f}: \mathbb{R}^{D_\phi} \rightarrow \mathbb{R}^{P}$. More explicitly, we can write this function as:

$$
\mathbf{y} = \boldsymbol{f}(\mathbf{x})
$$

Again, if we stack a lot of samples together, $\mathcal{D} = \left\{ \mathbf{x}_n, \mathbf{y}_n\right\}_{n=1}^N$, we get a stack of matrices, $\mathcal{D} = \left\{ \mathbf{X}, \mathbf{Y}\right\}$.


---
**Special Case: $D = P$**

$$
\boldsymbol{f}:\mathbb{R}^2 \rightarrow \mathbb{R}^2
$$

where each of the functions takes in a 2D vector, $(x,y)$, and outputs a vector, $(u, v)$. This is analagous to scalar field for $u$ and $v$ which appears in physics. So

$$
\begin{aligned}
f_1(x,y) &= u \\
f_2(x,y) &= v
\end{aligned}
$$

We have our functional form given by:

$$
\mathbf{f}\left(
\begin{bmatrix}
x \\ y
\end{bmatrix}
\right) =
\begin{bmatrix}
f_1(x,y) \\ f_2(x,y)
\end{bmatrix} =
\begin{bmatrix}
u \\ v
\end{bmatrix}
$$



---
## Common Terms

|           Notation           | Description                                                                                                                     |
| :--------------------------: | :------------------------------------------------------------------------------------------------------------------------------ |
|           $\theta$           | a parameter                                                                                                                     |
|       $\theta_\alpha$        | a hyperparameter                                                                                                                |
|    $\boldsymbol{\theta}$     | a collection of parameters, $\boldsymbol{\theta}=[\theta_1, \theta_2, \ldots, \theta_p]$                                        |
| $\boldsymbol{\theta_\alpha}$ | a collection of hyperparameters, $\boldsymbol{\theta_\alpha}=[\theta_{\alpha,1}, \theta_{\alpha,2}, \ldots, \theta_{\alpha,p}]$ |


---
## Probability


|                             Notation                             | Description                                                                                                                              |
| :--------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------- |
|                         $\mathcal{X,Y}$                          | the space of data                                                                                                                        |
|                              $P,Q$                               | the probability space of data                                                                                                            |
|                   $f_\mathcal{X}(\mathbf{x})$                    | the probability density function (PDF) on $\mathbf{x}$                                                                                   |
|                   $F_\mathcal{X}(\mathbf{x})$                    | the cumulative density function (CDF) on $\mathbf{x}$                                                                                    |
|                 $F_\mathcal{X}^{-1}(\mathbf{x})$                 | the  Quantile or Point Percentile Function (ppf) (i.e. inverse cumulative density function) on $\mathbf{x}$                              |
|                          $p(x;\theta)$                           | A probability distribution, $p$, of the variable $x$, parameterized by $\theta$                                                          |
|                          $p_\theta(x)$                           | A probability distribution, $p$, of the variable $x$, parameterized by $\theta$ (succinct version)                                       |
|               $p(\mathbf{x};\boldsymbol{\theta})$                | A probability distribution, p, of the multidimensional variable, $\mathbf{x}$, parameterized by $\boldsymbol{\theta}$                    |
|              $p_{\boldsymbol{\theta}}(\mathbf{x})$               | A probability distribution, p, of the multidimensional variable, $\mathbf{x}$, parameterized by $\boldsymbol{\theta}$ (succinct version) |
|                  $\mathcal{N}(x; \mu, \sigma)$                   | A normal distribution for $x$ parameterized by $\mu$ and $\sigma$.                                                                       |
| $\mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma})$ | A multivariate normal distribution for $\mathbf{x}$ parameterized by $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$.                       |
|             $\mathcal{N}(\mathbf{0}, \mathbf{I}_D)$              | A multivariate normal distribution with a zero mean and 1 variance.                                                                      |

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

---

## Field Space

The first case, we have

$$
\mathbf{y} = \boldsymbol{H}(\mathbf{x}) + \epsilon
$$

This represents the state, $\mathbf{x}$, as a representation fo the field


* $\mathbf{x} \in \mathbb{R}^{D_x}$ - state
* $\boldsymbol{\mu}_{\mathbf{x}} \in \mathbb{R}^{D_x}$ - mean prediction for state vector
* $\boldsymbol{\sigma^2}_{\mathbf{x}} \in \mathbb{R}^{D_x}$ - variance prediction for state vector
* $\mathbf{X}_{\boldsymbol{\Sigma}} \in \mathbb{R}^{D_x \times D_x}$ - covariance prediction for state vector
* $\mathbf{X}_{\boldsymbol{\mu}} \in \mathbb{R}^{N \times D_x}$ - variance prediction for state vector


## State (Coordinates)

* $\boldsymbol{x} \in \mathbb{R}^{D_\phi}$ - the coordinate vector
* $\boldsymbol{\mu}_{\boldsymbol{x}} \in \mathbb{R}^{D_\phi}$ - mean prediction for state vector
* $\boldsymbol{\sigma^2}_{\boldsymbol{x}} \in \mathbb{R}^{D_x}$ - variance prediction for state vector
* $\boldsymbol{X}_{\boldsymbol{\Sigma}} \in \mathbb{R}^{D_\phi \times D_\phi}$ - covariance prediction for state vector
* $\boldsymbol{X}_{\boldsymbol{\mu}} \in \mathbb{R}^{N \times D_\phi}$ - variance prediction for state vector


## Observations

* $\mathbf{z} \in \mathbb{R}^{D_z}$ - latent domain
* $\mathbf{y} \in \mathbb{R}^{D_y}$ - observations

## Matrices

* $\mathbf{Z} \in \mathbb{R}^{N \times D_z}$ - latent domain
* $\mathbf{X} \in \mathbb{R}^{N \times D_x}$ - state

* $\mathbf{Y} \in \mathbb{R}^{N \times D_y}$ - observations

---
## Functions

### Coordinates

In this case, we assume that the state, $\mathbf{x} \in \mathbb{R}^{D_\phi}$, are the coordinates, $[\text{lat,lon,time}]$, and the output is the value of the variable of interest, $\mathbf{y}$, at that point in space and time.


* $[\mathbf{K}]_{ij} = \boldsymbol{k}(\mathbf{x}_i, \mathbf{x}_j)$ - covariance matrix for the coordinates
* $\boldsymbol{k}_{\mathbf{X}}(\mathbf{x}_i) = \boldsymbol{k}(\mathbf{X}, \mathbf{x}_i)$ - cross covariance for the data
* $\boldsymbol{k}(\mathbf{x}_i, \mathbf{x}_j) : \mathbb{R}^{D_\phi} \times \mathbb{R}^{D_\phi} \rightarrow \mathbb{R}$ - the kernel function applied to two vectors.


### Data Field

In this case, we assume that the state, $\mathbf{x}$, is the input

* $[\mathbf{C}]_{ij} = \boldsymbol{c}(\mathbf{x}_i, \mathbf{x}_j)$ - covariance matrix for the data field


---
## Operators



---
### Jacobian


So here, we're talking about gradients and how they operate on functions.


**Scalar Input-Output**

$$
f: \mathbb{R} \rightarrow \mathbb{R}
$$

There are no vectors in this operation so this is simply the derivative.

$$
\begin{aligned}
J_f: \mathbb{R} &\rightarrow \mathbb{R} \\
J_f(x) &= \frac{df}{dx}
\end{aligned}
$$


---
**Vector Input, Scalar Output**

$$
\boldsymbol{f} : \mathbb{R}^D \rightarrow \mathbb{R}
$$

This has vector-inputs so the output dimension of the Jacobian operator will be the same dimensionality as the input vector.

$$
\begin{aligned}
\boldsymbol{J}[\boldsymbol{f}](\mathbf{x}) &: \mathbb{R}^{D} \rightarrow \mathbb{R}^D \\
\mathbf{J}_{\boldsymbol{f}}(\mathbf{x}) &=
\begin{bmatrix}
\frac{\partial f}{\partial x_1} &\cdots \frac{\partial f}{\partial x_D}
\end{bmatrix}
\end{aligned}
$$



---
**Vector Input, Vector Output**

$$
\vec{\boldsymbol{f}} : \mathbb{R}^D \rightarrow \mathbb{R}^P
$$

The inputs are the vector, $\mathbf{x} \in \mathbb{R}^D$, and the outputs are a vector, $\mathbf{y} \in \mathbb{R}^P$. So the Jacobian operator will produce a matrix of size $\mathbf{J} \in \mathbb{R}^{P \times D}$.

$$
\begin{aligned}
\boldsymbol{J}[{\boldsymbol{f}}](\mathbf{x}) &: \mathbb{R}^{D} \rightarrow \mathbb{R}^{P\times D}\\
\mathbf{J}[\boldsymbol{f}](\mathbf{x}) &=
\begin{bmatrix}
\frac{\partial f_1}{\partial x_1} &\cdots &\frac{\partial f_1}{\partial x_D} \\
\ldots &\ddots & \ldots \\
\frac{\partial f_p}{\partial x_1} &\cdots &\frac{\partial f_p}{\partial x_D}
\end{bmatrix}
\end{aligned}
$$

---
**Alternative Forms**

I've also seen alternative forms which depends on whether the authors want to highlight the inputs or the outputs.

**Form I**: Highlight the input vectors

$$
\mathbf{J}_{\boldsymbol{f}}(\mathbf{x}) =
\begin{bmatrix}
\frac{\partial \boldsymbol{f}}{\partial x_1} & \cdots & \frac{\partial \boldsymbol{f}}{\partial x_D}
\end{bmatrix} =
\begin{bmatrix}
\frac{\nabla \boldsymbol{f}}{\partial x_1} & \cdots & \frac{\nabla \boldsymbol{f}}{\partial x_D}
\end{bmatrix}
$$

**Form II**: Highlights the output vectors

$$
\mathbf{J}_{\boldsymbol{f}}(\mathbf{x}) =
\begin{bmatrix}
\frac{\partial \boldsymbol{f}_1}{\partial \mathbf{x}} \\ \vdots \\ \frac{\partial \boldsymbol{f}_p}{\partial \mathbf{x}}
\end{bmatrix} =
\begin{bmatrix}
\boldsymbol{\nabla}^\top \boldsymbol{f}_1 \\ \vdots \\ \boldsymbol{\nabla}^\top \boldsymbol{f}_P
\end{bmatrix}
$$


---
### Special Cases

There are probably many special cases where we have closed-form operators but I will highlight one here which comes up in physics a lot.


---
**2D Vector Input, 2D Vector Output**


Recall the special case from the above vectors where the dimensionality of the input vector, $\mathbf{x} \in \mathbb{R}^2$, is the same dimensionality of the output vector, $\mathbf{y} \in \mathbb{R}^2$.

$$
\begin{aligned}
\boldsymbol{f}&:\mathbb{R}^2 \rightarrow \mathbb{R}^2 \\
\end{aligned}
$$

The functional form was:

$$
\mathbf{f}\left(
\begin{bmatrix}
x \\ y
\end{bmatrix}
\right) =
\begin{bmatrix}
f_1(x,y) \\ f_2(x,y)
\end{bmatrix} =
\begin{bmatrix}
u \\ v
\end{bmatrix}
$$

So in this special case, our Jacobian matrix, $\mathbf{J}$, will be:

$$
\mathbf{J}_{\boldsymbol{f}(x,y)} =
\begin{bmatrix}
\frac{\partial u}{\partial x} & \frac{\partial u}{\partial y} \\
\frac{\partial v}{\partial x} & \frac{\partial v}{\partial y}
\end{bmatrix}
$$

**Note**: This is a square matrix because the dimension of the input vector, $(x,y)$, matches the dimension of the output vector, $(u,v)$.


---
### Determinant Jacobian

The determinant of the Jacobian is the amount of (volumetric) change. It is given by:

$$
\det \boldsymbol{J}_{\boldsymbol{f}}(\mathbf{x}): \mathbb{R}^D \rightarrow \mathbb{R}
$$

Notice how we input the vectors, $\mathbf{x}$, and it results in a scalar, $\mathbb{R}$.

**Note**: This can be a very expensive operation especially with high dimensional data. A naive linear function, $\boldsymbol{f}(\mathbf{x}) = \mathbf{Ax}$, will have an operation of $\mathcal{O}(D^3)$. So the name of the game is to try and look at the Jacobian structure and find tricks to reduce the expense of the calculation.

---
**Special Case: Input Vector 2D, Output Vector - 2D**

Again, let's go back to the special case where we have a two input vector, $\mathbf{x}\in \mathbb{R}^2$, and a 2D output vector, $\mathbf{y} \in \mathbb{R}^2$. Recall that the Jacobian matrix for the function, $\boldsymbol{f}$, is a $2\times 2$ square matrix. More generally, we can write this as:

$$
\boldsymbol{J}
\begin{bmatrix}
A(x,y) \\
B(x,y)
\end{bmatrix} =
\begin{bmatrix}
\frac{\partial A}{\partial x} & \frac{\partial A}{\partial y} \\
\frac{\partial B}{\partial x} & \frac{\partial B}{\partial y}
\end{bmatrix}
$$

To calculate the determinant of this Jacobian matrix, we has a closed-form expression. It's given by:

$$
\det \mathbf{J} = AD - BC
$$

So if we apply it to our notation

$$
\det \mathbf{J}_{\mathbf{f}}(x,y) = \frac{\partial f_1}{\partial x}\frac{\partial f_2}{\partial y} - \frac{\partial f_1}{\partial y}\frac{\partial f_2}{\partial x}
$$

This is probably the **easiest** determinant Jacobian to calculate (apart from the scalar-valued which is simply the gradient) and it comes up from time to time in physics.

**Note**: I have seem an alternaive form in the geoscience literature, $\boldsymbol{J}(\boldsymbol{f}_1, \boldsymbol{f}_2)$. I personally don't like this notation because in no way does it specify the **determinant**. I propose a better, clearer notation: $\det \boldsymbol{J}(\boldsymbol{f}_1, \boldsymbol{f}_2)$. Now we at least have the

---
**Example**: This is in the QG PDE. It is given by:

$$
\partial_t q + \boldsymbol{J}(\psi, q) = 0
$$

where the Jacobian operator is given by:

$$
\boldsymbol{J}(\psi, q) = \partial_x \psi \partial_y q - \partial_y \psi \partial_x q
$$

With my updated notation, this would now be:

$$
\partial_t q + \det\boldsymbol{J}(\psi, q) = 0
$$


where the *determinant* Jacobian operator is given:

$$
\det\boldsymbol{J}(\psi, q) = \partial_x \psi \partial_y q - \partial_y \psi \partial_x q
$$

In my eyes, this is clearer. Especially in the papers where people recycle the equations without explicitly defining the operators and their meaning.
