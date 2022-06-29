# Coupling Layers

## Overview

There are three ingredients to coupling layers:

* A split function
* A coupling function, $\boldsymbol{h}$
* A condition function, $\boldsymbol{\Theta}$

---

* Element-wise
* Autoregressive
* Coupling


---
## Algorithm (TLDR)

**Step 1**: Split the features into two disjoint sets.

$$
\mathbf{x}^A, \mathbf{x}^B = \text{Split}(\mathbf{x})
$$

**Step 2**: Apply identity to partial $A$.

$$
\mathbf{z}^A = \mathbf{x}^A
$$

**Step 3**: Apply conditioner, $\Theta$, to partition $A$.

$$
\boldsymbol{\Theta}_A = \boldsymbol{\Theta}(\mathbf{x}^A)
$$

**Step 4**: Apply bijection, $\boldsymbol{h}$, given the parameters from the conditioner, $\boldsymbol{\Theta}$.

$$
\mathbf{z}^B = \boldsymbol{h}(\mathbf{x}^A; \boldsymbol{\Theta}(\mathbf{x}^A))
$$

**Step 5**: Concatenate the two partitions, $A,B$.

$$
\mathbf{z} = \text{Concat}(\mathbf{z}^A, \mathbf{z}^B)
$$

---
## Formulation

We have a bijective, diffeomorphic parameterized function, $\boldsymbol{T}_{\boldsymbol \theta}$, which is a mapping from inputs, $\mathbf{x} \in \mathbb{R}^D$, to some outputs, $\mathbf{y} \in \mathbb{R}^D$, i.e. $\boldsymbol{T}:\mathcal{X}\in\mathbb{R}^{D} \rightarrow \mathcal{Y}\in\mathbb{R}^D$. So more compactly, we can write this as:

$$
\mathbf{y} = \boldsymbol{T}(\mathbf{x};\boldsymbol{\theta})
$$

Let's partition the inputs, $\mathbf{x}$, into two disjoint subspaces

$$
\mathbf{x}^A,\mathbf{x}^B = \text{Split}(\mathbf{x})
$$

where $\mathbf{x}^A \in \mathbb{R}^{D_A}$ and $\mathbf{x}^B \in \mathbb{R}^{D_B}$ where $D = D_A + D_B$.

Now we do not transform the $A$ features however we use a bijective, diffeomorphic coupling function transformation, $\boldsymbol{h}$, where the parameters are given by a conditioner function, $\boldsymbol{\Theta}: \mathbb{R}^{D_A} \rightarrow \mathbb{R}^{D_{\boldsymbol{\theta}}}$. So we can write this explicitly as:

$$
\begin{aligned}
\mathbf{y}^A &= \mathbf{x}^A \\
\mathbf{y}^B &= \boldsymbol{h}\left(\mathbf{x}^B; \boldsymbol{\Theta}(\mathbf{x}^A)\right) \\
\end{aligned}
$$

where



---
## Simplification

$$
\begin{aligned}
\mathbf{y}^A &= \boldsymbol{h}_A\left(\mathbf{x}^A; \boldsymbol{\Theta}(\mathbf{x}^B)\right) \\
\mathbf{y}^B &= \boldsymbol{h}_B\left(\mathbf{x}^B; \boldsymbol{\Theta}(\mathbf{x}^A)\right) \\
\end{aligned}
$$


---
## Log Determinant Jacobian

To see how we can calculate the log determinant jacobian (LDJ), we can demonstrate this with a partition.

$$
\boldsymbol{\nabla}_\mathbf{x}\boldsymbol{T}(\mathbf{x}) = 
\begin{bmatrix}
A & B \\
C &
D
\end{bmatrix}
$$

And so we can simply showcase the LDJ for each of them

$$
\begin{aligned}
A &:= \boldsymbol{\nabla}_{\mathbf{x}^A}(\mathbf{x}^A) = 1 \\
B &:= \boldsymbol{\nabla}_{\mathbf{x}^B}(\mathbf{x}^A) = 0 \\
\end{aligned}
$$

The $C$ partition is a function of the input partition, $\mathbf{x}^A$ but the derivative is wrt to

---
The Jacobian, $\mathbf{J},\nabla$

$$
\boldsymbol{\nabla}_\mathbf{x}\boldsymbol{T}(\mathbf{x}) = 
\begin{bmatrix}
\mathbf{I} & \mathbf{0} \\
\boldsymbol{\nabla}_{\mathbf{x}^A}\boldsymbol{h}(\mathbf{x}^B; \boldsymbol{\Theta}(\mathbf{x}^A)) &
\boldsymbol{\nabla}_{\mathbf{x}^B}\boldsymbol{h}(\mathbf{x}^B; \boldsymbol{\Theta}(\mathbf{x}^A))
\end{bmatrix}
$$

We end up with a very simple formulation

$$
\det \boldsymbol{\nabla}_{\mathbf{x}}\boldsymbol{T}(\mathbf{x}) = \det \boldsymbol{\nabla}_{\mathbf{x}} \boldsymbol{T}(\mathbf{x}^B; \boldsymbol{\Theta}(\mathbf{x}^A))
$$


---
## General Form

We can write this more generally if we consider a masked transformation. This formulation was introduced in the original [REALNVP paper]().

$$

$$


---

## Extensive Literature Review


### Coupling Layers

The concept of coupling layers was introduced in the [NICE paper]() whereby the authors used an additive coupling layer. They also coined this type of coupling layer as *non-volume preserving* because the `logdetjacobian` is equal to 1 in this case. The [REALNVP paper]() was later extended to affine coupling layers 



* Additive (NICE, 2014)
* Affine (RealNVP, 2015)
* Spline Function (NSF, 2018; LRS, 2020)
* Neural Network (NAF,2018; BNAF, 2019)
* Affine MixtureCDF (Flow++, 2019)
* Hierarchical (HINT, 2019)
* Incompressible (GIN, 2020)
* Lop Sided (IRN, 2020)
* Bipartite (DenseFlow, 2021)
* MixtureCDF (Gaussianization, 2022)


### Conditioners

* Fully Connected (NICE, 2014)
* Convolutional (NICE, 2014)
* ResNet (NSF, 2018)
* (Gated) Self-Attention (Flow++, 2019)
* Transformer (Nystroemer) (DenseFlow, 2021)
* AutoEncoder (Highway) (OODF, 2020)
* Equivariant (NeuralFlows, 2021)
* Fourier Neural Operator (Gaussianization, 2022)
* Wavelet Neural Operator (Gaussianization, 2022)


### Masks

* Half (NICE, 2014)
* Permutation (RealNVP, 2015)
* Checkerboard (RealNVP, 2015)
* Horizontal (OODFlows, 2020)
* Vertical (OODFlows, 2020)
* Center (OODFlows, 2020)
* Cycle (OODFlows, 2020)
* Augmentation (DenseFlow, 2021)

---
## 

