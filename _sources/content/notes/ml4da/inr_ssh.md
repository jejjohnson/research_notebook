# Implicit Neural Representations


---

## Trade-offs

### Pros

**Mesh-Free**

**Lots of Data**

### Cons

**Transfer Learning**


---
## Data

$$
\begin{aligned}
\mathbf{x}_\phi \in \mathbb{R}^{D_\phi}, \;\;\; \mathbf{u} \in \mathbb{R}^{}
\end{aligned}
$$

---
## Model

$$
\boldsymbol{f_\theta}:\mathcal{X} \rightarrow \mathcal{U}
$$


---
## Architectures

We are interested in the case of regression. We have the following generalized architecture.

$$
\begin{aligned}
\mathbf{x}^{(1)} &= \boldsymbol{\phi} \left( \mathbf{x} ; \boldsymbol{\gamma}\right) \\
\mathbf{x}^{(\ell+1)} &= \text{NN}_\ell \left( \mathbf{x}^{(\ell)}; \boldsymbol{\theta}_\ell\right)\\
\boldsymbol{f}(\mathbf{x}; \boldsymbol{\theta},\boldsymbol{\gamma}) &= \mathbf{w}^{(L)}\mathbf{x}^{(L)} + \mathbf{b}^{(L)}
\end{aligned}
$$

where $\boldsymbol{\phi}$ is the basis transformation with some hyperparameters $\gamma$, $\text{NN}$ is the neural network layer parameterized by $\boldsymbol{\theta}$, and we have $L$ layers, $L = \{1, 2, \ldots, \ell, \ldots, L-1, L\}$

#### Standard Neural Network

In the standard neural network, we typically have the following standard functions

$$
\begin{aligned}
\boldsymbol{\phi}(\mathbf{x}) &= \mathbf{x} \\
 \text{NN}_{siren} \left( \mathbf{x}^{(\ell)}; \boldsymbol{\theta}\right) &= \boldsymbol{\sigma} \left( \mathbf{w}^{(\ell)} \mathbf{x}^{(\ell)} + \mathbf{b}^{(\ell)} \right), \hspace{10mm} \boldsymbol{\theta} = \{ \mathbf{w}^{(\ell)}, \mathbf{b}^{(\ell)} \}
\end{aligned}
$$

So more explicitly, we can write it as:

$$
\begin{aligned}
\mathbf{x}^{(1)} &= \mathbf{x} \\
\boldsymbol{f}^{(\ell)}(\mathbf{x}^{(\ell)}) &= \boldsymbol{\sigma} \left( \mathbf{w}^{(\ell)} \mathbf{x}^{(\ell)} + \mathbf{b}^{(\ell)} \right)\\
\boldsymbol{f}^{(L)}(\mathbf{x}^{(L)}) &= \mathbf{w}^{(L)}\mathbf{x}^{(L)} + \mathbf{b}^{(L)}
\end{aligned}
$$

where $\ell = \{1, 2, \ldots, L-1\}$.
Noteably:

* The first layer is the identity (i.e. there is not basis function transformation)
* The second layer is the standard neural network architecture, i.e. a linear function and a nonlinear activation function
* The final layer is always a linear function (in regression; classification would have a sigmoid)

---
### Fourier Features


$$
\boldsymbol{\phi} \left(\mathbf{x}\right) = 
\begin{bmatrix}
\sin \left( \boldsymbol{\omega}\mathbf{x}\right) \\
\cos \left( \boldsymbol{\omega} \mathbf{x}\right)
\end{bmatrix},\hspace{10mm} \boldsymbol{\omega} \sim p(\boldsymbol{\omega};\gamma)
$$


|  Method   | Kernel |                       Distribution                       |
| :-------: | :----: | :------------------------------------------------------: |
| Gaussian  |        | $\mathcal{N}(\mathbf{0},\frac{1}{\sigma^2}\mathbf{I}_r)$ |
| Laplacian |        |                    $\text{Cauchy}()$                     |
|  Cauchy   |        |                    $\text{Laplace}()$                    |
|  Matern   |        |                    $\text{Bessel}()$                     |
| ArcCosine |        |                                                          |

#### Alternative Formulation

$$
\boldsymbol{\phi}(\mathbf{x}) = \sqrt{\frac{2}{D_{rff}}}\cos \left( \boldsymbol{\omega}\mathbf{x} + \boldsymbol{b}\right)
$$

where $\boldsymbol{\omega} \sim p(\boldsymbol{\omega})$ and $\boldsymbol{b} \sim \mathcal{U}(0,2\pi)$.

**Source**: 
* [Blog](https://gregorygundersen.com/blog/2019/12/23/random-fourier-features/) - Gregory Gundersen
* Random Features for Large-Scale Kernel Machines - Rahimi & Recht (2008) - [Paper]()
* Random Features for Kernel Approximation: A Survey on Algorithms, Theory, and Beyond - Liu et al (2021)
* Scalable Kernel Methods via Doubly Stochastic Gradients - Dai et al (2015)

### SIREN

$$
\boldsymbol{\sigma} = \sin \left( \boldsymbol{\omega}_0 (\mathbf{wx} + b)\right)
$$

$$
\boldsymbol{\sigma} = \boldsymbol{\alpha} \odot \sin \left( \mathbf{wx} + \mathbf{b} \right)
$$

$$
\text{FiLM}(\mathbf{x}) = \boldsymbol{\alpha} \odot \mathbf{x} + \boldsymbol{\beta} 
$$

* pi-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis - Chan et al (2021)
* COIN++: 

#### Extended

$$
\begin{aligned}
\boldsymbol{\sigma} = \sin \left( \boldsymbol{\gamma}(\mathbf{wx} + b\right) + \boldsymbol{\beta})
\end{aligned}
$$

where $\boldsymbol{\gamma}$ corresponds to the frequencies and $\boldsymbol{\beta}$ corresponds to the phase shifts.


---
## Modulation

Modulation is 
$$
\boldsymbol{f}^\ell(\mathbf{x},\mathbf{z};\boldsymbol{\theta}) := \boldsymbol{h}_M^\ell\left(\;\text{NN}(\mathbf{x};\boldsymbol{\theta}_{NN})\;,\; \text{M}(\mathbf{z};\boldsymbol{\theta}_{M}) \;\right) 
$$

where $NN$ is the output of the neural network wrt the input, $\mathbf{x}$, where $M$ is the output of the modulation function wrt the latent variable, $\mathbf{z}$, and $\times$ is an arbitrary operator.

* Additive Layer
* Affine Layer
* Neural Implicit Flowss
* Neural Flows

---

[FILM, 2020]()


[Mehta, 2021]()


[Dupoint, 2022]()

Neural Implicit Flows [Pan, 2022]()

Neural 



---
### Affine Modulation


**Affine Modulations**

$$
\begin{aligned}
\mathbf{z}^{(1)} &= \mathbf{x} \\
\mathbf{z}^{(k+1)} &= \boldsymbol{\sigma} \left( \left(\mathbf{w}^{(k)} \mathbf{z}^{(k)} + \mathbf{b}^{(k)}\right)\odot \boldsymbol{s}_m(\mathbf{z}) + \boldsymbol{a}_m(\mathbf{z}) \right)\\
\boldsymbol{f}(\mathbf{x}) &= \mathbf{w}^{K}\mathbf{z}^{K} + \mathbf{b}^{(k)}
\end{aligned}
$$

**Shift Modulations**

#### Neural Implicit Flows

In this work, we have a version of the Modulated Siren as mentioned above. However, they use a version that separates the space and time neural networks.
$$
\boldsymbol{f}(\mathbf{x}_\phi, t) = \text{NN}_{space}(\mathbf{x}_\phi;\text{NN}_{time}(t))
$$


---

### Multiplicative Filter Networks

$$
\begin{aligned}
\mathbf{z}^{(1)} &= \mathbf{x} \\
\mathbf{z}^{(k+1)} &= \boldsymbol{\sigma} \left( \mathbf{w}^{(k)} \mathbf{z}^{(k)} + \mathbf{b}^{(k)} \right) \\
\boldsymbol{f}(\mathbf{x}) &= \mathbf{w}^{K}\mathbf{z}^{K} + \mathbf{b}^{(k)}
\end{aligned}
$$

where $K = \{1, 2, \ldots, K\}$


#### Non-Linear Functions


**FOURIERNET**

This method corresponds to the random Fourier Feature transformation.

$$
\boldsymbol{g}^{(\ell)}(\mathbf{x};\boldsymbol{\theta}^{(\ell)}) = \sin\left( \mathbf{w}^{(\ell)}\mathbf{x} + \mathbf{b}^{(\ell)}\right)
$$


where the parameters to be learned are:

$$
\boldsymbol{\theta}^{(\ell)} = \{\mathbf{w}_d^{(\ell)}, \;\; \mathbf{b}^{(\ell)}_d \}
$$


**GABORNET**

This method tries to improve upon the Fourier representation. The Fourier representation has global support and would have more difficulties representing more local features. The Gabor filter (see below) will be able to capture both frequency and spatial locality component.

$$
\boldsymbol{g}^{(\ell)}(\mathbf{x};\boldsymbol{\theta}^{(\ell)}) = \exp\left( - \frac{\gamma_d^{(\ell)}}{2}||\mathbf{x} - \boldsymbol{\mu}_d^{(\ell)}||_2^2 \right) \odot \sin\left( \mathbf{w}^{(\ell)}\mathbf{x} + \mathbf{b}^{(\ell)}\right)
$$

where the parameters to be learned are:

$$
\boldsymbol{\theta}^{(\ell)} = \{ \gamma_d^{(\ell)} \in \mathbb{R},\;\;\boldsymbol{\mu}_d^{(\ell)}, \;\; \mathbf{w}_d^{(\ell)}, \;\; \mathbf{b}^{(\ell)}_d \}
$$



---
## Probabilistic


### Deterministic

$$
\mathcal{L}(\boldsymbol{\theta}) = \underset{\boldsymbol{\theta}}{\text{argmin }} \lambda \sum_{n \in \mathcal{D}} ||\boldsymbol{f}(\mathbf{x}_n;\boldsymbol{\theta}) - \boldsymbol{u}_n||_2^2 - \log p(\boldsymbol{\theta})
$$


### Normalizing Flows



### Bayesian

* Random Feature Expansions (RFEs)


---
## Physics Constraints


### Mass

### Momentum

### QG Equations


---
## Applications


### Interpolation


### Surrogate Modeling


### Sampling


---
## Feature Engineering

$$
\mathbf{x} \in \mathbb{R}^{D_\phi}, \hspace{10mm} D = \{ \text{lat, lon, time} \}
$$

---

### Spatial Features

For the spatial features, we have spherical coordinates (i.e. longitude and latitude)

$$
\begin{aligned}
x &= r \cos(\lambda)\cos(\phi) \\
y &= r \cos(\lambda)\sin(\phi) \\
z &= r \sin(\lambda)
\end{aligned}
$$

where $\lambda$ is the latitude, $\phi$ is the longitude and $r$ is the radius. Here $x,y,z$ are bounded between 0 and 1.

---

### Temporal Features


#### Tanh

$$
f(t) = \tanh(t)
$$

#### Fourier Features


#### Sinusoidal Positional Encoding

$$
\boldsymbol{\phi}(t) =
\begin{bmatrix}
\sin(\boldsymbol{\omega}_k t) \\
\cos(\boldsymbol{\omega}_k t)
\end{bmatrix}
$$

where
$$
\boldsymbol{\omega}_k = \frac{1}{10,000^{\frac{2k}{d}}}
$$

**Sources**:

* Transformer Architecture: The Positional Encoding - Amihossein - [Blog](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)
* Position Information in Transformers: An Overview - Dufter et al (2021) - Arxiv - [Paper]()
* Rethinking Positional Encoding - Zheng et al (2021) - Arxiv [Paper]()
* Self-Attention with Functional Time Representation Learning - Xu et al (2019) - Arxiv - [Paper]()
* AI Coffee Break with Letita - Video [1](https://www.youtube.com/watch?v=1biZfFLPRSY) | [2](https://www.youtube.com/watch?v=M2ToEXF6Olw)
* Attention is all you need. A Transformer Tutorial: 5. Positional Encoding - [Video](https://www.youtube.com/watch?v=LSCsfeEELso)