# Models


---
## Explicit


These models take the input features, $\mathbf{x}_D$, and output a scalar/vector value for the signal. So this is defined as:

$$
\mathbf{y} = \boldsymbol{f}(\mathbf{x}_D;\boldsymbol{\theta})
$$

**Examples**:
* DeepVoxels
* Neural Volumes


**Pros n Cons**:
* 👍 Computational Efficiency
* 👎 Memory Efficient
* 👎 Online Multiscale
* 👎 Pruning


---
## Global Implicit

These take the coordinates as inputs and output a scalar/vector value for the signal. So this is defined as:

$$
y = \boldsymbol{f}_{\boldsymbol{\theta}}(\mathbf{x}_\phi)
$$

**Examples**: 
* DeepSDF
* (Modulated) SIREN
* FFN
* Positional Encoding
* Multiplicative Filter Networks


**Pros n Cons**:
* 👎 Computational Efficient
* 👍 Memory Efficient
* 👎 Online Multiscale
* 👎 Pruning


---
## Local Implicit

These use a hybrid of the above two approaches. They have some function that takes in the features and the coordinates and outputs a signal.

$$
\begin{aligned}
\mathbf{z} &= \text{Encoder}(\mathbf{x}_D) \\
\mathbf{y} &= \boldsymbol{f}(\mathbf{x}_\phi, \mathbf{z};\boldsymbol{\theta})
\end{aligned}
$$

Notice how we have an encoder for the input vector $\mathbf{x}_D$ because often we want these in some lower dimensional space. Then we concatenate the feature vector and the coordinate vector to learn the output signal value.

**Examples**:
* Local Implicit Image Functions
* Deep Local Shapes
* Convolutional Occupancy Nets
* Neural Geometric Level of Detail
* Neural Sparse Voxel Fields


**Pros n Cons**

* 👎 / 👍 Computational Efficient
* 👎 / 👍 Memory Efficient
* 👎 Online Multiscale
* 👍 / 👎 Pruning

They are usually more efficient (computationally and memory). But they only represent the signal only on a single scale. This might be difficult for pruning approaches later on.


---

## Hybrid Implicit-Explicit

This framework is a hybrid implicit-explicit architecture as an attempt to bypass the issues of each of the individual approaches. It has a *global coordinate encoder* which transforms the global coordinates to a set of features. It also has a *local coordinate decoder* which transforms the encoded features and the local coordinates to output the signal.

$$
\begin{aligned}
\mathbf{z} &= \text{GlobalEncoder}_{\boldsymbol\theta}(\mathbf{x}_{\phi_G}) \\
\mathbf{y} &= \text{LocalDecoder}_{\boldsymbol\theta}(\mathbf{x}_{\phi_L},\mathbf{z})
\end{aligned}
$$


**Examples**:
* ACORN


**Pros n Cons**:
* 👍 Computational Efficient
* 👍 Memory Efficient
* 👍 Online Multiscale
* 👍 Pruning