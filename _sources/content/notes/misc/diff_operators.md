# Differential Operators



---
## Difference



This is the case where we have a function $\boldsymbol{f}:\mathbb{R}^D \rightarrow \mathbb{R}^P$ which maps an input vector $\vec{\mathbf{x}}$ to a scalar value. We denote this operation as

$$
\begin{aligned}
\text{Difference}
&:= \partial_x \boldsymbol{f} \\
&= \nabla_i \boldsymbol{f}
\end{aligned}
$$

So this operator is

$$
\partial_i \boldsymbol{f}: \mathbb{R} \rightarrow \mathbb{R}
$$

where $_i$ is the index of the input vector, $\vec{\mathbf{x}}$, of the function $\boldsymbol{f}$. We can also right the functional transformation version

$$
\partial_i [\boldsymbol{f}](\vec{\mathbf{x}}): \mathbb{R}^D \rightarrow \mathbb{R}
$$


````{admonition} Pseudo-Code
:class: dropdown info

```python
u: Array["P Dx"] = ...
du_dx: Array["P"] = derivative(u, step_size=dx, axis=0, order=1, accuracy=2)
du_dx: Array["P"] = derivative(u, step_size=dy, axis=1, order=1, accuracy=2)

# second partial derivative
d2u_dx2: Array["P"] = derivative(u, step_size=dx, axis=0, order=2, accuracy=2)
d2u_dy2: Array["P"] = derivative(u, step_size=dy, axis=1, order=2, accuracy=2)
```

````


````{admonition} Resources
:class: dropdown seealso

* 3Blue3Brown - Visualization of Derivatives - [Video](https://www.youtube.com/watch?v=CfW845LNObM)
* 3Blue3Brown - [Essence of Calculus](https://youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)

````

---

## Gradient

> The directions of the fastest change and the directional derivative. Tells me locally where something is increasing or decreasing the fastest. Tells us the rate of change at every point (a vector direction of change)

> Turns a *scalar* field into a *vector* field!


$$
\begin{aligned}
\text{Gradient}
&:=\text{grad}(\boldsymbol{f}) \\
&= \boldsymbol{\nabla} \boldsymbol{f}\\
&=
\begin{bmatrix}
\frac{\partial}{\partial x} \\
\frac{\partial}{\partial y} \\
\frac{\partial}{\partial z}
\end{bmatrix} \cdot \boldsymbol{f} \\
&=
\begin{bmatrix}
\frac{\partial f}{\partial x} \\
\frac{\partial f}{\partial y} \\
\frac{\partial f}{\partial z}
\end{bmatrix}\\
&= \boldsymbol{J}_i(\boldsymbol{f})
\end{aligned}
$$

So the operation is:

$$
\text{grad}(\boldsymbol{f}) = \boldsymbol{\nabla}\boldsymbol{f}: \mathbb{R} \rightarrow \mathbb{R}^D
$$

where $D$ is the size of the input vector, $\vec{\mathbf{x}}$. Let's take a scalar field with vector-valued inputs.

$$
f=\boldsymbol{f}(x,y,z)=\boldsymbol{f}(\vec{\mathbf{x}}) \hspace{10mm} f:\mathbb{R}^3\rightarrow \mathbb{R}
$$

Then the gradient is

$$
\text{grad}(\boldsymbol{f})
=
\begin{bmatrix}
\frac{\partial \boldsymbol{f}}{\partial x} \\
\frac{\partial \boldsymbol{f}}{\partial y} \\
\frac{\partial \boldsymbol{f}}{\partial z}
\end{bmatrix}
$$


We can also write the functional transformation version:

$$
\text{grad}[\boldsymbol{f}](\vec{\mathbf{x}}) = \boldsymbol{\nabla}\boldsymbol{f}: \mathbb{R}^D \rightarrow \mathbb{R}^D
$$



````{admonition} Pseudo-Code
:class: dropdown info

```python
# scalar value
x: Array[""] = ...
y: Array[""] = ...
output: Array[""] = f(x,y)

# vectorized
x: Array["N"] = ...
y: Array["N"] = ...
output: Array["N"] = vmap(f, args=(0,1))(x,y)

# meshgrid
x: Array["N"] = ...
y: Array["M"] = ...
X: Array["N M"], Y: Array["N M"] = meshgrid(x,y, indexing="ij")
x: Array["NM"] = flatten(X)
y: Array["NM"] = flatten(Y)
output: Array["NM"] = vmap(f, args=(0,1))(x,y)
```

````

````{admonition} Example
:class: dropdown info


Let's take the function

$$
f(x,y) = x^2 + y^2 \hspace{10mm} f:\mathbb{R}\times\mathbb{R}\rightarrow \mathbb{R}
$$

Then the gradient would be

$$
\nabla f =
\begin{bmatrix}
2x \\
2y
\end{bmatrix}
$$

$$
\begin{aligned}
\mathbf{x},\mathbf{y}&\in\Omega && && \Omega_x\in\mathbb{R}\\
\boldsymbol{f}(\mathbf{x},\mathbf{y}) &= \mathbf{x}^2 + \mathbf{y}^2 && &&\boldsymbol{f}:\mathbb{R}^N\times\mathbb{R}^N\rightarrow \mathbb{R}^{N\times 2} \\
\text{grad}[\mathbf{f}](\mathbf{x},\mathbf{y}) &=
\begin{bmatrix}
2\mathbf{x} \\
2\mathbf{y}
\end{bmatrix} && && \text{grad}[\boldsymbol{f}]: \mathbb{R}^{N\times 2}\rightarrow\mathbb{R}^{N\times 2}
\end{aligned}
$$

````

### Vector Fields

````{admonition} Psuedo-Code
:class: dropdown info

```python
# scalar value
x: Array[""] = ...
y: Array[""] = ...
output: Array[""] = f(x,y)

# vectorized
x: Array["N"] = ...
y: Array["N"] = ...
output: Array["N"] = vmap(f, args=(0,1))(x,y)

# meshgrid
x: Array["N"] = ...
y: Array["M"] = ...
X: Array["N M"], Y: Array["N M"] = meshgrid(x,y, indexing="ij")
x: Array["NM"] = flatten(X)
y: Array["NM"] = flatten(Y)
output: Array["NM"] = vmap(f, args=(0,1))(x,y)
```

````

---
## Jacobian

````{admonition} Resources
:class: dropdown seealso

* Mathemaniac Visualization of Jacobian - [Video](https://www.youtube.com/watch?v=wCZ1VEmVjVo)
* Serpentine Integral Visualization of the Change of Variables and the Jacobian - [Video](https://www.youtube.com/watch?v=hhFzJvaY__U&t=515s)

````

---
## Divergence

> Turns a *vector* field into a *scalar* field. It measures locally how much stuff is flowing away or flowing towards a single point in space. Basically, how much the vector field is expanding outwards or into a point in space!

> How we measure sources and sinks!

Let's take a vector valued function:

$$
\vec{\boldsymbol{f}}:\mathbb{R}^{D}\rightarrow\mathbb{R}^D
$$

The divergence operator does the following transformation:

$$
\text{div }(\vec{\boldsymbol{f}}): \mathbb{R}^D \rightarrow \mathbb{R}
$$


Then the divergence operator is the following:

$$
\begin{aligned}
\text{Divergence }
&:= \text{div }(\vec{\boldsymbol{f}}) \\
&= \vec{\nabla}\cdot \vec{\boldsymbol{f}} \\
&= \left(\frac{\partial}{\partial x}, \frac{\partial}{\partial y}, \frac{\partial}{\partial z} \right)\cdot \left(f_1, f_2, f_3\right) \\
&= \left(\frac{\partial}{\partial x}, \frac{\partial}{\partial y}, \frac{\partial}{\partial z} \right)\cdot \left(f\hat{i} + f\hat{j} + f\hat{k}\right) \\
&= \frac{\partial f_1}{\partial x} + \frac{\partial f_2}{\partial y} +  \frac{\partial f_3}{\partial z}
\end{aligned}
$$

We can also write the functional transformation version that maps a vector input, $\vec{\mathbf{x}}$, through the transformation $\boldsymbol{f}(\cdot)$ to the output of the divergence operator $\text{div}(\cdot)$. We have the following:

$$
\text{div}\left[\vec{\boldsymbol{f}}\right](\vec{\mathbf{x}}): \mathbb{R}^D \rightarrow \mathbb{R}
$$

````{admonition} Pseudo-Code
:class: dropdown info

```python
u: Array["P D"]

# from scratch (differences)
du_dx: Array["P"] = difference(u, step_size=dx, axis=0, order=1, accuracy=4)
du_dy: Array["P"] = difference(u, step_size=dy, axis=1, order=1, accuracy=4)
u_div: Array["P"] = du_dx + du_dy

# from scratch (gradient)
u_grad: Array["P D"] = gradient(u, step_size=(dx,dy,dz), order=1, accuracy=4)
u_div: Array["P"] = np.sum(u_grad, axis=1)

# divergence operator
u_div: Array["P"] = divergence(u, step_size=(dx,dy), accuracy=4)
```

````


## Curl

> How we measure rotation!

$$
\begin{aligned}
\text{Curl}
&:= \text{curl}(\vec{\boldsymbol{f}}) \\
&= \nabla\times \vec{\boldsymbol{f}} \\
&= \det
\begin{vmatrix}
\hat{i} & \hat{j} & \hat{j} \\
\frac{\partial}{\partial x} & \frac{\partial}{\partial y} & \frac{\partial}{\partial z} \\
f_1 & f_2 & f_3
\end{vmatrix} \\
&= \begin{bmatrix}
\frac{\partial f_2}{\partial z} -  \frac{\partial f_3}{\partial y}\\
\frac{\partial f_1}{\partial z} -  \frac{\partial f_3}{\partial x} \\
\frac{\partial f_1}{\partial z} -  \frac{\partial f_2}{\partial y}
\end{bmatrix} \\
&= \left(\frac{\partial f_2}{\partial z} -  \frac{\partial f_3}{\partial y}\right)\hat{i}
\left( \frac{\partial f_1}{\partial z} -  \frac{\partial f_3}{\partial x}\right)\hat{j}
\left( \frac{\partial f_1}{\partial z} -  \frac{\partial f_2}{\partial y}\right)\hat{k}
\end{aligned}
$$

We can write this as

$$
\text{curl}(\vec{\boldsymbol{f}}): \mathbb{R}^D \rightarrow \mathbb{R}^D
$$

We can also write the functional transformation version

$$
\text{curl}[\vec{\boldsymbol{f}}](\vec{\mathbf{x}}): \mathbb{R}^D \rightarrow \mathbb{R}^D
$$


````{admonition} Resources
:class: dropdown seealso

* Explanation using Vorticity - [Part 1](https://youtu.be/M55LbJr_SsQ) | [Part 2](https://youtu.be/lqDx2uiKx_Q)



````


## Laplacian

> The second derivative


$$
\begin{aligned}
\text{Laplacian }
&:= \Delta u \\
&= \nabla^2 u \\
&= \text{div}(\nabla u) \\
&= \partial_{xx}u + \partial_{yy}u + \partial_{zz}u
\end{aligned}
$$

We can also write this as the functional transformation version

$$
\text{Laplacian}[\boldsymbol{f}](\vec{mathbf{x}}): \mathbf{R}^D \rightarrow \mathbf{R}
$$

````{admonition} Pseudo-Code
:class: dropdown info

```python
u: Array["P D"] = ...

# from scratch (partial derivatives)
d2u_dx2: Array["P"] = derivative(u, step_size=dx, axis=0, order=2, accuracy=4)
d2u_dy2: Array["P"] = derivative(u, step_size=dy, axis=1, order=2, accuracy=4)
u_lap: Array["P"] = d2u_dx2 + d2u_dy2

# from scratch (divergence)
u_grad: Array["P D"] = gradient(u, step_size=(dx,dy), order=1, accuracy=4)
u_lap: Array["P"] = divergence(u, step_size=(dx,dy), accuracy=4)

# laplacian operator
u_lap: Array["P"] = laplacian(u, step_size=(dx,dy), accuracy=4)
```

````


## Material Derivative



### Scalar Field

Given a scalar field:

$$
\phi:=\boldsymbol{\phi}(\vec{\mathbf{x}},t)=\boldsymbol{\phi}(x,y,z,t) \hspace{10mm} \phi:\mathbb{R}^3\times\mathbb{R}\rightarrow \mathbb{R}
$$

We can write the Material derivative as

$$
\frac{D\phi}{Dt} := \frac{\partial \phi}{\partial t} + \vec{\mathbf{u}} \cdot \nabla \phi
$$

where

$$
\vec{\mathbf{u}} \cdot \nabla \phi =
u_1\frac{\partial \phi}{\partial x} +
u_2\frac{\partial \phi}{\partial y} +
u_3\frac{\partial \phi}{\partial z}
$$

---
### Vector Field

Given a vector valued field:

$$
\vec{\boldsymbol{F}}:=\vec{\boldsymbol{F}}(\vec{\mathbf{x}},t)=
\vec{\boldsymbol{F}}(x,y,z,t) \hspace{10mm}
\vec{\boldsymbol{F}}:\mathbb{R}^3\times\mathbb{R}\rightarrow \mathbb{R}^{3}
$$

We can write the Material derivative as

$$
\frac{D \vec{\boldsymbol{F}}}{Dt} := \frac{\partial \vec{\boldsymbol{F}}}{\partial t} + \vec{\mathbf{u}} \cdot \nabla \vec{\boldsymbol{F}}
$$

where

$$
\vec{\mathbf{u}} \cdot \nabla \vec{\boldsymbol{F}} =
u_1\frac{\partial \vec{\boldsymbol{F}}}{\partial x} +
u_2\frac{\partial \vec{\boldsymbol{F}}}{\partial y} +
u_3\frac{\partial \vec{\boldsymbol{F}}}{\partial z}
$$

````{admonition} Resources
:class: dropdown seealso

* [Good ol' Wikipedia Page](https://en.wikipedia.org/wiki/Material_derivative)
* Fluid Mechanics Lecture - [Video](https://youtu.be/p7qFS9umcx8)
* Atmospheric Dynamics Lecture - [Video](https://youtu.be/TPmkx_hfuAc)


````


---
## Determinant Jacobian

From a differential operator perspective, we have

$$
\begin{aligned}
\det\boldsymbol{J}(A,B) &= -\det\boldsymbol{J}(B,A)\\
&= \mathbf{k}\cdot\left(\nabla A\times \nabla B\right) \\
&= - \mathbf{k}\cdot \nabla\times\left(A\nabla B\right) \\
&= - \mathbf{k}\cdot \nabla\times\left(B\nabla A\right) \\
&= -\mathbf{k}\text{ curl}\left(B\nabla A\right)
\end{aligned}
$$

If we think of Cartesian coordinates, we have

$$
\begin{aligned}
\det \boldsymbol{J}(A,B) &= \frac{\partial A}{\partial x}\frac{\partial B}{\partial y} -\frac{\partial A}{\partial y}\frac{\partial B}{\partial x} \\
&= \frac{\partial }{\partial x}\left(A\frac{\partial B}{\partial y}\right) -\frac{\partial }{\partial y}\left(A\frac{\partial B}{\partial x}\right) \\
&= \frac{\partial }{\partial y}\left(B\frac{\partial A}{\partial x}\right) -\frac{\partial }{\partial x}\left(B\frac{\partial A}{\partial y}\right) \\
\end{aligned}
$$

We can write this transformation as

$$
\det\boldsymbol{J}(\boldsymbol{f}, \boldsymbol{g}): \mathbf{R}^D\times\mathbb{R}^{D} \rightarrow \mathbf{R}^{D}
$$

We can also write this as the functional transformation version

$$
\det\boldsymbol{J}[\boldsymbol{f}, \boldsymbol{g}](\vec{\mathbf{x}}): \mathbf{R}^D \rightarrow \mathbf{R}
$$

````{admonition} Pseudo-Code
:class: dropdown info

```python
u: Array["P D"] = ...
v: Array["P D"] = ...

# det Jacobian operator
step_size = ((dx,dy),(dx,dy))
u_detj: Array[""] = det_jacobian(u, v, step_size, accuracy)

# from scratch (partial derivatives)
du_dx: Array["P"] = derivative(u, step_size=dx, axis=0, order=1, accuracy=2)
du_dy: Array["P"] = derivative(u, step_size=dy, axis=1, order=1, accuracy=2)
dv_dx: Array["P"] = derivative(v, step_size=dx, axis=0, order=1, accuracy=2)
dv_dy: Array["P"] = derivative(v, step_size=dy, axis=1, order=1, accuracy=2)
u_detj: Array["P"] = du_dx * dv_dy - du_dy * dv_dx

# from scratch (partial derivatives + divergence)
du_dx: Array["P"] = derivative(u, step_size=dx, axis=0, order=1, accuracy=2)
du_dy: Array["P"] = derivative(u, step_size=dy, axis=1, order=1, accuracy=2)
vdu_dx: Array["P D"] = v * du_dx
vdu_dy: Array["P D"] = v * du_dy

# from scratch (gradient + divergence)
u_grad: Array["P D"] = gradient(u, step_size=(dx,dy), order=1, accuracy=2)
vu_grad: Array["P D"] = v * u_grad
u_detj: Array["P"] = curl(vu_grad, step_size=(dx,dy), accuracy=2)
```

````


````{admonition} Resources
:class: seealso dropdown

* Explanation of Jacobian from Flows Perspective - [Slides](https://drive.google.com/file/d/10LqMqk9gT97avcPJqRHA2F0CNSlTpQLx/view)
* Serpentine Integral Visualization of the Change of Variables and the Jacobian - [Video](https://www.youtube.com/watch?v=hhFzJvaY__U&t=515s)
* Khan Academy Explanation of Determinant - [Video](https://www.youtube.com/watch?v=wCZ1VEmVjVo)
* 3Blue3Brown Visualization of Determinant Jacobian - [Video](https://www.youtube.com/watch?v=Ip3X9LOh2dk)

````


---
## Helmholtz Equation


$$
\begin{aligned}
\nabla \boldsymbol{f}(\vec{\mathbf{x}}) - k^2 \boldsymbol{f}(\vec{\mathbf{x}}) &= 0 \\
\left( \nabla  - k^2 \right)\boldsymbol{f}(\vec{\mathbf{x}}) &= 0
\end{aligned}
$$


### Helmholtz Decomposition

$$
\vec{\boldsymbol{f}} = \underbrace{- \nabla\phi}_{\text{Div-Free}}+ \underbrace{\nabla\times \mathbf{A}}_{\text{Curl-Free}}
$$
