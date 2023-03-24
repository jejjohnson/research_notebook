
# Spatial Discretization


## Motivation

---
## Ordinary Differential Equations

We can often think about "the simplest thing we haven't tried yet". For spatial data, before we try to express everything with derivatives, we can think about summarizing the activity with a single parameter. This is the basis of Ordinary Differential Equations (ODEs).

$$
\begin{aligned}
\frac{d}{dt}\boldsymbol{x}(t) &= \boldsymbol{f}(\boldsymbol{x}(t), t; \boldsymbol{\theta}) && &&\boldsymbol{x}:\mathbb{R}\rightarrow\mathbb{R}^D && \boldsymbol{f}:\mathbb{R}^D\times\mathbb{R}\times\mathbb{R}^{D_\theta}\rightarrow\mathbb{R}^D
\end{aligned}
$$





---
## Partial Differential Equations

In many cases, a simple parameter representation of an entire space is impossible. In this case, we have to resort to PDEs whereby we need a spatial discretization.



---
## Difference



$$
\partial_x u = \lim_{h \rightarrow 0}\frac{u(x + h) + 2 u(x) - u(x-h)}{2h}
$$

Spatial Gradients - dealing with the derivatives


```python
u: Field = ...

du_dx = difference(u, axis, step_size, accuracy, order, method)
```

`N[u](x,t)`

- Analytical (Symbolic)
- Finite difference (Slicing, Convolutions)
- Finite Volume
- Finite Element
- Spectral

---
### Other Gradients

So there are other methods which have been defined within the physics community which describe all of the motion in space. These are:

$$
\begin{aligned}
\text{grad}(\boldsymbol{f}) &:=
\boldsymbol{\nabla}\boldsymbol{f} :
\mathbb{R}\rightarrow\mathbb{R}^D\\
\text{div}(\vec{\boldsymbol{f}}) &:=
\boldsymbol{\nabla}\cdot\vec{\boldsymbol{f}} :
\mathbb{R}^D\rightarrow\mathbb{R}\\
\text{curl}(\vec{\boldsymbol{f}}) &:=
\boldsymbol{\nabla}\times\vec{\boldsymbol{f}} :
\mathbb{R}^D\rightarrow\mathbb{R}^D
\end{aligned}
$$

And of course we can derive higher order methods from these, e.g. $\text{Lap}$ and $\det \boldsymbol{J}(\cdot,\cdot)$. See the [differential operators](../misc/diff_operators.md) page for a more in-depth walkthrough of some of the most common differential operators.
