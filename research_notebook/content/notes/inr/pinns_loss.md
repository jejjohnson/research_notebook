# Physics-Informed Loss


## QG Model


**Assumptions**

$$
\text{Pressure} \propto \text{Sea Surface Height (SSH)}
$$

$$
\text{Coriolis Force} \propto \text{Velocity}
$$


**Equations**

$$
\frac{\partial \mathbf{q}}{\partial t} + \det \mathbf{J}(\boldsymbol{\psi},\boldsymbol{q}) = 0
$$

**Stream Function**

$$
\boldsymbol{\psi} = \frac{g}{f}SSH
$$

where $g$ is the gravity constant and $f$ is the coriolis parameter.

**Potential Vorticity**

This function will implement the conservation of potential vorticity.

$$
\boldsymbol{q} = \boldsymbol{\nabla}^2 \boldsymbol{\psi} - \frac{1}{LR^2}\boldsymbol{\psi}
$$

where $\boldsymbol{\nabla}^2$ is the Laplace operator, $LR^2$ is the baroclinic Rossby radius of deformation,$\boldsymbol{\nabla}^2 \boldsymbol{\psi}$ is the relative vorticity and $\frac{1}{LR^2}\boldsymbol{\psi}$ is the vortex streatching.


**Potential Vorticity**: the vorticity of a fluid as viewed in the rotating frame of Earth

**Vortex Stretching** - describes the mechanisms of stretching of the fluid column.

So we can plug in the definition of the stream function within the potential vorticity term.

$$
\boldsymbol{q} = \boldsymbol{\nabla}^2 \left(\frac{g}{f}SSH\right) - \frac{1}{LR^2}\left(\frac{g}{f}SSH\right) 
$$


---
## Jacobian

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
$$
\det \mathbf{J} = AD - BC
$$