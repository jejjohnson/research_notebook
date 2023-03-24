

## QG Formulations


### Florian

$$
\begin{aligned}
\frac{\partial q}{\partial t} + \det\boldsymbol{J}(\psi,q) &= 0 \\
\psi &= \frac{g}{f}\eta \\
q &= \nabla^2 \psi - \frac{1}{L_R^2}\psi
\end{aligned}
$$ (eq:qg_flo)

### Hugo

$$
\begin{aligned}
\frac{\partial q}{\partial t} + \det\boldsymbol{J}(\psi,q) &= \nu\nabla^2 q-\mu q -\beta \partial_x\psi + F \\
\psi &= \frac{g}{f}\eta \\
q &= \nabla^2 \psi \\
\boldsymbol{u} &= (-\partial_y\psi,\partial_x\psi)
\end{aligned}
$$ (eq:qg_hugo)

where:
* $\nu$ is the viscosity
* $\mu$ is the linear drag coefficient
* $\beta$ - rossby parameter
* $F$ - source term

### Louis

Here, we have a stacked QG model:

$$
\begin{aligned}
\mathbf{q} &= [q_1, \ldots, q_N]^\top \\
\mathbf{\psi} &= [\psi_1, \ldots, \psi_N]^\top
\end{aligned}
$$

But we have the same equations as above.


$$
\begin{aligned}
\frac{\partial \mathbf{q}}{\partial t} +
\begin{bmatrix}
\mathbf{u} \\ \mathbf{v}
\end{bmatrix}
\cdot \nabla \mathbf{q} &= 0 \\
\begin{bmatrix}
\mathbf{u} \\ \mathbf{v}
\end{bmatrix} &= \nabla^{\perp}\boldsymbol{\psi} \\
\psi &= \frac{g}{f}\eta \\
q &= \nabla^2 \psi - f_0^2\mathbf{A}\psi +\beta y
\end{aligned}
$$

$$
\mathbf{A} =
\begin{bmatrix}
\frac{1}{H_1 g_1'} & \frac{-1}{H_1 g_1'} & \ldots & \ldots & \ldots  \\
\frac{-1}{H_1 g_1'} & \frac{1}{H_1}\left(\frac{1}{g_1'} + \frac{1}{g_2'} \right) & \frac{-1}{H_1 g_1'} & \ldots & \ldots  \\
\ldots & \ldots & \ldots & \ldots & \ldots \\
\ldots & \ldots & \frac{-1}{H_{n-1} g_{n-2}'} & \frac{1}{H_{n-1}}\left(\frac{1}{g_{n-2}'} + \frac{1}{g_{n-1}'} \right) & \frac{-1}{H_{n-1} g_{n-2}'}  \\
\ldots & \ldots& \ldots & \frac{-1}{H_n g_{n-1}'} & \frac{1}{H_n g_{n-1}'}   \\
\end{bmatrix}
$$

where:
* $f_0 + \beta(y-y_0)$ - Coriolis parameter under the beta-plane approximation with the meridional axis center $y_0$
* $\nabla^{\perp}=(-\partial_y,\partial_x)$ - perpendicular gradient
* $\nabla^2 =\partial_{xx}+\partial_{yy}$ - horizontal Laplacian


## Solving QG Equations - I


Given the equations in terms of q and $\psi$.

$$
\begin{aligned}
\partial_t q + \det J(\psi, q) &= 0 \\
\psi &= \frac{g}{f}\eta \\
q &= \nabla^2 \psi - \frac{1}{L_R^2}\psi
\end{aligned}
$$

We are stepping through the $q$ term.

$$
\partial_t q = -  \det(\psi, q)
$$

**Step I**: Find $\psi$

We need to calculate $\psi$ from $q$ using the expression above.

$$
\begin{aligned}
\nabla^2 \psi - \frac{1}{L_R^2}\psi &= q \\
(\nabla^2 - \frac{1}{L_R^2})\psi &= q
\end{aligned}
$$

which involves solving a linear system of equations:

$$
\psi = (\nabla^2 - \frac{1}{L_R^2})^{-1}q
$$

**Step II**: Find the determinant Jacobian


$$
\begin{aligned}
-\det J(\psi, q) &= \left(\frac{\partial\psi}{\partial y}\frac{\partial q}{\partial x} - \frac{\partial\psi}{\partial x}\frac{\partial q}{\partial y} \right) \\
\end{aligned}
$$

**Step III**: Put Everything together and step


$$
\begin{aligned}
\partial_t q = (\nabla^2 - \frac{1}{L_R^2})^{-1}\left(\frac{\partial\psi}{\partial y}\frac{\partial q}{\partial x} - \frac{\partial\psi}{\partial x}\frac{\partial q}{\partial y} \right)
\end{aligned}
$$

## Solving QG Equations for SSH

$$
\partial_t \left(\nabla^2 - \frac{1}{L_R^2} \right)\eta + \frac{g}{f}\det J(\eta, \nabla^2\eta) = 0
$$


### Step I: Calculate Determinant Jacobian

$$
\begin{aligned}
-\alpha\beta\det J(\eta, \nabla^2\eta) &=
\alpha\beta\left(\frac{\partial\eta}{\partial y}\frac{\partial \nabla^2\eta}{\partial x} - \frac{\partial\eta}{\partial x}\frac{\partial \nabla^2\eta}{\partial y} \right) \\
\end{aligned}
$$

### Step II: Isolate $\eta$

$$
\begin{aligned}
\partial_t(\nabla^2 - \frac{1}{L_R^2})\eta &= F \\
(\nabla^2 - \frac{1}{L_R^2})\partial_t\eta &= F \\
\end{aligned}
$$

which involves solving a linear system of equations:

$$
\partial_t\eta = (\nabla^2 - \frac{1}{L_R^2})^{-1}F
$$

### Step III: Step Forward

$$
\eta^{n+1} = \eta^n + \Delta t\boldsymbol{RHS}
$$
