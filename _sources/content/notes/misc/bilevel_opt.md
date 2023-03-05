# Bi-Level Optimization


$$
\begin{aligned}
\boldsymbol{\theta}^* &= \underset{\boldsymbol{\theta}}{\text{argmin  }}  \mathcal{L}(\boldsymbol{\theta},\mathbf{x}^*(\boldsymbol{\theta})) \\
\mathbf{x}^*(\boldsymbol{\theta}) &= \underset{\mathbf{x}}{\text{argmin  }} \mathcal{U}(\mathbf{x},\boldsymbol{\theta})
\end{aligned}
$$ (eqn:learn-bilevel)

## Unrolling

We assume the solution at the end of the unrolling is the solution to the minimization problem

$$
x_k(\theta) \approx x^*(\theta)
$$

We define a loss function which takes this value

$$
\mathcal{L}_k(\theta) := f(\theta, x_k(\theta))
$$

We can define the process for unrolling. Let's choose an initial value, $x_0(\theta)$. We define the unrolling step as

$$
x_k(\theta): x_{k+1}(\theta) = x_k(\theta) - \lambda \nabla_x\mathcal{U}(\theta,x_k(\theta))
$$

Note: We can choose whatever algorithm we want for this entire step. For example, we can choose SGD or even Adam.

Similarly, we can also choose an algorithm for how we find the parameters

$$
\theta_n^*:=\theta_{n+1} = \theta_n - \mu \nabla_\theta \mathcal{L}_k(\theta_n)
$$

Note, we assume that the best parameter for the minimization problem, $\theta_k^*$ is a good choice for the parameter estimation problem, $\theta_n$.


---
## Implicit Differentiation

We focus on the argmin differentiation problem

$$
\mathbf{x}^*(\boldsymbol{\theta}) = \underset{\mathbf{x}}{\text{argmin  }} \mathcal{U}(\mathbf{x},\boldsymbol{\theta})
$$


Assumptions:
* Strongly convex in $x$
* Smooth

**Implicit Function Theorem**

This states that $x^*(\theta)$ is a unique solution of

$$
\nabla_x \mathcal{U}(\theta, x^*(\theta)) = 0
$$

**Note**: unrolling just does $\nabla_x (x_k(\theta))$. My job is to construct the solution!

Implications: This holds for all $\theta$'s!

Goal: Estimation $\nabla_\theta x^*(\theta)$.

Result: A Linear System!

$$
\partial_x \partial_\theta \mathcal{U}(\theta,x^*(\theta)) + \partial^2_x\mathcal{U}(\theta,x^*(\theta)) \nabla_\theta x^*(\theta) = 0
$$

We can simplify this

$$
B(\theta) + A(\theta) \nabla_\theta x^*(\theta) = 0
$$

---

In Theory: We can find a Jacobian by solving the linear system (in theory).

In Practice: We don't need the Hessian. We just need Hessian vector products! If we observe the term, we notice that we get

$$
A(\theta) = [D_\theta \times D_x]
$$

---

Observe: Let's look at the original loss function. Using the chain rule we get:

$$
\nabla \mathcal{L}(\theta) = \partial_\theta f(\theta, x^*(\theta)) + \nabla_\theta x^*(\theta)^\top \partial_x f(\theta, x^*(\theta)) = 0
$$

And now, let's look at the linear system we want to solve

$$
\nabla_\theta x^*(\theta) = - \left[A(\theta)\right]^{-1}B(\theta)
$$

which is awful. So plugging this back into the equation, we get:

$$
\nabla \mathcal{L}(\theta) = \partial_\theta f(\theta, x^*(\theta)) + \left(-\left[A(\theta)\right]^{-1}B(\theta)\right)^\top \partial_x f(\theta, x^*(\theta)) = 0
$$

However, looking at the sizes, we notice that

`(D_x x D_x)(???)(D_x)`

This is hard: `(D_x x D_x x ???)()`

This is easy: `(D_x x D_x)(????x D_x)`

This is simply a `vjp` -> $B(\theta)^\top$.

---

**Note**: We can also solve the linear system using gradient descent and Hessian products instead of pure hessians.

---
#### Computational Cost

The cost is almost the same in terms of computations.

| Unrolling | Implicit Differentiation |
|:---------:|:------------------------:|
| $k$-steps forward (unrolling) | $k$-steps for optimization $\nabla_\theta x^*(\theta)$ |
| $k$-steps backwards (backprop unrolling) | $k$-steps for linear system opt ($Ax-b$) |

The cost is better for memory because we don't have to do unrolling!


#### Approximate Soln or Gradient

$$
\mathcal{L}_k(\theta) \approx \nabla\mathcal{L}(\theta)
$$

Do we approximate the solution (unrolling) or approximate the gradient (Implicit Diff)

#### Warm Starts

We can do a warmstart for the linear system optimization.

$$
x^*(\theta)=-[A(\theta)]^{-1}B(\theta)
$$

by starting from an already good solution. (Is this called pre-conditioning?)


#### Strongly Convex Solution


Not really... (Michael Work says overparameterized systems converges faster!)
