# Sparse GP From Scratch


## Objective Function

So I think it is important to make note of the similarities between methods; specifically between FITC and VFE which are some staple methods one would use to scale GPs naively. Not only is it helpful for understanding the connection between all of the methods but it also helps with programming and seeing where each method differs algorithmically. Each sparse method is a method of using some set of inducing points or subset of data $\mathcal{Z}$ from the data space $\mathcal{D}$. We typically have some approximate matrix $\mathbf{Q}$ which approximates the kernel matrix $\mathbf{K}$:

$$\mathbf{Q}_{ff}=\mathbf{K}_{fu}\mathbf{K}_{uu}^{-1}\mathbf{K}_{uf}$$

Then we would use the Sherman-Morrison formula to reduce the computation cost of inverting the matrix $\mathbf{K}$. Below is the negative marginal log likelihood cost function that is minimized where we can see the each term broken down:

$$
\mathcal{L}(\theta)= \frac{N}{2}\log 2\pi + \underbrace{\frac{1}{2} \log\left| \mathbf{Q}_{ff}+\mathbf{G}\right|}_{\text{Complexity Penalty}} + \underbrace{\frac{1}{2}\mathbf{y}^{\top}(\mathbf{Q}_{ff}+\mathbf{G})^{-1}\mathbf{y}}_{\text{Data Fit}} + \underbrace{\frac{1}{2\sigma_n^2}\text{trace}(\mathbf{T})}_{\text{Trace Term}}
$$

The **data fit** term penalizes the data lying outside the covariance ellipse, the **complexity penalty** is the integral of the data fit term over all possible observations $\mathbf{y}$ which characterizes the volume of possible datasets, the **trace term** ensures the objective function is a true lower bound to the MLE of the full GP. Now, below is a table that shows the differences between each of the methods. 


| Algorithm |                          $\mathbf{G}$                           |           $\mathbf{T}$            |
| :-------: | :-------------------------------------------------------------: | :-------------------------------: |
|   FITC    | diag $(\mathbf{K}_{ff}-\mathbf{Q}_{ff}) + \sigma_n^2\mathbf{I}$ |                 0                 |
|    VFE    |                     $\sigma_n^2 \mathbf{I}$                     | $\mathbf{K}_{ff}-\mathbf{Q}_{ff}$ |
|    DTC    |                     $\sigma_n^2 \mathbf{I}$                     |                 0                 |

Another thing to keep in mind is that the FITC algorithm approximates the model whereas the VFE algorithm approximates the inference step (the posterior). So here we just a have a difference in philosophy in how one should approach this problem. Many people in the Bayesian community will [argue](https://www.prowler.io/blog/sparse-gps-approximate-the-posterior-not-the-model) for approximating the inference but I think it's important to be pragmatic about these sorts of things.


$$
\begin{aligned}
\mathbf{W} &= \mathbf{L_{uu}}^\top \mathbf{K_{uf}}^\top \\
\mathbf{Q_{ff}} &= \mathbf{K_{fu}}\mathbf{K_{uu}}^{-1} \mathbf{K_{uf}}=\mathbf{WW^\top}
\end{aligned}
$$

---


```python
Kuu : (M,M)
Luu : (M,M)
W : (M,N) = (M,M) @ (M,N)
Q : (N,N) = (N,M) @ (M,M) @ (M, N)
```

---

##### W-Term


```python
# gram matrix: (M,M)
Kuu = gram_matrix(X_u, X_u, η, ℓ)
Kuu = add_to_diagonal(Kuu, jitter)

# cross covariance matrix: (M,N)
K_uf = gram_matrix(X_u, X, η, ℓ)


# L Term: (M,M)
Luu = cholesky(Kuu, lower=True)
Luu = numpyro.deterministic("Luu", Luu)

# W matrix: (N,M)
W = solve_triangular(Luu, Kuf, lower=True).T
W = numpyro.deterministic("W", W)
```

---

##### D-Term

```python
# Diagonal
D = identity_matrix(n_samples) * noise
```

---

##### Trace Term


```python
K_ff_diag = gram_matrix(X_u, X_u, η, ℓ, diag=True)
Q_ff_diag = power(W, 2).sum(axis=1)
trace_term = (K_ff_diag - Q_ff_diag).sum() / noise
trace_term = clip(trace_term, min=0.0)
```

**Note**: the trace term should never be negative but sometimes this can happen due to numerical errors.


#### Numpyro Factor

We can simply add this to the log-likelihood by explicitly specifying a factor.

```python
numpyro.factor("trace_term", - 0.5 * trace_term)
```

