# Linear Algebra Tricks




## Identities


### Woodbury Identity

Given some matrices $\mathbf{A} \in \mathbb{R}^{D \times D}, \mathbf{U} \in \mathbb{R}^{D \times d}, \mathbf{B} \in \mathbb{R}^{d \times d}, \mathbf{V} \in \mathbb{R}^{d \times D}$ where $d < D$.

$$
(A + UBV^\top)^{-1} = A^{-1} - A^{-1} U (B^{-1} + V^\top A^{-1} U)^{-1}V^\top A^{-1}
$$(woodbury)

#### Symmetric Version

This is an easier version with symmetric assumptions about the decomposition.

$$
(A + CBC^\top)^{-1} = A^{-1} - A^{-1} C (B^{-1} + C^\top A^{-1} C)^{-1}C^\top A^{-1}
$$(woodbury_sum)


### Sylvester Determinant Theorem

$$
\left|A + \sigma_y^2 \mathbf I_N \right| \approx |\mathbf \Lambda_{MM} | \left|\sigma_y^{2} \mathbf \Lambda_{MM}^{-1} + U_{NM}^{\top} \mathbf U_{NM} \right|
$$(woodbury_det)


---
## Frobenius Norm (Hilbert-Schmidt Norm)

### Intiution

The Frobenius norm is the common matrix-based norm.

---

### Formulation

$$
\begin{aligned}
||A||_F &= \sqrt{\langle A, A \rangle_F} \\
||A|| &= \sqrt{\sum_{i,j}|a_{ij}|^2} \\
&= \sqrt{\text{tr}(A^\top A)} \\
&= \sqrt{\sum_{i=1}\lambda_i^2}
\end{aligned}$$


<details>
<summary>
    <font color="red">Proof
    </font>
</summary>

Let $A=U\Sigma V^\top$ be the Singular Value Decomposition of A. Then

$$||A||_{F}^2 = ||\Sigma||_F^2 = \sum_{i=1}^r \lambda_i^2$$

If $\lambda_i^2$ are the eigenvalues of $AA^\top$ and $A^\top A$, then we can show 

$$
\begin{aligned}
||A||_F^2 &= tr(AA^\top) \\
&= tr(U\Lambda V^\top V\Lambda^\top U^\top) \\
&= tr(\Lambda \Lambda^\top U^\top U) \\
&= tr(\Lambda \Lambda^\top) \\
&= \sum_{i}\lambda_i^2
\end{aligned}
$$

</details>

---

### Code

**Eigenvalues**

```python
sigma_xy = covariance(X, Y)
eigvals = np.linalg.eigvals(sigma_xy)
f_norm = np.sum(eigvals ** 2)
```

**Trace**

```python
sigma_xy = covariance(X, Y)
f_norm = np.trace(X @ X.T) ** 2
```

**Einsum**

```python
X -= np.mean(X, axis=1)
Y -= np.mean(Y, axis=1)
f_norm = np.einsum('ij,ji->', X @ X.T)
```

**Refactor**

```python
f_norm = np.linalg.norm(X @ X.T)
```

---

## Frobenius Norm

$$||X + Y||^2_F = ||X||_F^2 + ||Y||_F^2 + 2 \langle X, Y \rangle_F$$


### Frobenius Norm (or Hilbert-Schmidt Norm) a matrix

$$
\begin{aligned}
||A|| &= \sqrt{\sum_{i,j}|a_{ij}|^2} \\
&= \sqrt{\text{tr}(A^\top A)} \\
&= \sqrt{\sum_{i=1}\lambda_i^2}
\end{aligned}$$


<!-- <details> -->
<summary>
    <font color="black">Details
    </font>
</summary>

Let $A=U\Sigma V^\top$ be the Singular Value Decomposition of A. Then

$$||A||_{F}^2 = ||\Sigma||_F^2 = \sum_{i=1}^r \lambda_i^2$$

If $\lambda_i^2$ are the eigenvalues of $AA^\top$ and $A^\top A$, then we can show 

$$
\begin{aligned}
||A||_F^2 &= tr(AA^\top) \\
&= tr(U\Lambda V^\top V\Lambda^\top U^\top) \\
&= tr(\Lambda \Lambda^\top U^\top U) \\
&= tr(\Lambda \Lambda^\top) \\
&= \sum_{i}\lambda_i^2
\end{aligned}
$$
