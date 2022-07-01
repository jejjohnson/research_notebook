# Mutual Information and Total Correlation

---
This is the reduction of uncertainty of one random variable due to the knowledge of another (like the definition above). It is the amount of information one r.v. contains about another r.v..


**Definition**: The mutual information (MI) between two discreet r.v.s $X,Y$ jointly distributed according to $p(x,y)$ is given by:

$$I(X;Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$$

$$I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)$$

$$I(X;Y) = H(X) + H(Y) - H(X,Y)$$

**Sources**:
* [Scholarpedia](http://www.scholarpedia.org/article/Mutual_information)


<details>
<summary>Code</summary>

1. We need a PDF estimation...


2. Normalize counts to probability values

```python
pxy = bin_counts / float(np.sum(bin_counts))
```

3. Get the marginal distributions

```python
px = np.sum(pxy, axis=1) # marginal for x over y
py = np.sum(pxy, axis=0) # marginal for y over x
```

4. Joint Probability
</details>

---
### Total Correlation

This isn't really talked about outside of the ML community but I think this is a useful measure to have; especially when dealing with multi-dimensional and multi-variate datesets. 

In general, the formula for Total Correlation (TC) between two random variables is as follows:

$$TC(X,Y) = H(X) + H(Y) - H(X,Y)$$

**Note**: This is the same as the equation for mutual information between two random variables, $I(X;Y)=H(X)+H(Y)-H(X,Y)$. This makes sense because for a Venn Diagram between two r.v.s will only have one part that intersects. This is different for the multivariate case where the number of r.v.s is greater than 2.

Let's have $D$ random variables for $X = \{ X_1, X_2, \ldots, X_D\}$. The TC is:

$$TC(X) = \sum_{d=1}^{D}H(X_d) - H(X_1, X_2, \ldots, X_D)$$

In this case, $D$ can be a feature for $X$.

Now, let's say we would like to get the **difference in total correlation** between two random variables, $\Delta$TC.

$$\Delta\text{TC}(X,Y) =  \text{TC}(X) - \text{TC}(Y)$$

$$\Delta\text{TC}(X,Y) =  \sum_{d=1}^{D}H(X_d) - \sum_{d=1}^{D} H(Y_d) - H(X) + H(Y)$$

**Note**: There is a special case in [RBIG](https://github.com/jejjohnson/rbig) where the two random variables are simply rotations of one another. So each feature will have a difference in entropy but the total overall dataset will not. So our function would be reduced to: $\Delta\text{TC}(X,Y) =  \sum_{d=1}^{D}H(X_d) - \sum_{d=1}^{D} H(Y_d)$ which is overall much easier to solve.


---
### Higher Order

This is a term that measures the statistical dependency of multi-variate sources using the common mutual-information measure.

$$
\begin{aligned}
I(\mathbf{x})
&= 
D_\text{KL} \left[ p(\mathbf{x}) || \prod_d p(\mathbf{x}_d) \right] \\
&= \sum_{d=1}^D H(x_d) - H(\mathbf{x})
\end{aligned}
$$

where $H(\mathbf{x})$ is the differential entropy of $\mathbf{x}$ and $H(x_d)$ represents the differential entropy of the $d^\text{th}$ component of $\mathbf{x}$. This is nicely summaries in equation 1 from ([Lyu & Simoncelli, 2008][1]).

?> Note: We find that $I$ in 2 dimensions is the same as mutual information.

We can decompose this measure into two parts representing second order and higher-order dependencies:

$$
\begin{aligned}
I(\mathbf{x}) &=
\underbrace{\sum_{d=1}^D \log{\Sigma_{dd}} - \log{|\Sigma|}}_{\text{2nd Order Dependencies}} \\
&- \underbrace{D_\text{KL} \left[ p(\mathbf{x}) || \mathcal{G}_\theta (\mathbf{x}) \right] - \sum_{d=1}^D D_\text{KL} \left[ p(x_d) || \mathcal{G}_\theta (x_d) \right]}_{\text{high-order dependencies}}
\end{aligned}
$$

again, nicely summarized with equation 2 from ([Lyu & Simoncelli, 2008][1]).

**Sources**:
* Nonlinear Extraction of "Independent Components" of elliptically symmetric densities using radial Gaussianization - Lyu & Simoncelli - [PDF](https://www.cns.nyu.edu/pub/lcv/lyu08a.pdf)


---
## Normalized Variants



---
## References


* [MI w. Numpy](https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy)
* [Predictions and Correlations in Complex Data](https://www.freecodecamp.org/news/how-machines-make-predictions-finding-correlations-in-complex-data-dfd9f0d87889/)