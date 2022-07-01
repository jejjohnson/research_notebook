# Rotation-Based Iterative Gaussianization (RBIG)

> Gaussianization - Given a random variance $\mathbf x \in \mathbb R^d$, a Gaussianization transform is an invertible and differentiable transform $\mathcal \Psi(\mathbf)$ s.t. $\mathcal \Psi( \mathbf x) \sim \mathcal N(0, \mathbf I)$.

$$\mathcal G:\mathbf x^{(k+1)}=\mathbf R_{(k)}\cdot \mathbf \Psi_{(k)}\left( \mathbf x^{(k)} \right)$$

where:
* $\mathbf \Psi_{(k)}$ is the marginal Gaussianization of each dimension of $\mathbf x_{(k)}$ for the corresponding iteration.
* $\mathbf R_{(k)}$ is the rotation matrix for the marginally Gaussianized variable $\mathbf \Psi_{(k)}\left( \mathbf x_{(k)} \right)$


---
## Iterative: Pros and Cons


#### Pro: Simple

This method is relatively simple to implement. We cannot underestimate the simplicity making it very attractive to use for experts and non-experts alike. It's iterative, so you won't have to deal with too many issues related to neural networks and gradients such as initialization, learning rate, exploding gradients. The new worry is more about convergence and how does one effectively determine the best stopping criteria.


#### Pro: Backwards Compatibility

You can use any plug-in-play estimator of your choice available from a wide range of methods in the literature. For the side of the marginal density estimator, we have methods that are piecewise such as histogram, smooth like kernel density estimation and adaptive like k-Nearest Neighbours. At the end of the day, we're very good at estimating 1D densities. One thing to note is that all of the problems are inherited while doing this such as boundaries and parameters. For the side of the random orthogonal rotation, we have methods like ICA and PCA. And thanks to Laparra et al (2011), we can use any estimator.


#### Pro: Guaranteed Convergence

There are some theoretical guarantees that given enough successive transformations, the resulting distribution will be Gaussian; curtesy of {cite}`ChenGauss` for ICA and {cite}`LaparraRBIG` for any random rotation. This means that you are guaranteed a solution and that it won't be a question of non-convex optimization. Not many methods in the normalizing flow literature can offer such strong guarantees.

#### Con: Inefficient

This is the biggest weakness of the iterative approaches: they're not efficient. Inevitably you will need a lot of layers even for simple transformations because it is such a basic method. It means that it will require a lot of memory when there are a lot of layers especially with very high dimensional data. A key factor is there is no batch processing since these methods don't have any sort of stochastic approximations or gradient-based training. So with datasets as large as 1 million points with 1K+ dimensions, iterative techniques will suffer. This also prevents these methods from using modern hardware such as GPUs do to memory constraints.

#### Con: Same Problems as 1D Density Estimators

Just like there are issues with 1D density estimators, the same problems exist on a large scale in Gaussianization schemes. Boundaries at 1D PDF functions which affect outliers are still relevant, except there are many many more of them. Each 1D PDF estimator will have parameters and these are still present in the Gaussianization methods as well. And since they are not optimized for the dataset, they can affect many parts of your algorithm, e.g. the quality of the density estimation, the quality of your generated samples, and the values of the information theory metrics. You'll find that this algorithm can be sensitive to the decision you do make.

#### Con: Error Accumulation

This very related to the above issues but it becomes very apparent with iterative techniques: any errors or problems accumulate with each iteration. As there is no global optimization strategy to correct any weights or parameters, the errors can accumulate quickly. In addition, if you use a stopping criteria that doesn't have anything to do with the optimization, then that can lead to even more issues; again with specific parts of the method e.g. the sampling, the density or the metrics.


---
## References
```{bibliography} ../../bibs/appendix/pdf_est/rbig.bib
```