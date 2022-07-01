# Formulation

> *Distribution flows through a sequence of invertible transformations* - Rezende & Mohamed (2015)


We want to fit a density model $p_\theta(x)$ with continuous data $x \in \mathbb{R}^N$. Ideally, we want this model to:

* **Modeling**: Find the underlying distribution for the training data.
* **Probability**: For a new $x' \sim \mathcal{X}$, we want to be able to evaluate $p_\theta(x')$
* **Sampling**: We also want to be able to generate samples from $p_\theta(x')$.
* **Latent Representation**: Ideally we want this representation to be meaningful.

---

Change-of-variables formula lets us compute the density over x:


$$
p_\theta(\mathbf{x}) = p(f_\theta(\mathbf{x})) \left| \frac{\partial f_\theta(\mathbf{x})}{\partial \mathbf{x}} \right|
$$

Train with maximum likelihood:


$$
\arg\min_\theta \mathbb{E}_\mathbf{x} \left[ -\log p_\theta(\mathbf{x}) \right] = \mathbb{E}_\mathbf{x} \left[ -\log p(f_\theta(\mathbf{x})) - \log \mathrm{det} \left| \frac{\partial f_\theta(\mathbf{x})}{\partial \mathbf{x}} \right| \right]
$$

**Note**: Maximum likelihood objective $\text{KL}(\text{data} || f^{-1}(z))$ is equivalent to $\text{KL}(f(\text{data}) || z)$ -- i.e. training by maximum likelihood tries to make the latents match the prior. This makes sense: if this happens, then samples will be good.

**New key requirement**: the Jacobian determinant must be easy to calculate and differentiate!

---


Let's assume that we can find some probability distribution for $\mathcal{X}$ but it's very difficult to do. So, instead of $p_\theta(x)$, we want to find some parameterized function $f_\theta(x)$ that we can learn.

$$x = f_\theta(x)$$

We'll define this as $z=f_\theta(x)$. So we also want $z$ to have certain properties. 

1. We want this $z$ to be defined by a probabilistic function and have a valid distribution $z \sim p_\mathcal{Z}(z)$
2. We also would prefer this distribution to be simply. We typically pick a normal distribution, $z \sim \mathcal{N}(0,1)$





 We begin with in initial distribution and then we apply a sequence of $L$ invertible transformations in hopes that we obtain something that is more expressive. This originally came from the context of Variational AutoEncoders (VAE) where the posterior was approximated by a neural network. The authors wanted to 

$$
\begin{aligned}
\mathbf{z}_L = f_L \circ f_{L-1} \circ \ldots \circ f_2 \circ f_1 (\mathbf{z}_0)
\end{aligned}
$$



### Loss Function

We can do a simple maximum-likelihood of our distribution $p_\theta(x)$. 

$$\underset{\theta}{\text{max}} \sum_i \log p_\theta(x^{(i)})$$

However, this expression needs to be transformed in terms of the invertible functions $f_\theta(x)$. This is where we exploit the rule for the change of variables. From here, we can come up with an expression for the likelihood by simply calculating the maximum likelihood of the initial distribution $\mathbf{z}_0$ given the transformations $f_L$. 



$$
\begin{aligned}
p_\theta(x) = p_\mathcal{Z}(f_\theta(x)) \left| \frac{\partial f_\theta(x)}{\partial x} \right|
\end{aligned}
$$

So now, we can do the same maximization function but with our change of variables formulation:

$$
\begin{aligned}
\underset{\theta}{\text{max}} \sum_i \log p_\theta(x^{(i)}) &= 
\underset{\theta}{\text{max}} \sum_i \log p_\mathcal{Z}\left(f_\theta(x^{(i)})\right) +
\log \left| \frac{\partial f_\theta (x^{(i)})}{\partial x} \right|
\end{aligned}
$$

And we can optimize this using stochastic gradient descent (SGD) which means we can use all of the autogradient and deep learning libraries available to make this procedure relatively painless.

#### Stochastic Gradients

$$
\nabla_\theta \mathbb{E}_{x\sim p_{\text{data}}(x)}\left[ \log p_\theta(x) \right] =
\mathbb{E}_{x\sim p_{\text{data}}(x)} \left[ \nabla_\theta \log p_\theta(x) \right]
$$


#### Negative Log-Likelihood

$$
\log p_\theta (x) = \log p(f(x)) + \log \left| \det \frac{\partial f_\theta(x)}{\partial x} \right|
$$

$$
-\mathbb{E}_\mathbf{x}\left[  \log p_\theta(x)\right] = - \mathbb{E}_x \left[ \log p_z(\mathcal{G}_\theta(x)) + \log |\nabla_x \mathcal{G}_\theta (x)| \right]
$$

Empirically, this can be calculated by:

$$
-\mathbb{E}_\mathbf{x}\left[  \log p_\theta(x)\right] =
-\frac{1}{N} \sum_{i=1}^N \log p_z(\mathcal{G}_\theta(x_i)) -
\frac{1}{N} \sum_{i=1}^N \log |\nabla_x \mathcal{G}_\theta (x_i)|
$$

---
##### Non-Gaussianity

Another perspective is the "Non-Gaussianity" of your data.

$$
\text{D}_\text{KL}\left[p(f_\theta(\mathbf{x})) || \mathcal{N}(\mathbf{0}, \mathbf{1})  \right]
$$

$$J(p_y) = \mathbb{E}_x \left[  \log p_x(x) - \log \left| \nabla_x \mathcal{G}_\theta(x)  \right| - \log \mathcal{N}\left(\mathcal{G}_\theta(x)\right)\right]
$$

If we assume that the probability of $p_x(x)=c$ because it will never change, it means that the only thing we have to do is minimize the 2nd and 3rd terms.

$$
\begin{aligned}
J(p_y) &= - \mathbb{E}_x \left[  \log \left| \nabla_x \mathcal{G}_\theta(x)  \right| \right] -
\mathbb{E}_x \left[  \log \mathcal{N}\left(\mathcal{G}_\theta(x)\right) \right] \\
\end{aligned}
$$

which we can find empirically:

$$J(p_y) = 
\sum_{i=1}^N \log \left| \nabla_x \mathcal{G}_\theta(x)  \right| -
\sum_{i=1}^N \log \mathcal{N}\left(\mathcal{G}_\theta(x_i)\right)
$$

>! **Question**: What's the difference between the two equations? Perhaps part 1, you fit a Gaussian...


### Sampling

If we want to sample from our base distribution $z$, then we just need to use the inverse of our function. 

$$x = f_\theta^{-1}(z)$$

where $z \sim p_\mathcal{Z}(z)$. Remember, our $f_\theta(\cdot)$ is invertible and differentiable so this should be no problem.


---


$$
\begin{aligned}
q(z') = q(z) \left| \frac{\partial f}{\partial z} \right|^{-1}
\end{aligned}
$$

or the same but only in terms of the original distribution $\mathcal{X}$


We can make this transformation a bit easier to handle empirically by calculating the Log-Transformation of this expression. This removes the inverse and introduces a summation of each of the transformations individually which gives us many computational advantages.

$$
\begin{aligned}
\log q_L (\mathbf{z}_L) = \log q_0 (\mathbf{z}_0) - \sum_{l=1}^L \log \left| \frac{\partial f_l}{\partial \mathbf{z}_l} \right|
\end{aligned}
$$

So now, our original expression with $p_\theta(x)$ can be written in terms of $z$.



TODO: Diagram with plots of the Normalizing Flow distributions which show the direction for the idea.

In order to train this, we need to take expectations of the transformations.

$$
\begin{aligned}
\mathcal{L}(\theta) &= 
\mathbb{E}_{q_0(\mathbf{z}_0)} \left[ \log p(\mathbf{x,z}_L)\right] -
\mathbb{E}_{q_0(\mathbf{z}_0)} \left[ \log q_0(\mathbf{z}_0) \right] -
\mathbb{E}_{q_0(\mathbf{z}_0)} 
\left[ \sum_{l=1}^L \log \text{det}\left| \frac{\partial f_l}{\partial \mathbf{z}_k} \right| \right]
\end{aligned}
$$


---

## Choice of Transformations

$$
\underbrace{\log p(\mathbf{x})}_{\text{Model Distribution}} =
\underbrace{\log p\left(\overbrace{f_{\theta}}^{\color{green}{\text{arbitrary}}}(\mathbf{x})\right)}_{\text{Base Distribution}} +
\log \; \underbrace{\left| \det \frac{\partial f_\theta(\mathbf{x})}{\partial \mathbf{x}} \right|}_{\color{red}{\text{Bottleneck}}}
$$ 

The main thing that many of the communities have been looking into is how one chooses the aspects of the normalizing flow: the prior distribution and the Jacobian. 

<details>
<summary>Algorithm Steps</summary>

**Step 1**: Obtain an invertible architecture.

**Step 2**: Perform an efficient computation of a change of variables formula.

</details>

### Prior Distribution

This is very consistent across the literature: most people use a fully-factorized Gaussian distribution. Very simple.

### Arbitrary Function, $f(\cdot)$

One main challenge in neural density estimation is to design the transformations $f(·)$ such that evaluation of the density can be done exactly. In addition, we want a transformation f(·) that is universal i.e. it can approximate any density function arbitrarily well. This is lacking in the literature as the only real proof-based conclusion of universal approximation is in {cite}`LaparraRBIG,HuangNAF,MengGaussFlow`.

### Jacobian

```{figure} ./pics/nfs_jacobian.png
---
height: 150px
name: directive-fig
---
Examples of Jacobian structures ([Source](http://www.cs.toronto.edu/~rtqichen/posters/residual_flows_poster.pdf))
```

This is the area of the most research within the community. There are many different complicated frameworks but almost all of them can be put into different categories for how the Jacobian is constructed.

#### Diagonal

These Jacobian matrices incorporate the least structure as every transformation is only applied to the dataset feature-wise. The most recent is the Gaussianization Flow {cite}`MengGaussFlow`. This will be the least expressive transformation but it will be the cheapest and simplest to compute because the determinant of a diagonal matrix is the sum of its diagonal entries. While it appears to be the least expressive, the results from {cite}`MengGaussFlow` are quite competitive.

#### Identities

These are Jacobian matrices whose structure is determined by the transformation. Often these result in low-rank matrices or orthogonal matrices. Some examples in the literature include planar flows {cite}`rezende15` which do an affine transformation and sylvester flows {cite}`berg2019sylvester` which do an orthogonal transformation via householder transforms.

#### Coupling Blocks

These are by far the most popular forms of normalizing flows. It works by partitioning the transformations such that they are only applied on a subset of dimensions. This results in a structured triangular Jacobian with a block sparse-like structure. Some noteable examples include the NICE algorithm {cite}`DinhNICE` and its successor RealNVP {cite}`DinhRealNVP`. It also includes one of the most popular and SOTA method GLOW {cite}`KingmaGLOW` which features 1x1 Convolutional blocks.


#### Autoregressive

Another very popular class of models which feature more general neural network architectures are autoregressive functions (AFs). These are typically more used for density estimation and not sampling because it is very expensive for these methods to compute samples as it needs to do 1 dimension at a time. Some noteable examples include the Invertible AF (IAF) {cite}`KingmaIAF`, the Neural AF (NAF) {cite}`HuangNAF`, and the Masked AF (MAF) {cite}`PapamakariosMAF`.


#### Free-Form

The final class of methods features free-form transformations. There is no restriction and thereby is the most expressive transformation you'll find. Some of the SOTA at the moment feature continuous-time transformations called FFJORD {cite}`GrathwohlFFJORD` and residual flows {cite}`ChenResFlow`. These methods tend to be more expensive and a lot more complicated to implement. But of course the trade-off is that you'll need a lot less layers to effectively learn the PDF of a difficult dataset. {cite}`inouye2020weakflow_workshop`

---
## Connections to Other Generative Models


```{figure} ./pics/nfs_others.png
---
height: 350px
name: directive-fig
---
Examples of other generative models ([Source](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html#types-of-generative-models))
```

---

## References

```{bibliography} ../../bibs/appendix/pdf_est/nde.bib
```


---
<details>

## Resources

#### Best Tutorials

* [Flow-Based Deep Generative Models](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html) - Lilian Weng
  > An excellent blog post for Normalizing Flows. Probably the most thorough introduction available.
* [Flow Models](https://docs.google.com/presentation/d/1WqEy-b8x-PhvXB_IeA6EoOfSTuhfgUYDVXlYP8Jh_n0/edit#slide=id.g7d4f9f0446_0_43) - [Deep Unsupervised Learning Class](https://sites.google.com/view/berkeley-cs294-158-sp20/home), Spring 2010 
* [Normalizing Flows: A Tutorial](https://docs.google.com/presentation/d/1wHJz9Awhlp-PWLZGWJKzF66gzvqdSrhknb-iLFJ1Owo/edit#slide=id.p) - Eric Jang



## Survey of Literature

---

### Neural Density Estimators

### Deep Density Destructors

## Code Tutorials

* Building Prob Dist with TF Probability Bijector API - [Blog](https://tiao.io/post/building-probability-distributions-with-tensorflow-probability-bijector-api/)
* https://www.ritchievink.com/blog/2019/10/11/sculpting-distributions-with-normalizing-flows/





### Tutorials

* RealNVP - [code I](https://github.com/bayesgroup/deepbayes-2019/blob/master/seminars/day3/nf/nf-solution.ipynb)
* [Normalizing Flows: Intro and Ideas](https://arxiv.org/pdf/1908.09257.pdf) - Kobyev et. al. (2019)


### Algorithms

*


### RBIG Upgrades

* Modularization
  * [Lucastheis](https://github.com/lucastheis/mixtures)
  * [Destructive-Deep-Learning](https://github.com/davidinouye/destructive-deep-learning/tree/master)
* TensorFlow
  * [NormalCDF](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/normal_cdf.py)
  * [interp_regular_1d_grid](https://www.tensorflow.org/probability/api_docs/python/tfp/math/interp_regular_1d_grid)
  * [IT w. TF](https://nbviewer.jupyter.org/github/adhiraiyan/DeepLearningWithTF2.0/blob/master/notebooks/03.00-Probability-and-Information-Theory.ipynb)


### Cutting Edge

* Neural Spline Flows - [Github](https://github.com/bayesiains/nsf)
  * **Complete** | PyTorch
* PointFlow: 3D Point Cloud Generations with Continuous Normalizing Flows - [Project](https://www.guandaoyang.com/PointFlow/)
  * PyTorch
* [Conditional Density Estimation with Bayesian Normalising Flows](https://arxiv.org/abs/1802.04908) | [Code](https://github.com/blt2114/CDE_with_BNF)

### Github Implementations

* [Bayesian and ML Implementation of the Normalizing Flow Network (NFN)](https://github.com/siboehm/NormalizingFlowNetwork)| [Paper](https://arxiv.org/abs/1907.08982)
* [NFs](https://github.com/ktisha/normalizing-flows)| [Prezi](https://github.com/ktisha/normalizing-flows/blob/master/presentation/presentation.pdf)
* [Normalizing Flows Building Blocks](https://github.com/colobas/normalizing-flows)
* [Neural Spline Flow, RealNVP, Autoregressive Flow, 1x1Conv in PyTorch](https://github.com/tonyduan/normalizing-flows)
* [Clean Refactor of Eric Jang w. TF Bijectors](https://github.com/breadbread1984/FlowBasedGenerativeModel)
* [Density Estimation and Anomaly Detection with Normalizing Flows](https://github.com/rom1mouret/anoflows)

</details>


