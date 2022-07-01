# Normalizing Flows

Let $\mathbf{Z}\in \mathbb{R}^D$ be a random variable with a tractable PDF: $p_{\mathbf{z}}: \mathbb{R}^D \rightarrow \mathbb{R}^D$. The objective is to find some invertible function $\boldsymbol{f}_{\boldsymbol \theta}$ such that $\boldsymbol{f}_{\boldsymbol \theta}(\mathbf{x})=\mathbf{z}$. This function is parameterized by $\boldsymbol{\theta}$ which allows us to learn the transformation. Using the change-of-variables formula, we can compute the density of $\mathbf{x}$:

$$
\begin{equation}
    p_{\boldsymbol \theta}(\mathbf{x}) = p(\boldsymbol{f}_\theta(\mathbf{x})) \left| \boldsymbol{\nabla}_\mathbf{x} \boldsymbol{f}_{\boldsymbol \theta}(\mathbf{x})\right|
\end{equation}
$$

where $\boldsymbol{\nabla}_\mathbf{x}$ is the Jacobian of $\boldsymbol{f}$ and $|\cdot|$ is the absolute determinant. Intuitively, this $|\cdot|$ represents the change in volume of the transformation. $\boldsymbol{f}$ is the normalizing direction as it goes from a more complicated data distribution to a simpler base distribution $p_{\mathbf{z}}$. The inverse function of $\boldsymbol{f}$, i.e. $\boldsymbol{f}^{-1}$, is the generative direction as it allows us to sample from $\mathbf{z}$ which we can propagate the samples from the latent space through the function $\boldsymbol{f}^{-1}$ to get samples in $\mathbf{x}$. 



Now the challenge is to design a function $\mathbf{f}(\cdot)$ such that we can learn the mapping from our data to the latent domain. In the case of high-dimensional data, it is nearly impossible to define a transformation expressive enough such that one transformation is enough. Analogous to standard neural networks, we can stack together multiple compositions of simpler arbitrary functions to create more expressive transformations, e.g. figure~\ref{fig:c4.4-flows}. As $\mathbf{f}$ is invertible, we can have $\mathbf{f}=\mathbf{f}_L \circ \ldots \circ \mathbf{f}_1$ which would result in a more expressive transformation. Likewise the inverse is possible $\mathbf{f}^{-1} = \mathbf{f}^{-1}_1 \circ \ldots \circ \mathbf{f}_L^{-1}$. The determinant of the Jacobian in this case is simply the product of all of the transformations $\mathbf{f}_\ell$,

$$
\begin{equation}
    \left|\boldsymbol{\nabla}_\mathbf{x} \boldsymbol{f}(\mathbf{x})\right| = \prod_{\ell=1}^L \left| \boldsymbol{\nabla}_\mathbf{x} \boldsymbol{f}_\ell(\mathbf{x}) \right|,
\end{equation}
$$

which gives us more expressivity to transform arbitrarily complex distributions to simpler distributions. Given the correctly chosen set of $\boldsymbol{f}(\cdot)$, this would be sufficient to estimate any arbitrary distribution~\citep{Bogachev2005,JainiKYB20,Meng2020}.


$$
\begin{aligned}
    \mathcal{L}(\boldsymbol{\theta}) &= \arg\min_{\boldsymbol{\theta}} \mathbb{E}_\mathbf{x} \left[ -\log p_{\boldsymbol{\theta}}(\mathbf{x}) \right] = \mathbb{E}_\mathbf{x} \left[ -\log p(\boldsymbol{f}_{\boldsymbol{\theta}}(\mathbf{x})) - \log \left| \boldsymbol{\nabla}_{\mathbf{x}} \boldsymbol{f}_{\boldsymbol{\theta}}(\mathbf{x})\right| \right].
\end{aligned}
$$

The standard procedure is to estimate this using Monte Carlo sampling using stochastic gradients. Given the need to constantly evaluate the Jacobian during training, this becomes a bottleneck of this procedure. We want a transformation $\boldsymbol{f}(·)$ that is universal i.e. it can approximate any density function arbitrarily well. Hence, the community has put a lot of effort into constructing Jacobian matrices that are easy and cheap to compute yet still expressive enough to learn the complex distribution.

---
## Jacobian Form

As alluded to in the previous section, the bottleneck during the training procedure is evaluating the determinant of the Jacobian. 

$$
\begin{equation}
\log p(\mathbf{x}) =
\log p\left(\boldsymbol{f}_{\theta}(\mathbf{x})\right)+
\log \; \underbrace{\left| \boldsymbol{\nabla}_{\mathbf{x}} \boldsymbol{f}_{\boldsymbol{\theta}}(\mathbf{x})\right| }_{\color{black}{\text{Bottleneck}}}.
\end{equation}
$$

This is an area of intensive research and a really effective way to break down each of the methods~\citep{Kobyzev2020, PapamakariosNFs}. There are many different (and complicated) frameworks but almost all of them can be put into different categories for how the Jacobian is constructed. A naive full Jacobian matrix is of order $\mathcal{O}(D^3)$. This is fine with simple datasets but this can be prohibitive with datasets with more dimensions. A diagonal Jacobian is $\mathcal{O}(D)$ to evaluate but it lacks the expressivity due to its lack of cross-dimensional considerations. There are also hybrid methods in between these extremes. %, each constructed in a different way. 
Figure~\ref{fig:c4-jacobian} gives a visual reference for the differences. Below we list and briefly highlight each of the Jacobians found in the literature. For a more in-depth breakdown, please see the NF survey literature~\citep{PapamakariosNFs, KobyzevNFs}.

### Diagonal

This is known in the NF community as \textit{element-wise} transformations. A function $f(x)$ is applied to each of the features of the dataset. This is analogous to the non-linear layer for neural networks. These Jacobian matrices incorporate the least structure as every transformation has no mixing of variables thus it cannot model correlations between dimensions (figure~\ref{fig:c4-jacobian} (a)). While it is the least expressive transformation, it is the cheapest and simplest to compute as well, mainly because the determinant of a diagonal matrix is the sum of its diagonal entries. Originally, the NF community used the invertible Leaky ReLU~\citep{NFLEAKYRELU2015} in order to incorporate this non-linearity into Flow models. More recently, there has been a lot of success in Mixture of Gaussian CDF transformations~\citep{NFFLOWPP2019} as well as spline transformations~\citep{NFSPLINE2019,NFLRSPLINE2020}. Both of these methods have shown SOTA results. The Gaussianization Flow~\citep{MengGaussFlow} utilizes the Mixture of Logistics (similar to the Mixture of Gaussians) as one of the layers and it has shown competitive results. Because these transformations cannot model correlations between dimensions, they are often coupled with other transformations which do have cross-correlation considerations.

### Low Rank

These are Jacobian matrices whose determinant can be easily computed due to some transformation or property which often result in low-rank matrices (figure~\ref{fig:c4-jacobian} (b)). Some simple examples include orthogonal transformations, e.g. PLU flows~\citep{NFORTHOAUTO2018}, QR flows~\citep{NFORTHO2019}, Expontential \& Cayley map, and Householder transformations~\citep{NFHOUSE16}. The Jacobian of these transformations typically have a determinant of exactly $\pm 1$. However, these are the least expressive transformations even when multiple transformations are composed. Thus they are not typically used alone and instead are often coupled with other transformations~\citep{NFORTHO2019,NFSYLVESTER18}. Some non-linear transformations, like planar flows~\citep{RezendeM15} and radial flows~\citep{NFTABAK2013}, utilize the matrix determinant lemma~\citep{RezendeM15} which allows for more efficient computation of the determinant of the Jacobian; often $\mathcal{O}(D)$ instead of $\mathcal{O}(D^3)$. Sylvester flows~\citep{NFSYLVESTER18} extend planar flows to allow for more expressivity by doing an additional matrix multiplications parameterized by an orthogonal transformation. One disadvantage of these low-rank transformations is that they do not have analytical inverse transformations. So many times, these non-linear affine flows are used for variational inference~\citep{RezendeM15} and not for standard generative models. 

### Lower Triangular

Another very popular class of models which feature more general neural network architectures are autoregressive functions (AFs) which are constructed by factorizing over the dimension. This results in a lower triangular structure (figure~\ref{fig:c4-jacobian} (c)) which is cheap determinant calculation $\mathcal{O}(D)$. Some noteable examples include the Invertible AF (IAF)~\citep{NFIAF2016}, the Neural AF (NAF)~\citep{NFNAF2018}, the Masked AF (MAF)~\citep{NFMAF2017}, and the Block NAF (BNAF)~\citep{NFBAF2019}. These methods are very flexible and allow the user to use arbitrary neural network architectures within the algorithm which help with expressivity. Both the forward direction $\boldsymbol{f}_\theta$ and the inverse direction $\boldsymbol{f}_\theta$ are theoretically equivalent given some conditions~\citep{NFMAF2017}, but one has to be conscious about the application because AFs are dimension sensitive. For example, for density estimation, the standard AF methods are applicable~\citep{NFMAF2017,NFNAF2018,NFBAF2019} whereas for sampling, one should use the inverse variant~\citep{NFIAF2016}.

### Structured

These are by far the most popular forms of normalizing flows because they are fairly flexible yet inexpensive to compute. They work by partitioning the transformations such that they are only applied on a subset of dimensions. This results in a structured triangular Jacobian with a block sparse-like structure (figure~\ref{fig:c4-jacobian} (d)). Because of the structure, the determinant of the Jacobian is as cost efficient as the diagonal Jacobian, $\mathcal{O}(D)$. Some notable examples include the NICE algorithm~\citep{NFNICE14} and its successor RealNVP~\citep{NFREALNVP16}. Like AFs, one can use any parameterized NN architecture for the block-regions and these transformations do allow for more feature dependencies to be capture across dimensions yet they do not increase the computational complexity.  It also includes one of the most popular methods for image GLOW~\citep{NFGLOW18}, which features 1x1 Convolutional blocks.

### Free-Form


The final class of methods features free-form transformations. There is no restriction and thereby is the most expressive transformation in the literature (figure~\ref{fig:c4-jacobian} (e)). Residual Flows (RFs)~\citep{NFRESFLOW19,NFRESFLOW219} are based on residual neural network architectures. These have a very expensive Jacobian to calculate but they use a biased stochastic estimate~\citep{NFRESFLOW19} and an unbiased stochastic estimate via a power series approximation~\citep{NFRESFLOW219}. There are also a class of continuous-time flows~\citep{NFCTF19} which are based on the neural ODEs literature~\citep{NFNODE18}. All free-form methods tend to be more expensive (even with the estimation tricks) and a lot more complicated to implement. But of course the trade-off is that you'll have more expressive Jacobians, and thus will need a lot less layers to effectively learn the probability density function of a difficult dataset.

### How do they compare?

As seen in the above section, there is an abundance of methods available for the NF literature. It is, however, very difficult to compare each of them because of the many combinations one could choose. In general, the best method it depends on the task at hand. For tabular datasets (e.g. POWER, GAS, HEPMASS, etc) in the latest review survey~\citep{KobyzevNFs}, the Neural Autoregressive Flow (NAF) algorithm~\citep{NFNAF2018} and the AF with a Spline coupling layer does the best. However, the recent Gaussianization Flows paper~\citep{MengGaussFlow} (which was not included in the survey) shows substantial improvement for their method over some of the tabular datasets. For the standard image datasets, the Flow++ model~\citep{NFFLOWPP2019} (which is a make up of convolutions and CDF Mixture Layers) performs substantially better than the other methods. Surprisingly, the free-form methods are not the best despite the fact that they are the most expressive and flexible. Instead, it appears that a combination of simpler transformations and element-wise transformations like splines or cumulative distribution function (CDF) mixtures seem to perform the best for the standard datasets.

---
## Tutorials

* [Iterative Gaussianization](./lecture_1_ig.md)

---

* [Coupling Layers](./coupling_layers.md)