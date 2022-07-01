Similarity
==========

## What is it?

When making comparisons between objects, the simplest question we can ask ourselves is how 'similar' one object is to another. It's a simple question but it's very difficult to answer. Simalirity in everyday life is somewhat easy to grasp intuitively but it's not easy to convey specific instructions to a computer. A [1](1). For example, the saying, "it's like comparing apples to oranges" is usually said when you try to tell someone that something is not comparable. But actually, we can compare apples and oranges. We can compare the shape, color, composition and even how delicious we personally think they are. So let's consider the datacube structure that was mentioned above. How would we compare two variables $z_1$ and $z_2$.

### Trends

In this representation, we are essentially doing one type or processing and then

* A parallel coordinate visualization is practical byt only certain pairwise comparisons are possible.

If we look at just the temporal component, then we could just plot the time series at different points along the globe.

> Trends often do not expose similarity in an intuitive way.

{cite}`kornblith2019similarity`

---

## Constraints

1. Invariance to Rotations
2. Invariance to Permutations
3. Invariance to Isotropic Scaling
4. Invariance to Linear Transformations
5. Invariance to Sample Size

### Transformations

> Affect the size and position of the data

* Position -> Translation, Rotation, Reflection
* Size -> Dilation (e.g. isotropic scaling)

**Translation**: slides across a plane or through space (some distance, some direction)

- Transformation that slides across a plane or through a space
- All points of a figure/dataset move the same distance and in the same direction
- "change of location", specified by direction and distance


**Rotation**: turns a figure around a point or line

* spin in the shape
* turn a figure
* reflection symmetry 

**Reflection**: transformation that flips across a line


- rotation symmetry

**Dilation**: changes the size but not the actual shape

- e.g. enlarge, reduce
- usually given as a scale factor

$$
\alpha \mathbf{X} \mathbf{R} + b
$$



### Orthogonal Transformations

### Linear Transformations

### Isotropic Scaling

### Multivariate

#### Curse of Dimensionality

---

## Classic Methods

These are some notes based off of an excellent review paper {cite}`josse2016`.


* Evaluate Similarity of the whole config of units
* Transform the data into an $N\times N$ square matrix, b) evaluate the similarity.


### Congruence Coefficient

* Unadjusted Correlation - {cite}`burt1948`
* Congruence - {cite}`harman1976,tucker1951`
* Monotonicty Coefficient - {cite}`borg1997modern`

> Sensitive to patters of similarity byt not to rotations of dilations.

### RV Coefficient

> Early instance of a natural generalization of the notion of correlation to groups of variables. Can explain everything and can be generalized to almost every other method!

* {cite}`Escoufier1973,Robert1976`
  > Measure the similarity between squared matrices (PSD); multivariate analysis techniques
* Abdi (2003), Abdi & Valentin (2007), Abdi (2007, 2009), Holmes (1989, 2007)
  > STATB; DISTATIS
  > Transform rectangular matrices into squared matrices
  > Proof Abdi 2007a, 

#### Single Sample, Multiple Features

$\mathbf{x,y}\in \mathcal{R}^D$.

$$
\rho V(\mathbf{x,y}) = \frac{\text{Tr}\left( \sum_{\mathbf{xy}} \sum_{\mathbf{yx}}\right)}{\sqrt{\text{Tr}\left( \sum_{\mathbf{xx}}^2\right)\text{Tr}\left( \sum_{\mathbf{yy}}^2\right)}}
$$

* Relation to Pearson Correlation Coefficient: $D=1, \rho V=\rho^2$
* $\rho V \in [0,1]$
* $\rho V (\mathbf{x}, a\mathbf{xr}+c)=1$

**Connections**

* {cite}`Holmes_2008`; Connections between RV and PCA, LDA, LR, etc.


#### Multiple Samples, Multiple Features

> Considers two sets of variables similar if the relative positions of the observations in one set is similar to the relative positions of another.

$\mathbf{X,Y}\in \mathcal{R}^{N \times D}$. Let's create some matrices describing the empirical covaraince: $\sum_\mathbf{XY} = \frac{1}{N-1}\mathbf{X^\top Y}\in\mathcal{R}^{D \times D}$. We can summarize this as follows:

$$
\rho_\text{RV}(\mathbf{X,Y}) = \frac{\text{Tr}(\sum_\mathbf{XY}\sum_\mathbf{YX})}{\sqrt{\text{Tr}\left(\sum_\mathbf{XX}^2\right) \text{Tr}\left(\sum_\mathbf{YY}^2\right)}}
$$

#### Sample Space

Consider matrices in the sample space: $\mathbf{K_x}=\mathbf{XX^\top}\in\mathcal{R}^{N \times N}$. By taking the Hilbert-Schmidt norm, we can measure their proximity:

$$
\langle \mathbf{K_X, K_Y}\rangle_F = \text{Tr}(\mathbf{K_X K_Y}) = \sum_i^{D_\mathbf{x}}\sum_j^{D_\mathbf{y}} \text{Cov}^2(\mathbf{X_{.i}},\mathbf{Y_{.j}})
$$

$$
\begin{aligned}
\rho_\text{RV}(\mathbf{X,Y}) 
&= \frac{\langle \mathbf{K_X}\mathbf{K_Y}\rangle_F}{||\mathbf{K_X}||_F ||\mathbf{K_Y}||_F } \\
&= \frac{\text{Tr}(\mathbf{K_X}\mathbf{K_Y})}{\sqrt{\text{Tr}\left(\mathbf{K_X}\right)^2 \text{Tr}\left(\mathbf{K_Y}\right)^2}} 
\end{aligned}
$$

#### Relation


> Comparing features is the same as comparing samples. {cite}`kornblith2019similarity` ([slides](https://icml.cc/media/Slides/icml/2019/halla(13-11-00)-13-12-15-4871-similarity_of_n.pdf))

$$
||\mathbf{X}^\top \mathbf{Y}||_F^2 = \text{Tr}\left(\mathbf{XX^\top YY^\top} \right)
$$ (eq:symmetry)

* The LHS is the sum of squared dot product similarities between the features
* The RHS is the dot product between reshaped inter-example similarity matrices

<details>
<summary>Proof</summary>

A very simple proof as we simply need to expand the following the trace term using the "cyclic property" of matrices.

$$
\begin{aligned}
\text{Tr}\left(\mathbf{XX^\top YY^\top} \right) 
&= \text{Tr}\left(\mathbf{X^\top YY^\top X} \right)\\
&= \text{Tr}\left(\mathbf{X^\top Y} \right)^2 \\
&= ||\mathbf{X}^\top \mathbf{Y}||_F^2 \\
\end{aligned}
$$ 

</details>

In the same way as the $\rho$ coefficient, we can normalize both sides of the equation to get equivalent bounded correlations:

$$
\frac{||\mathbf{X}^\top \mathbf{Y}||_F^2}{||\mathbf{X}^\top \mathbf{X}||_F||\mathbf{Y}^\top \mathbf{Y}||_F} 
= \frac{\text{Tr}\left(\mathbf{XX^\top YY^\top} \right)}{||\mathbf{X} \mathbf{X}^\top||_F||\mathbf{Y} \mathbf{Y}^\top||_F}
$$

as shown in {cite}`cortes2014,harman1976,tucker1951`

---
### Distance Correlation

* Originally proposed by: {cite}`szekely2007`.

$$
K_{ij} = - \frac{1}{2}(d_{ij}^2 - d_{i.}^2 - d_{.j}^2 + d_{..}^2)
$$

where $.$ is the mean sum.

$$
\rho_\text{RV} = \frac{\langle H\triangle_x^2H, H\triangle_y^2H  \rangle_F}{|| H\triangle_x^2H||_F ||H\triangle_y^2H ||_F}
$$

where:

$$ 
  \mathbf{H}=\mathbf{I}_N - \left(\frac{1}{N} \right)\mathbf{1}_N
$$ (centering)

1. The features are standardized to have unit variance. So we have to be careful with the preprocessing.
2. It's a unifying tool that maximizes the association coefficients under certain constraints.

Assuming Euclidean distance, we can have:

$$
\text{dCor}(\mathbf{X,Y}) = \frac{\langle H\triangle_xH, H\triangle_yH  \rangle_F}{|| H\triangle_xH||_F ||H\triangle_yH ||_F}
$$

* We use $\triangle$ instead of square matrices
* $\text{dCorr}$ can detect non-linear relationships, RV can only detect linear relationships
* If we don't square the distances, we can pick up more complex relationships
* Statistical consistency: $n \rightarrow \infty$
* $0 \leq \text{dCor}(\mathbf{X,Y}) \leq 1$
* $(\mathbf{X,Y})=0$ iff $\mathbf{X,Y}$ are independent
* $\text{dCor}(\mathbf{X}, a \mathbf{XB} + c)=1$

**Connections**


* PCA {cite}`Robert1976`
* Discriminant Analysis, {cite}`Holmes_2008,De_la_Cruz_2011`
* Canonical Analysis, {cite}`Holmes_2008,De_la_Cruz_2011`
* Multivariate Regression {cite}`Holmes_2008,De_la_Cruz_2011`

---

### Kernel Methods

* MMD -> distance between the joint distribution of two random variables & the product of their marginal distributions
* HSIC

$$
  \rho_\text{HSIC}(\mathbf{X,Y}) = \frac{1}{(N-1)^2}\text{Tr}(\tilde{\mathbf{K}}_\mathbf{x}\tilde{\mathbf{K}}_\mathbf{y})
$$ (hsic)

where: $\tilde{\mathbf{K}}=\mathbf{HKH}$, $\mathbf{H}$ is the same centering matrix as above in equation {eq}`centering`.



**Connections**:

* {cite}`De_la_Cruz_2011`: some talks, [Slides](http://www.mmds-data.org/presentations/2010/Holmes.pdf) | [Slides](https://web.stanford.edu/class/bios221/Pune/Lectures/Lecture_Day5_CCA_Multitable.pdf)
* {cite}`purdom2006multivariate`; multivariate kernel methods in the analysis of graphs
* {cite}`kta2002`; Kernel Tangent Alignment -> Normalized version HSIC (w/o centering)
* {cite}`cortes2014`; Centered Kernel Tangent Alignment -> Normalized Version of HSIC
* {cite}`Sejdinovic_2013`; Energy statistics; connections between distance correlations and kernel methods.

---
#### Normalized Variants

Just like comparing covariance versus correlation, HSIC is difficult to interpret because it is unbounded and inconsistent with respect to samples and dimensions. HSIC suffers from the curse of dimensionality because for a certain set of samples and dimensions you can get a vastly different HSIC values even though they should be more consistent. As such, normalized variants were used such as {cite}`kta2002,cortes2014` which sought to alleviate this issue. Just like correlation, this works by normalizing the HSIC metric with the corresponding 


$$
  \rho_\text{nHSIC}(\mathbf{X,Y}) = \frac{\text{Tr}(\tilde{\mathbf{K}}_\mathbf{x}\tilde{\mathbf{K}}_\mathbf{y})}{\sqrt{||\tilde{\mathbf{K}}_\mathbf{x}||_F ||\tilde{\mathbf{K}}_\mathbf{y}||_F} }
$$ (nhsic)

This particular variant is called the Centered Kernel Alignment {cite}`cortes2014` but through-out this thesis, we will just called it _normalized HSIC_ to simplify the terminology. For more information, see this [**_reference_**](./../appendix/kernels/hsic_.md) in the appendix for more details regarding HSIC.

---

#### Randomized Kernels

Randomized kernel approximations allow us to estimate kernel matrices $\mathbf{K}$ in less time by means of an approximation.

$$
\mathbf{K} \approx \hat{\mathbf{K}}= \mathbf{Z}\mathbf{Z}^\top
$$

The most important thing to take home from this is that using the relationship between the dot product of kernel matrices in the sample space versus the dot product of datasets in the feature as seen in equation {eq}`eq:symmetry`, we can potentially estimate the Frobenius norm much easily especially for very large scale datasets.

$$
\begin{aligned}
\text{Tr}\left(\tilde{\mathbf{K}}_\mathbf{x}\tilde{\mathbf{K}}_\mathbf{y} \right) &= 
\text{Tr}\left(\hat{\mathbf{K}}_\mathbf{x}\hat{\mathbf{K}}_\mathbf{y} \right)\\
&= \text{Tr}\left(\mathbf{Z}_\mathbf{x}\mathbf{Z}^\top_\mathbf{x}\mathbf{Z}_\mathbf{y}\mathbf{Z}^\top_\mathbf{y} \right)\\
&= ||\mathbf{Z}_\mathbf{x}^\top \mathbf{Z}_\mathbf{y}||_F^2
\end{aligned}
$$ (eq:symmetry)

In {cite}`Zhang2018LargescaleKM`, did an experiment to see how well approximation methods such as random fourier features (RFF) and Nyström did versus their original counterparts in independence testing. They found that for large-scale synthetic data, these methods performed rather well and encouraged uses to use these in applications in the future.

---

## Mutual Information

### Information Theory

* Mutual Information is the counterpart to using information theory methods.
* It requires an estimation step which may introduce additional uncertainties
* Extends nicely to different types of data (e.g. discrete, categorical, multivariate, multidimentional)
* Exposes non-linearities which may be difficult to see via (linear) correlations
* Kernel Approximations: Although there are some differences for different estimators, relative distances are consistent

#### A Primer

* Entropy - measure of information uncertainty of $X$
* Joint Entropy - uncertinaty of $X,Y$
* Conditional Entropy - uncertainty of $X$ given that I know $Y$
* Mutual Information - how much knowning $X$ reduces the uncertainty of $Y$
  * $I(X,Y)=$
* Normalized Mutual Information
  * $\tilde{I}(X,Y) = \frac{I(X,Y)}{\sqrt{H(X)H(Y)}}$

### Variation of Information

> A measure of distance in information theory space.

$$
\begin{aligned}
VI(X,Y) &= H(X|Y) + H(Y|X)\\
&= H(X) + H(Y) -2I(X,Y)
\end{aligned}
$$

where:

* $VI(X,Y)=0$ Iff $X$ and $Y$ are the same
  * $H(X,Y)=H(X)=H(Y)=I(X,Y)$
* $VI(X,Y) < H(X,Y)$ If $X$ and $Y$ are different but dependent
  * $H(X,Y)<H(X) + H(Y)$
* $VI(X,Y)=H(X,Y)$ if $X$ and $Y$ are independent
  * $H(X,Y)=H(X) + H(Y)$
  * $I(X,Y)=0$

* Variation of Information: {cite}`MeilaVI`
* Normalized Variants: {cite}`vivinh10a`

---
## Summary of Methods


|        Name         |     Non-Linear     | Multi-Dimensional  | Isotropic Scaling  | Orthogonal Transforms |    Coefficient     | Computational Cost |
| :-----------------: | :----------------: | :----------------: | :----------------: | :-------------------: | :----------------: | :----------------: |
|   Pearson, $\rho$   |        :x:         |        :x:         | :heavy_check_mark: |  :heavy_check_mark:   | :heavy_check_mark: |        $n$         |
|  Spearman, $\rho$   |        :x:         |        :x:         | :heavy_check_mark: |  :heavy_check_mark:   | :heavy_check_mark: |     $n\log n$      |
| RV Coeff, $\rho RV$ |        :x:         | :heavy_check_mark: | :heavy_check_mark: |  :heavy_check_mark:   | :heavy_check_mark: |       $n^2$        |
|        dCorr        | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  :heavy_check_mark:   | :heavy_check_mark: |       $n^2$        |
|        HSIC         | :heavy_check_mark: | :heavy_check_mark: |        :x:         |  :heavy_check_mark:   |        :x:         |       $n^2$        |
|        nHSIC        | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  :heavy_check_mark:   | :heavy_check_mark: |       $n^2$        |
|         MI          | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  :heavy_check_mark:   |        :x:         |         -          |
|         nMI         | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  :heavy_check_mark:   |        :x:         |         -          |

---
## Questions

1. Are there **correlations** across seasons or latitudes
2. Are there large descrepancies in the different outputs?



### Classes of Methods









---

## Resources



### Websites

* [Taylor Diagrams](https://climatedataguide.ucar.edu/climate-data-tools-and-analysis/taylor-diagrams)
* 

### Papers

[1]:   "The Mutual Information Diagram for Uncertainty Visualization - Correa & Lindstrom (2012)"
[2]: 	"Summarizing multiple aspects of model performance in a single diagram"

---

## References

```{bibliography}
```