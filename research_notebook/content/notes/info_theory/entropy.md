# Entropy & Relative Entropy

### Entropy

This is an upper bound on the amount of information you can convey without any loss ([source](https://blog.evjang.com/2019/07/likelihood-model-tips.html)). More entropy means more **randomness** or **uncertainty**

$$H(X)=\int_{\mathcal{X}}p(x)\cdot \log p(x) \cdot dx$$

We use logs so that wee get sums of entropies. It implies independence but the log also forces sums.

$$H(Y|X) = H(X,Y)-H(X)$$

$$H(Y|X) = \int_{\mathcal{X}, \mathcal{Y}}p(x,y) \log \frac{p(x,y)}{p(x)}dxdy$$


#### Examples

**Example Pt II: Delta Function, Uniform Function, Binomial Curve, Gaussian Curve**

#### Under Transformations

In my line of work, we work with generative models that utilize the change of variable formulation in order to estimate some distribution with 

$$H(Y) = H(X) + \mathbb{E}\left[ \log |\nabla f(X)|\right]$$

* Under rotation: Entropy is invariant
* Under scale: Entropy is ...???
* Computational Cost...?

---
### Relative Entropy (Kullback Leibler Divergence)

This is the measure of the distance between two distributions. I like the term *relative entropy* because it offers a different perspective in relation to information theory measures.

$$D_{KL}(p||q) = \int_{\mathcal{X}}p(x) \cdot \log \frac{p(x)}{q(x)} \cdot dx \geq 0$$

If you've studied machine learning then you are fully aware that it is not a distance as this measure is not symmetric i.e. $D_{KL}(p||q) \neq D_{KL}(q||p)$.


Furthermore, the KL divergence is the difference between the cross-entropy and the entropy.

$$D_{KL}(P||Q) = H(P, Q) - H(P)$$

So this is how far away our predictions are from our actual distribution.


#### Under Transformations

The KLD is invariance under invertible affine transformations, e.g. $b = \mu + Ga, \nabla F = G$

$$
\text{D}_\text{KL}\left[p(y)||q(y) \right] = \int_\mathcal{Y}p(y)\log \frac{p(y)}{q(y)}dy
$$

Let's make a transformation on $p(y)$ using some nonlinear function $f()$. So that leaves us with $y = f(x)$. So let's apply the change of variables formula to get the probability of $y$ after the transformation.

$$
p(y)dy=p(x)dx = p(x)\left|\frac{dx}{dy} \right|
$$

Remember, we defined our function as $y=f(x)$ so technically we don't have access to the probability of $y$.  Only the probability of $x$. So we cannot take the derivative in terms of y. But we can take the derivative in terms of $x$. So let's rewrite the function:

$$
p(y) = p(x) \left| \frac{dy}{dx} \right|^{-1}
$$

Now, let's plug in this formula into our KLD formulation.

$$
\text{D}_\text{KL}\left[p(y)||q(y) \right] =
\int_\mathcal{y=?}p(x)\left| \frac{dy}{dx} \right|^{-1}
\log \frac{p(x) \left| \frac{dy}{dx} \right|^{-1}}{q(y)}dy
$$

We still have two terms that need to go: $dy$ and $q(y)$. For the intergration, we can simply multiple by 1 to get $dy\frac{dx}{dx}$ and then with a bit of rearranging we get: $\frac{dy}{dx}dx$. I'm also going to change the notation as well to get $\left| \frac{dy}{dx}  \right|dx$. And plugging this in our formula gives us:

$$
\text{D}_\text{KL}\left[p(y)||q(y) \right] =
\int_\mathcal{y=?}p(x)\left| \frac{dy}{dx} \right|^{-1}
\log \frac{p(x) \left| \frac{dy}{dx} \right|^{-1}}{q(y)}
\left| \frac{dy}{dx}  \right|dx
$$

Now, we still have the distribution $q(y)$.



---
## Normalized Variants


---

> Expected uncertainty.

$$H(X) = \log \frac{\text{\# of Outcomes}}{\text{States}}$$

* Lower bound on the number of bits needed to represent a RV, e.g. a RV that has a unform distribution over 32 outcomes.
  * Lower bound on the average length of the shortest description of $X$
* Self-Information


<details>
The standard definition of Entropy can be written as:



$$\begin{aligned}
D_{KLD}(P||Q) &=-\int_{-\infty}^{\infty} P(x) \log \frac{Q(y)}{P(x)}dx\\
&=\int_{-\infty}^{\infty} P(x) \log \frac{P(x)}{Q(y)}dx
\end{aligned}$$

and the discrete version:

$$\begin{aligned}
D_{KLD}(P||Q) &=-\sum_{x\in\mathcal{X}} P(x) \log \frac{Q(x)}{P(x)}\\
&=\sum_{x\in\mathcal{X}} P(x) \log \frac{P(x)}{Q(y)}
\end{aligned}$$


If we want the viewpoint in terms of expectations, we can do a bit of rearranging to get:

$$\begin{aligned}
D_{KLD} &= \sum_{x\in\mathcal{X}} P(x) \log \frac{P(x)}{Q(y)}\\
&= \sum_{x\in\mathcal{X}} P(x) \log P(x)- \sum_{-\infty}^{\infty}P(x)\log Q(y)dx \\
&= \sum_{x\in\mathcal{X}} P(x)\left[\log P(x) - \log Q(y) \right] \\
&= \mathbb{E}_x\left[ \log P(x) - \log Q(y)  \right]
\end{aligned}$$


</details>

<details>

#### Code - Step-by-Step

1. Obtain all of the possible occurrences of the outcomes. 
   ```python
   values, counts = np.unique(labels, return_counts=True)
   ```

2. Normalize the occurrences to obtain a probability distribution
   ```python
   counts /= counts.sum()
   ```

3. Calculate the entropy using the formula above
   ```python
   H = - (counts * np.log(counts, 2)).sum()
   ```

As a general rule-of-thumb, I never try to reinvent the wheel so I look to use whatever other software is available for calculating entropy. The simplest I have found is from `scipy` which has an entropy function. We still need a probability distribution (the counts variable). From there we can just use the entropy function.

2. Use Scipy Function
   ```python
   H = entropy(counts, base=base)
   ```
</details>

## Formulas

$$H(\mathbf{X}) = - \int_\mathcal{X} p(\mathbf{x}) \log p(\mathbf{x}) d\mathbf{x}$$

And we can estimate this empirically by:

$$H(\mathbf{X}) = -\sum_{i=1}^N p_i \log p_i$$

where $p_i = P(\mathbf{X})$.

### Code - Step-by-Step

```python
# 1. obtain all possible occurrences of the outcomes
values, counts = np.unique(labels, return_counts=True)

# 2. Normalize the occurrences to obtain a probability distribution 
counts /= counts.sum()

# 3. Calculate the entropy using the formula above
H = - (counts * np.log(counts, 2)).sum()
```

As a general rule-of-thumb, I never try to reinvent the wheel so I look to use whatever other software is available for calculating entropy. The simplest I have found is from `scipy` which has an entropy function. We still need a probability distribution (the counts variable). From there we can just use the entropy function.


### Code - Refactored

```python
# 1. obtain all possible occurrences of the outcomes
values, counts = np.unique(labels, return_counts=True)

# 2. Normalize the occurrences to obtain a probability distribution 
counts /= counts.sum()

# 3. Calculate the entropy using the formula above
base = 2
H = entropy(counts, base=base)
```

---
## Other Spaces


### Renyi

Above we looked at Shannon entropy which is a special case of Renyi's Entropy measure. But the generalized entropy formula actually is a generalization on entropy. Below is the given formula. 

$$
H_\alpha(x) = \frac{1}{1-\alpha} \log_2 \sum_{x \in \mathcal{X}} p^{\alpha}(x)
$$


---
## References

* Lecture Notes I - [PDF](http://www.ece.tufts.edu/ee/194NIT/lect01.pdf)
* Video Introduction - [Youtube](https://www.youtube.com/watch?v=ErfnhcEV1O8)
* [Prezi](https://gtas.unican.es/files/docencia/TICC/apuntes/tema1bwp_0.pdf) - Entropy and Mutual Info
    > Good formulas, good explanations, Gaussian stuff