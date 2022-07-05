# Confidence Intervals

* Last Update: 19-April-2021

> My notes on how one can obtain confidence intervals using Machine learning methods. 


---
## Methods


---
### TLDR


:::{note}
A lot of these methods are common within the **machine learning** literature, not the **scientific literature**. There is a huge difference between the fields regarding what's popular for research and what's popular for practical applications. A lot of the more complex methods are because people are using giant models trying to fit large and very complex datasets. Many times, smaller data problems (<10K data points) don't require so much difficult stuff.
:::

**Bootstrapping** - A very simple way to obtain confidence intervals is to using bootstrapping. This works by taking random permutations of your dataset and then training multiple models given this subset of data. This is advantageous because we can use any model we want, we just have to find ways of permuting the data effectively. For small data problems, we do run the risk of using so few data points that it becomes ill-advised. However, in general, it's a very trustworthy method.


**Ensembles** - This method is analagous to the above approach but instead of many models using different permutations of the data, we train multiple models using different initial conditions. It follows the idea that combinations many smaller "weak learners" can result in a robust learner. Random forests are a notoriously good example of this. With a few modifications, one can get confidence intervals. This method is also very popular with neural networks because they tend to have a lot of local minima. This can get more expensive and more because it requires training, storing and predicting these huge models; essentially a lot of plumbing. However, these are currently the state-of-the-art methods right now.


**Gaussian Processes** - The best thing to use is a Gaussian process (GP). They come equipped with confidence intervals already. They work by using a mean function and a kernel function defined by hyperparameters to obtain a function approximation conditioned on your data. For small data problems (< 2,000 points) you're not going to find a better method. Scaling is an issue after 10K points but there are many sparse methods which compensate this with approximations. There are studies that show the confidence intervals can deteriorate with these approximations but there are other methods investigating ways to compensate this.

**Bayesian Methods** - Alternatively, there are many parametric methods available. These work by strict adherence to the Bayesian rule whereby we define a prior over the parameters, a likelihood describing the data generating process and we try to obtain a posterior which gives us the parameters given the data. Simpler to GPs, they also come equipped with confidence intervals because we inherently have to marginalize over the distribution of parameters given the data. For simple linear problems, it's highly recommended to try these first before attempting anything more difficult. For more non-linear, difficult problems, the issue of finding the best parameterization becomes an issue. And we also find that the uncertainty estimation starts to become an issue with larger models (e.g. Bayesian Neural Networks).


**Quantile Regression** - This method works by augmenting the loss function such that you find a model that predicts the quantile instead of just the mean. Recall that standard ML models attempt to predict the mean and others attempt to predict the distribution. With quantile regression methods, you decide which quantile you want to predict, e.g. the 50th quantile (the mean/median) and 5th quantile (the lower confidence interval) and/or the 95th quantile (the uppter confidence interval). It the most popular method you find in the literature and I have no idea why. It's a very simple method and possibly preferable with linear methods. But there are some good examples for non-linear methods (like [Gradient Boosting](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html)). Some downsides I've observed are that the confidence intervals tend to look a bit rigid compared to many other methods (e.g. GPs). Also sometimes the quantiles are a lot higher than I would have expected.  

---




---
### Gaussian Processes


#### Standard


#### Fully Bayesian

These put priors on everything and they solve the inference problem using sampling such as Monte Carlo estimation.


#### Sparse

The problem with standard GPs is that they are very slow after about 10K samples (on a beefy machine). So most people resort to sparse methods. This works by conditioning the best subset of points (inducing points) in order to reduce the computation cost of $\mathcal{O}(N^3)$ to $\mathcal{O}(NM^2)$ which is significantly less expensive. There are quite a few standard methods out there including:

* Fully Independent Training Conditional (FITC)
* Variational Free Energy (VFE)
* Stochastic Variational Inference (SVI)

These methods all work really well with SVI being the most flexible and FITC being the most restrictive.

#### Deep Kernel Learning

One of the biggest limiations of Gaussian processes is the expressivity of the kernel function. There are just some datasets that are hard to fit. So one augmentation is to use something called Deep Kernel Learning (DKL) {cite}`dkl_wilson_2016,dklsvi_wilson_2016`.This works by attaching a neural network before the kernel. So instead of the inputs going directly into the kernel function, they go through a neural network first (kind of like an encoder) and then they are passed through to the kernel function.

However, because DKL are using neural networks, standard training procedures (i.e. maximum likelihood estimation) tend to exhibit the same pathologies that we see in standard neural networks {cite:p}`DKLPitFalls2021`.



---
### Neural Networks


#### DropOut

Dropout is probably the most common method {cite}`dropout_2016`. Dropout works by randomly removing neurons during the training **and** testing phase. This works as an approximate method to Bayesian neural networks. The advantage of this is that it's very simple and it's relatively fast. If you're already using a neural network, chances are you're already using dropout as a regularization technique. So there will be minimum changes to your code. The downside is there there have been studies showing that it the confidence intervals aren't the best quality.


#### Probabilistic

This method is a nice trade-off between the standard neural network and the Bayesian neural network. Basically, one would just tack on a stochastic layer at the end of the neural network and voila, uncertainty.


#### Bayesian Neural Networks

**Warning**: Training BNNs is very difficult. Especially for high dimensional, large sample problems. If you have the time and computational power, then go for it, why not. Otherwise, I would sit tight and


---
#### Laplace Approximation


* [Laplace Redux](https://arxiv.org/abs/2106.14806)


---

### Bootstrapping


---

### Ensembles

* [Confidence Intervals for Random Forests: The Jackknife and the Infinitesimal Jackknife](https://arxiv.org/abs/1311.4555) - Wager et al (2013) | [Software](https://github.com/scikit-learn-contrib/forest-confidence-interval)


---
### Quantile Regression


---
### Conformal Prediction

* A Tutorial on Conformal Prediction - Shafer and Vovk (2008) - [PDF](https://arxiv.org/abs/0706.3188)
* A Gentle Intro to Conformal Prediction and Distribution-Free Uncertainty Quantification - Angelopoulos & Bates (2022) - [arxiv](https://arxiv.org/abs/2107.07511)
* [Demo Notebook](https://nbviewer.org/github/gpeyre/numerical-tours/blob/master/python/ml_11_conformal_prediction.ipynb) - Numerical-Tours - Gabriel-Peyré



---
## Evaluating Uncertainties


---
## Resources


---
### Videos

**Practical Uncertainty Estimation and Out-of-Distribution Robustness in Deep Learning** - [NeuRIPS 2020 Tutorial](https://nips.cc/virtual/2020/public/tutorial_0f190e6e164eafe66f011073b4486975.html)

> The best tutorial I've seen for uncertainty in Neural networks. They focus almost exclusively on Neural networks (aka not small data problems).

---
### Papers

**A Review of Unvertainty Quantification in Deep Learning: Techniques, Applications and Challenges** - Abdar et al (2021) - [Information Fusion](https://arxiv.org/abs/2011.06225)

**A Survey of Uncertainty in Deep Neural Networks** - Gawlikowski et al (2021) - [arxiv](https://arxiv.org/abs/2107.03342)

**A Survey on Uncertainty Estimation in Deep Learning Classification Systems from a Bayesian Perspective** - Mena et al (2022) - [ACM Compute Surveys](https://www.semanticscholar.org/paper/A-Survey-on-Uncertainty-Estimation-in-Deep-Learning-Mena-Pujol/a03feff82c32a79ea8a8509193890266ae6fddf7)


**On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks** - Seitzer et al (2022) - [ICLR](https://arxiv.org/abs/2203.09168)



---
### Practical Resources



**A Regression Master Class with Aboleth** - [Blog](https://aboleth.readthedocs.io/en/stable/tutorials/some_regressors.html)

> Probably the best tutorial I've seen for regression methods.

**Probabilistic Layers Regression** - [TensorFlow Probability](https://www.tensorflow.org/probability/examples/Probabilistic_Layers_Regression) | [Keras](https://keras.io/examples/keras_recipes/bayesian_neural_networks/)

> Another great tutorial showcasing how to get uncertainty using the TensorFlow probability package.