---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.3
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="QMV_fJikDPKT" -->
# Parameterized Marginal Gaussianization


This is my notebook where I play around with all things normalizing flow with pyro. I use the following packages:

* PyTorch
* Pyro
* PyTorch Lightning
* Wandb
<!-- #endregion -->

```python id="hhn17BaNDNPP" colab={"base_uri": "https://localhost:8080/"} outputId="76dd5672-12ad-4123-83a8-8637e50468cd"
#@title Install Packages
# %%capture

!pip install --upgrade --quiet pyro-ppl tqdm wandb corner loguru pytorch-lightning lightning-bolts torchtyping einops plum-dispatch pyyaml==5.4.1 nflows
!pip install --upgrade --quiet scipy
```

```python colab={"base_uri": "https://localhost:8080/"} id="ZHd9hQBx6r5A" outputId="1f7c92dc-48d5-4a2d-8051-68850f44c21a"
!git clone https://github.com/jejjohnson/survae_flows_lib.git
!pip install survae_flows_lib/. --use-feature=in-tree-build
```

```python id="5vttXbI6DgSk" colab={"base_uri": "https://localhost:8080/"} outputId="757d217c-dc1c-4a0b-a678-9f5ef68625bf"
#@title Import Packages

# TYPE HINTS
from typing import Tuple, Optional, Dict, Callable, Union
from pprint import pprint

# PyTorch
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Pyro Imports
import pyro.distributions as dist
import pyro.distributions.transforms as T

# PyTorch Lightning Imports
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pl_bolts.datamodules import SklearnDataModule

# wandb imports
import wandb
from tqdm.notebook import trange, tqdm
from pytorch_lightning.loggers import TensorBoardLogger


# Logging Settings
from loguru import logger
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logger.info("Using device: {}".format(device))

# NUMPY SETTINGS
import numpy as np
np.set_printoptions(precision=3, suppress=True)

# MATPLOTLIB Settings
import corner
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

# SEABORN SETTINGS
import seaborn as sns
sns.set_context(context='talk',font_scale=0.7)

# PANDAS SETTINGS
import pandas as pd
pd.set_option("display.max_rows", 120)
pd.set_option("display.max_columns", 120)
```

<!-- #region id="mREQeNcanvJq" -->
#### HelpFul Functions
<!-- #endregion -->

<!-- #region id="EhmJphY1oHXc" -->
##### Generate 2D Grid
<!-- #endregion -->

```python id="WqJMiRNknwdt"
def generate_2d_grid(data: np.ndarray, n_grid: int = 1_000, buffer: float = 0.01) -> np.ndarray:

    xline = np.linspace(data[:, 0].min() - buffer, data[:, 0].max() + buffer, n_grid)
    yline = np.linspace(data[:, 1].min() - buffer, data[:, 1].max() + buffer, n_grid)
    xgrid, ygrid = np.meshgrid(xline, yline)
    xyinput = np.concatenate([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], axis=1)
    return xyinput
```

<!-- #region id="_2E_YHKyoJa-" -->
##### Plot 2D Grid
<!-- #endregion -->

```python id="WK3r6BZcoLnk"
from matplotlib import cm

def plot_2d_grid(X_plot, X_grid, X_log_prob):



    # Estimated Density
    cmap = cm.magma  # "Reds"
    probs = np.exp(X_log_prob)
    # probs = np.clip(probs, 0.0, 1.0)
    # probs = np.clip(probs, None, 0.0)


    cmap = cm.magma  # "Reds"
    # cmap = "Reds"

    fig, ax = plt.subplots(ncols=2, figsize=(12, 5))
    h = ax[0].hist2d(
        X_plot[:, 0], X_plot[:, 1], bins=512, cmap=cmap, density=True, vmin=0.0, vmax=1.0
    )
    ax[0].set_title("True Density")
    ax[0].set(
        xlim=[X_plot[:, 0].min(), X_plot[:, 0].max()],
        ylim=[X_plot[:, 1].min(), X_plot[:, 1].max()],
    )


    h1 = ax[1].scatter(
        X_grid[:, 0], X_grid[:, 1], s=1, c=probs, cmap=cmap, #vmin=0.0, vmax=1.0
    )
    ax[1].set(
        xlim=[X_grid[:, 0].min(), X_grid[:, 0].max()],
        ylim=[X_grid[:, 1].min(), X_grid[:, 1].max()],
    )
    # plt.colorbar(h1)
    ax[1].set_title("Estimated Density")


    plt.tight_layout()
    plt.show()
    return fig, ax

```

<!-- #region id="8UHy3Grgoe03" -->
##### Torch 2 Numpy
<!-- #endregion -->

```python id="xw1dmutgogkn"
def torch_2_numpy(X):

    if not isinstance(X, np.ndarray):
        try:
            X = X.numpy()
        except RuntimeError:
            X = X.detach().numpy()
        except TypeError:
            X = X.detach().cpu().numpy()
    

    return X
```

<!-- #region id="EMU9GR308W7G" -->
## 2D Toy Data
<!-- #endregion -->

```python id="NqgxNZVlQ_NX"
def get_toy_data(n_samples=1000, seed=123):
    rng = np.random.RandomState(seed=seed)

    x = np.abs(2 * rng.randn(n_samples, 1))
    y = np.sin(x) + 0.25 * rng.randn(n_samples, 1)
    data = np.hstack((x, y))

    return data
```

```python id="fRx3lhQmJ-N6"

```

```python id="kTyuN316RILe"
X = get_toy_data(5_000, 123)


# get marginal data
X_1 = X[:, 0][:, None]
X_2 = X[:, 1][:, None]
```

```python id="Ds7VSdS77Vm4"
# # Data

# ds = CheckerboardDataset



# test = Dataset(get_toy_data(2_000, 100)[0])
# train_loader = DataLoader(train, batch_size=64, shuffle=False)
# test_loader = DataLoader(test, batch_size=256, shuffle=True)


```

```python colab={"base_uri": "https://localhost:8080/", "height": 387} id="VCXWOwON7xwk" outputId="1694d80f-cf8a-45d5-953b-b92f3d1f5f85"
fig = corner.corner(X, color="blue")
fig.suptitle("Sine Wave")
plt.show()
```

<!-- #region id="ZUhElhna-B8f" -->
## Model
<!-- #endregion -->

<!-- #region id="AwFeXVKRb369" -->
## Parameterized Marginal Gaussianization
<!-- #endregion -->

<!-- #region id="cMpoxF2UeUPU" -->
### Univariate GMM

$$
p(\mathbf{x};\boldsymbol{\theta}) = \sum_k^K \pi_k \mathcal{N}(x| \mu_k, \sigma_k)
$$

where:

* $K$ - # of components
* $\pi_k$ - the probit weighting term for the $k$-th component
* $\boldsymbol{\theta} = \{ \boldsymbol{\mu}_K, \boldsymbol{\sigma}_K, \boldsymbol{\pi}_K \}$ 


We will use the `scikit-learn` implementation to learn the parameters for the Gaussian mixture model. They use the Expectation-Maximization (EM) scheme to solve for the parameters.
<!-- #endregion -->

```python id="kYLQGlIEb8Kc"
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
```

```python id="V3CZTjlTcGxk"
n_components = 4
random_state = 123
covariance_type = "diag"

# init gmm model
mg_bijection = GaussianMixture(
    n_components=n_components, 
    random_state=random_state, 
    covariance_type=covariance_type
)
```

```python colab={"base_uri": "https://localhost:8080/"} id="SQ3L-uV0ccgL" outputId="0d7c50f4-0721-4ce0-e965-386d36e18f19"
mg_bijection.fit(X_1)
```

<!-- #region id="vwiK66PEj5KQ" -->
##### Viz - PDF
<!-- #endregion -->

```python id="XxWFUdWmceG3"
x_domain = np.linspace(X_1.min(), X_1.max(), 100)
probs = mg_bijection.score_samples(x_domain[:, None])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 272} id="reJDhawWcolr" outputId="12b7b158-a67b-4bcc-de49-2f9fdf86ea62"
fig, ax = plt.subplots()
ax.plot(x_domain, np.exp(probs), label="Estimated Density")
ax.hist(X_1.ravel(), density=True, bins=50, color="Red", label="Samples")
plt.legend()
plt.show()
```

<!-- #region id="XxRR-bkmep3V" -->
#### CDF - From Scratch

$$
F(x) = \sum_k^K \pi_k \phi(x| \mu_k, \sigma_k)
$$

where $\phi$ is the CDF of the Gaussian distribution.


<!-- #endregion -->

<!-- #region id="cxPXStAzkFVM" -->

###### **Trick:** - Stabilization

We often like to put everything in terms of logs. This reparameterizes the functions so that the gradient updates of the parameters will be small irregardless of the function.


$$
\log U(x) = \sum_k^K \log \pi_k + \log \sum_k^K \mathcal{N}_{CDF}(x|\mu_k, \sigma_k)
$$
<!-- #endregion -->

```python id="SEYb7TyQfKco"
# extract parameters
logit_weights = mg_bijection.weights_[None, :]
means = mg_bijection.means_.T
sigmas = np.sqrt(mg_bijection.covariances_).T

# convert to tensors

# assert shapes
assert_shape = (1, n_components)

assert logit_weights.shape == assert_shape
assert means.shape == assert_shape
assert sigmas.shape == assert_shape
```

<!-- #region id="ftuqqvSHf3ik" -->
* `pi = (D, K)`
* `mu = (D, K)`
* `sigma = (D, K)`

where: 
* `K` - number of mixture components
* `D` - dimensionality of the data
<!-- #endregion -->

```python id="l9ZrHPlqtBb_"
# convert to tensors
logit_weights = torch.Tensor(logit_weights)
means = torch.Tensor(means)
sigmas = torch.Tensor(sigmas)


```

```python id="83DwBqP1shHz"
import torch.functional as F
import torch.distributions as dist
```

```python id="8FbwEXiHfbBC"
from scipy import stats
from scipy.special import log_softmax, logsumexp
from einops import repeat
```

```python id="JdGI4NdMr-yz"
# create base dist
mixture_dist = dist.Normal(means, sigmas)

# Calculate the CDF
x_cdfs = mixture_dist.cdf(torch.Tensor(x_domain).unsqueeze(-1))

# calculate mixture cdf
z_cdfs = logit_weights * x_cdfs

# sum mixture distributions
z_cdf = z_cdfs.sum(axis=-1)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 272} id="C5j7u9iEfuTv" outputId="84acf707-e85c-43a9-9c10-264aab5302a1"
fig, ax = plt.subplots()
ax.plot(torch_2_numpy(x_domain), torch_2_numpy(z_cdf), label="CDF")
plt.legend()
plt.show()
```

```python id="3o7IFan2tawo"
# Calculate the CDF
x_cdfs = mixture_dist.cdf(torch.Tensor(X_1))

# calculat mixture cdf
z_cdfs = logit_weights * x_cdfs

# sum mixture distributions
z_cdf = z_cdfs.sum(axis=-1)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 272} id="UZngFZl_mx-S" outputId="e966ddc3-78cd-4490-fb0d-6202b9c1546b"
fig, ax = plt.subplots()
ax.hist(torch_2_numpy(z_cdf), density=True, bins=50, color="Red", label="Uniform Domain")
plt.legend()
plt.show()
```

<!-- #region id="xq-AIi5wi-24" -->
#### PDF - From Scratch

We are going to use the same function as before. The only difference is the we will use the PDF instead of the CDF of a Gaussian.

$$
\log \nabla U(x) = \sum_k^K \log \pi_k + \log \sum_k^K \mathcal{N}_{PDF}(x|\mu_k, \sigma_k)
$$
<!-- #endregion -->

```python id="Vlip_OEitq7H"
# Calculate the PDF
x_pdfs = mixture_dist.log_prob(torch.Tensor(x_domain[:, None]))

# log softmax of weights
# log_weights = log_softmax(logit_weights, axis=-1)
log_weights = torch.log(logit_weights)

# calculat mixture cdf
z_logpdfs = log_weights + x_pdfs

# sum mixture distributions
z_logpdf = torch.logsumexp(z_logpdfs, axis=-1)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 272} id="MDOGxBrYjWhq" outputId="29fdd788-3c7a-45ea-8558-b87acc2ee014"
fig, ax = plt.subplots()
ax.plot(x_domain, torch_2_numpy(z_logpdf.exp()), label="PDF (Ours)", linewidth=4)
ax.plot(x_domain, torch_2_numpy(np.exp(probs)), label="PDF (GMM)", linestyle="dotted", linewidth=4)
ax.hist(X_1.ravel(), density=True, bins=50, color="Red", label="Samples")
plt.legend()
plt.show()
```

<!-- #region id="Ttn56-AKqWwD" -->
#### Inverse CDF 
<!-- #endregion -->

<!-- #region id="w6L0e_mCwyJ9" -->
##### Don't do this
<!-- #endregion -->

```python id="mvjEQG3vulVO"
z_domain = torch.linspace(0.01, 0.99, 100)
```

```python id="F5WiNRu_qY0H"
# Calculate the CDF
x_icdfs = mixture_dist.icdf(z_domain.unsqueeze(-1))

# calculat mixture cdf
xs = logit_weights * x_icdfs

# sum mixture distributions
x_domain = xs.sum(axis=-1)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 272} id="WEpzmBr3uc92" outputId="382d2196-8f07-4344-af49-8f8f85cc96d4"
fig, ax = plt.subplots()

ax.plot(z_domain, x_domain, label="Inverse CDF")
plt.legend()
plt.show()
```

```python id="XKVV6dfLwPGA"
# Calculate the CDF
x_icdfs = mixture_dist.icdf(z_cdf.unsqueeze(-1))

# calculat mixture cdf
xs = logit_weights * x_icdfs

# sum mixture distributions
x_approx = xs.sum(axis=-1)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 272} id="jWCSjbKEwVgp" outputId="b03c4f6c-552b-449f-b7d0-97a59fdb3d4e"
fig, ax = plt.subplots()
ax.hist(torch_2_numpy(X_1), density=True, bins=50, color="Blue", label="Original Data")
ax.hist(torch_2_numpy(x_approx), density=True, bins=50, color="Red", label="Inverse Transform")
plt.legend()
plt.show()
```

<!-- #region id="45JweZPOw2Qp" -->
#### Bisection Search
<!-- #endregion -->

```python id="s1ifMIu5Loqw"
from survae.transforms.bijections.functional.iterative_inversion import bisection_inverse
```

```python id="ifKSRkbAw3fv"
def mix_cdf(x, logit_weights, means, sigmas):

    # Calculate the CDF
    x_cdfs = mixture_dist.cdf(x.unsqueeze(-1))

    mix_dist = dist.Normal(means, sigmas)

    # calculat mixture cdf
    z_cdfs = logit_weights * x_cdfs

    # sum mixture distributions
    z_cdf = z_cdfs.sum(axis=-1)
    
    return z_cdf
```

```python id="Gow3Hg0TLX9Z"
# initialize the parameters

max_scales = torch.sum(sigmas, dim=-1, keepdim=True)
init_lower, _ = (means - 20 * max_scales).min(dim=-1)
init_upper, _ = (means + 20 * max_scales).max(dim=-1)

```

```python id="_JdZvStjLkY7"

```

```python id="Kkno6I5PLe9L"
x_approx = bisection_inverse(
    fn=lambda x: mix_cdf(x, logit_weights, means, sigmas),
    z=z_cdf,
    init_x=torch.zeros_like(z_cdf),
    init_lower=init_lower,
    init_upper=init_upper,
    eps=1e-10,
    max_iters=100,
)
```

```python id="yUezxqVLwRYl"
x_approx_ = gaussian_mixture_transform(z_cdf.unsqueeze(-1), logit_weights, means, sigmas.log(), inverse=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 272} id="QL2b0lSwwsJY" outputId="3fbcb98a-26ad-4054-c4a6-5942f5bd8792"
fig, ax = plt.subplots()
ax.hist(torch_2_numpy(X_1), density=True, bins=50, color="Blue", label="Original Data")
ax.hist(torch_2_numpy(x_approx), density=True, bins=50, color="Red", label="Inverse Transform")
plt.legend()
plt.show()
```

<!-- #region id="DYBLfdHyvNS1" -->
#### Bijections
<!-- #endregion -->

```python id="WhzT9H2IvOMQ"
from survae.transforms.bijections.functional.mixtures.gaussian_mixture import gaussian_mixture_transform
```

```python id="NLp4rFhnvYuG"

```

```python colab={"base_uri": "https://localhost:8080/", "height": 272} id="imNejGCTvi0s" outputId="694488bd-f4b7-4d9d-879d-0af9068ee396"
fig, ax = plt.subplots()

ax.plot(z_domain, x_approx, label="Inverse CDF")
ax.plot(z_domain, x_approx_, label="Inverse CDF (Bisection)")
plt.legend()
plt.show()
```

```python id="-UfPVDrtqrVt"

```

```python id="B2Fa1sATqepU"
fig, ax = plt.subplots()
ax.plot(x_domain, z_cdf, label="CDF")
plt.legend()
plt.show()
```

<!-- #region id="TLJz7T3XmWLn" -->
#### SurVAE Flows Function

Now, we will translate the code into PyTorch. I will create a functional form so that we just need to call it within our Bijector class.
<!-- #endregion -->

```python id="mQxIERxpmXcH"
# from survae.transforms.bijections.functional.mixtures import gaussian_mixture_transform
from torch.distributions import Normal

def gaussian_mixture_transform(inputs, logit_weights, means, log_scales):

    dist = Normal(means, log_scales.exp())

    def mix_cdf(x):
        return torch.sum(logit_weights * dist.cdf(x.unsqueeze(-1)), dim=-1)

    def mix_log_pdf(x):
        return torch.logsumexp(logit_weights.log() + dist.log_prob(x.unsqueeze(-1)), dim=-1)
    z = mix_cdf(inputs)
    ldj = mix_log_pdf(inputs)

    return z, ldj
```

<!-- #region id="zZkGooqwlHBQ" -->
#### Initialization

I will also create an initialization function which will allow us to initialization the parameters from the GMM model.
<!-- #endregion -->

```python id="3N52Ip-Gs5Ej"
from survae.transforms.bijections.functional.mixtures.gaussian_mixture import init_marginal_mixture_weights
```

```python id="WYvE3DOLtPkv"
init_marginal_mixture_weights??
```

```python id="b9qHjcxEoki_"
num_mixtures = 4

logit_weights, means, sigmas = init_marginal_mixture_weights(X, num_mixtures)
```

<!-- #region id="Poq0J4mKsizK" -->

<!-- #endregion -->

```python id="pYnVUYQLo9FJ"
X_mu, ldj = gaussian_mixture_transform(
    torch.Tensor(X), 
    torch.Tensor(logit_weights), 
    torch.Tensor(means), 
    torch.Tensor(np.log(sigmas))
)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 360} id="TfkfgvQQpfH1" outputId="cf1c8936-d9c5-44ff-a454-3f84911384e3"
fig = corner.corner(X_mu.numpy())
```

<!-- #region id="7xeo7jP9s3I8" -->
### PyTorch Class
<!-- #endregion -->

```python id="pGZ1fLfTtXO3"
from survae.transforms.bijections.elementwise_nonlinear import GaussianMixtureCDF
from survae.transforms.bijections.elementwise_nonlinear import InverseGaussCDF

```

```python id="4v3EKEbxtbDi"
GaussianMixtureCDF??
```

```python id="h41-6DJCpT9T"
InverseGaussCDF??
```

<!-- #region id="A2yKoa1hlVM0" -->
### Flow Model

So to have a flow model, we need two components:


---
**Base Distribution**: $p_Z = \mathbb{P}_Z$

This will describe the distribution we want in the transform domain. In this case, we will choose the uniform distribution because we are trying to uniformize our data.

---
**Bijections**: $f = f_L \circ f_{L-1} \circ \ldots \circ f_1$

The list of bijections. These are our functions which we would like to compose together to get our dataset.

<!-- #endregion -->

```python id="hF1JAad5pYxr"
from survae.distributions import StandardUniform, StandardNormal
from survae.flows import Flow


# base distribution
base_dist = StandardNormal((2,))

# transforms
transforms = [
              GaussianMixtureCDF((2,), None, 4),
              InverseGaussCDF()
]

# flow model
model = Flow(
    base_dist=base_dist,
    transforms=transforms
)
```

<!-- #region id="OJ1MJMjbmINi" -->
#### Forward Transformation
<!-- #endregion -->

```python id="6awHrNXXtqWE" colab={"base_uri": "https://localhost:8080/", "height": 360} outputId="6e60b789-95fa-421e-da8d-2fb36cdc047f"
with torch.no_grad():

    X_mu, ldj = model.forward_transform(torch.Tensor(X))

fig = corner.corner(torch_2_numpy(X_mu))
```

<!-- #region id="3VOdU-HwmNoL" -->
#### Inverse Transformation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 360} id="kDEZwe9pmO8j" outputId="d930b604-4644-485a-c8ce-07396da854f7"
with torch.no_grad():

    X_mu, ldj = model.forward_transform(torch.Tensor(X))

fig = corner.corner(torch_2_numpy(X_mu))
```

<!-- #region id="yX9SRYK-MbD3" -->

<!-- #endregion -->

<!-- #region id="qOFtkG4hivgQ" -->
## Training
<!-- #endregion -->

<!-- #region id="Wgwd--1eitR6" -->
### Dataset
<!-- #endregion -->

```python id="19EQPGSoiufB"
# # Data
X_train = get_toy_data(5_000, 123)

train_loader = DataLoader(torch.Tensor(X_train), batch_size=128, shuffle=True)

```

<!-- #region id="MFRD2bP2h3i8" -->
### Loss
<!-- #endregion -->

```python id="WYew7oTAh4WR"
def nll_loss(model, data):
    return - model.log_prob(data).mean()
```

<!-- #region id="QOGFZSU0h47U" -->
### Pytorch-Lightning Trainer
<!-- #endregion -->

```python id="FbmfP07Qh57R"
import pytorch_lightning as pl

class Learner2DPlane(pl.LightningModule):
    def __init__(self, model:nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        
        # loss function
        # loss = -self.model.log_prob(batch).mean()
        loss = nll_loss(self.model, batch)
        
        self.log("train_loss", loss)
        
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def train_dataloader(self):
        return train_loader
```

```python id="qDYHXvHjnyZW"
# initialize trainer
learn = Learner2DPlane(model)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Infobxf7nzns" outputId="ab222248-c90e-4ece-8620-979958fdbfb1"
n_epochs = 50
logger = TensorBoardLogger("tb_logs", name='mg_no_init')

# initialize trainer
trainer = pl.Trainer(min_epochs=1, max_epochs=n_epochs, gpus=1, enable_progress_bar=True, logger=logger)
```

<!-- #region id="np4yqwXQiK9G" -->
### Logging
<!-- #endregion -->

```python id="_3WDvw58iL5H"
# %load_ext tensorboard
# %tensorboard --logdir tb_logs/
```

<!-- #region id="J7RVhI5hiDfz" -->
### Training
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 277, "referenced_widgets": ["c9d47f14f2614818b2248f0a309377dd", "d0c2a714793b46028465415977dd8ab1", "ff9d869ba0df4089a0f417b505159f96", "ce564aedaae94b9c9e312092185bbb23", "d5f360c164954278939b799640cfb099", "593b00f036064cb897e9edefa5817508", "7ae0d48746a74c759f8a04b51994d7ca", "6d7204fc9ec545d8a66ab9b8dd3e73ec", "b91dcb062af249db90e645b5737e3751", "2aa1b159daaa489c89130e112d8c1c26", "623ccbdfb9644bec9c863939565aa931"]} id="S8-cz5qxhyIW" outputId="dea1b3d7-325f-45fa-f859-2a42dca6df01"
# train model
trainer.fit(learn, )
```

<!-- #region id="ifZDTNsWiZos" -->
## Results
<!-- #endregion -->

<!-- #region id="oTUVxFbZ5Vf0" -->

### Latent Domain
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 367} id="kCDNmj0J5XOE" outputId="c38a2616-c6d9-4867-f638-a4032e92a4ad"
with torch.no_grad():
    X_ = torch.Tensor(X)
    # X_ = X_.to(device)
    X_r, ldj = learn.model.forward_transform(X_)


fig = corner.corner(torch_2_numpy(X_r))
```

<!-- #region id="qaMRInWJ5c2L" -->
### Inverse
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 367} id="oyqf_7Gc5dtt" outputId="80fd62a6-9ed9-47d7-ae36-a3d148eaef0a"
with torch.no_grad():
    # X_ = X_.to(device)
    X_approx = learn.model.inverse_transform(X_r)

fig = corner.corner(torch_2_numpy(X_approx))
```
