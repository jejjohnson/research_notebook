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
# Parameterized Rotations


This is my notebook where I play around with all things normalizing flow with pyro. I use the following packages:

* PyTorch
* Pyro
* PyTorch Lightning
* Wandb
<!-- #endregion -->

```python id="hhn17BaNDNPP" colab={"base_uri": "https://localhost:8080/"} outputId="bc7b3dac-00ed-4271-e737-6af912999c32"
#@title Install Packages
# %%capture

!pip install --upgrade --quiet pyro-ppl tqdm wandb corner loguru pytorch-lightning lightning-bolts torchtyping einops plum-dispatch pyyaml==5.4.1 nflows
!pip install --upgrade --quiet scipy
```

```python colab={"base_uri": "https://localhost:8080/"} id="ZHd9hQBx6r5A" outputId="fa8abb41-6fd9-45d8-a349-d0d70efa031b"
!git clone https://github.com/jejjohnson/survae_flows_lib.git
!pip install survae_flows_lib/. --use-feature=in-tree-build
```

```python id="5vttXbI6DgSk" colab={"base_uri": "https://localhost:8080/"} outputId="cc3ffbc2-154c-4a6b-9d14-e764097db7f6"
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

init_X = torch.Tensor(X)
```

```python id="Ds7VSdS77Vm4"
# # Data

# ds = CheckerboardDataset



# test = Dataset(get_toy_data(2_000, 100)[0])
# train_loader = DataLoader(train, batch_size=64, shuffle=False)
# test_loader = DataLoader(test, batch_size=256, shuffle=True)


```

```python colab={"base_uri": "https://localhost:8080/", "height": 387} id="VCXWOwON7xwk" outputId="b60c1098-8a40-45fe-99ff-044fe9c84503"
fig = corner.corner(X, color="blue")
fig.suptitle("Sine Wave")
plt.show()
```

<!-- #region id="AwFeXVKRb369" -->
## Parameterized Marginal Gaussianization
<!-- #endregion -->

```python id="gdGWT0XnhH2x"
from survae.transforms.bijections.elementwise_nonlinear import GaussianMixtureCDF
from survae.transforms.bijections.elementwise_nonlinear import InverseGaussCDF
```

```python id="T44kvyk9hZYG"
# GaussianMixtureCDF??
```

```python id="iwTJpwHIhGz6"


# marginal gaussianization
mu_bijector = GaussianMixtureCDF((2,), None, 4)

# inverse gaussian cdf
icdf_bijector = InverseGaussCDF()


with torch.no_grad():

    X_mu, _ = mu_bijector.forward(init_X)
    X_mg, _ = icdf_bijector.forward(X_mu)


```

```python colab={"base_uri": "https://localhost:8080/", "height": 751} id="GbwnnAHWm0L3" outputId="6428a3d9-7181-4a69-c35a-b15893294237"
fig = corner.corner(torch_2_numpy(X_mu), color="blue")
fig.suptitle("Marginal Uniformization")
plt.show()

fig = corner.corner(torch_2_numpy(X_mg), color="blue")
fig.suptitle("Marginal Gaussianization")
plt.show()
```

<!-- #region id="1raRvhIDMh0W" -->
## Orthogonal Parameterization
<!-- #endregion -->

<!-- #region id="I3YL1BZQg2Nf" -->
### HouseHolder Reflections



<!-- #endregion -->

<!-- #region id="V3PS9SAfg2Nt" -->
**Algorithm**


$\mathbf{H}_k$ is reflection matrix and is defined by.

$$
\mathbf{H}_K = \mathbf{I} - 2 \frac{\mathbf{v}_k\mathbf{v}_k^\top}{||\mathbf{v}||_2^2}
$$

where $\mathbf{v}_k \in \mathbb{R}^{D}$. 
<!-- #endregion -->

```python id="48ndezqbg2Nw"
num_reflections = 3
num_dimensions = 2

# create vectors, v
v_vectors = torch.ones(num_reflections, num_dimensions)

# calc denominator
squared_norms = torch.sum(v_vectors ** 2, dim=-1)

# initialize loop
Q = torch.eye(num_dimensions)
```

<!-- #region id="RHTwenN6jvfP" -->
Multiply all matrices together.

$$
\mathbf{R} = \mathbf{H}_K \mathbf{H}_{K-1}\ldots \mathbf{H}_{1}
$$

where $\mathbf{R},\mathbf{H}_k \in \mathbb{R}^{D \times D}$ are orthogonal matrices. 
<!-- #endregion -->

```python id="M_WcIvrxg2Nw"
# loop through all vectors
for v_vector, squared_norm in zip(v_vectors, squared_norms):

    # Inner product.
    temp = Q @ v_vector  

    # Outer product.
    temp = torch.ger(temp, (2.0 / squared_norm) * v_vector)  
    Q = Q - temp
```

```python id="TuwROwzqg2Nw"
# check dimensions
assert Q.shape == (num_dimensions, num_dimensions)

# check it's orthogonal
assert (Q @ Q.T).all() == torch.eye(num_dimensions).all()
```

```python id="otlh3MZCg2Nx"
X_r = init_X @ Q
```

```python colab={"base_uri": "https://localhost:8080/", "height": 387} id="rtf6T4Ifg2Nx" outputId="4c1743db-814b-4527-c6a5-48607311de3b"
fig = corner.corner(X_r.cpu().numpy(), color="blue")
fig.suptitle("Sine Wave")
plt.show()
```

```python id="RyQoMojng2Nx"
def householder_product(vectors: torch.Tensor) -> torch.Tensor:
    """
    Args:
        vectors [K,D] - q vectors for the reflections
    
    Returns:
        R [D, D] - householder reflections
    """
    num_reflections, num_dimensions = vectors.shape

    squared_norms = torch.sum(vectors ** 2, dim=-1)

    # initialize reflection
    H = torch.eye(num_dimensions)

    for vector, squared_norm in zip(vectors, squared_norms):
        temp = H @ vector  # Inner product.
        temp = torch.ger(temp, (2.0 / squared_norm) * vector)  # Outer product.
        H = H - temp

    return H


```

```python id="QUXeZsoJg2Nx"
# initialize vectors
v_vectors = torch.ones(num_reflections, num_dimensions)
v_vectors = torch.nn.init.orthogonal_(v_vectors)


# householder product
R = householder_product(v_vectors)

# inverse householder product
reverse_idx = torch.arange(num_reflections - 1, -1, -1)
R_inv = householder_product(v_vectors[reverse_idx])


# check the inverse
torch.testing.assert_allclose(R @ R.T, torch.eye(num_dimensions), rtol=1e-5, atol=1e-5, )
```

<!-- #region id="z8HNiSCKg2Nx" -->
Cost - `O(KDN)`
`O(KD^2)`
<!-- #endregion -->

<!-- #region id="TkCwAZZAndFg" -->
### Pytorch Class
<!-- #endregion -->

```python id="ujGV8VQ_g2Nx"
from survae.transforms.bijections import Bijection

class LinearHouseholder(Bijection):
    """
    """

    def __init__(self, num_features: int, num_reflections: int = 2):
        super(LinearHouseholder, self).__init__()
        self.num_features = num_features
        self.num_reflections = num_reflections
        

        # initialize vectors param
        vectors = torch.randn(num_reflections, num_features)
        self.vectors = nn.Parameter(vectors)

        # initialize parameter to be orthogonal
        nn.init.orthogonal_(self.vectors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # get rotation matrix
        R = householder_product(self.vectors)

        # Z = x @ R
        z = torch.mm(x, R)

        # ldj -> identity
        batch_size = x.shape[0]
        ldj = x.new_zeros(batch_size)

        return z, ldj

    def inverse(self, z):
        # get rotation matrix (in reverse)
        reverse_idx = torch.arange(self.num_reflections - 1, -1, -1)
        R = householder_product(self.vectors[reverse_idx])
        
        x = torch.mm(z, R)

        return x
```

```python id="mxH01VYFg2Nx"

```

```python id="M3LEPFiUg2Nx"
with torch.no_grad():
    hh_bijector = LinearHouseholder(2, 2)
    X_r, ldj = hh_bijector.forward(init_X)
    X_approx = hh_bijector.inverse(X_r)

# check the inverse
torch.testing.assert_allclose(X_approx, init_X, rtol=1e-5, atol=1e-5, )
```

```python id="LEKgmwY4Mk7I"
from survae.transforms.bijections.linear_orthogonal import FastHouseholder, LinearHouseholder
```

<!-- #region id="kn76g_2Yhyvx" -->
## Flow Model


So to have a flow model, we need two components:


---
**Base Distribution**: $p_Z = \mathbb{P}_Z$

This will describe the distribution we want in the transform domain. In this case, we will choose the uniform distribution because we are trying to uniformize our data.

---
**Bijections**: $f = f_L \circ f_{L-1} \circ \ldots \circ f_1$

The list of bijections. These are our functions which we would like to compose together to get our dataset.

<!-- #endregion -->

```python id="5KEvITq4i2PS"
from survae.distributions import StandardUniform, StandardNormal
from survae.flows import Flow

# parameters
features_shape = (2,)
num_mixtures = 4
num_reflections = 2

# base distribution
base_dist = StandardNormal(features_shape)

# transforms
transforms = [
              GaussianMixtureCDF(features_shape, None, num_mixtures),
              InverseGaussCDF(),
              LinearHouseholder(features_shape[0], num_reflections)

]

# flow model
model = Flow(
    base_dist=base_dist,
    transforms=transforms
)
```

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

```python colab={"base_uri": "https://localhost:8080/"} id="Infobxf7nzns" outputId="02e137b8-eea7-4d05-8029-63abfe0c900e"
n_epochs = 50
# logger = TensorBoardLogger("tb_logs", name='mg_no_init')

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

```python colab={"base_uri": "https://localhost:8080/", "height": 312, "referenced_widgets": ["11dae21f0a194c9385e7f9b0645b31f9", "a7e4da5b19f345e6a2cbecc2eeff49bd", "6a5a3d22afe648fd9e57d5233e65ae82", "8053ef0a04bc4687ab6b6d72de6e592f", "a863d07e9a6d4b19b33bdd811a40e32f", "fb6fb2ceabee47a19111c724a40264b1", "2c7aa9e133fa423cba1927be6f4383d8", "aca5f2c7ba494d5e92f2b83f4f879353", "eed426797b1b472aad55c48b7ebd73bd", "d7af6347eea440859efefaaced93f7ca", "3b1ff75598bc42c59a72a1aeecbdfc51"]} id="S8-cz5qxhyIW" outputId="c7cf157a-f8ac-4391-92ce-00cbd78dda8b"
# train model
trainer.fit(learn, )
```

<!-- #region id="ifZDTNsWiZos" -->
## Results
<!-- #endregion -->

<!-- #region id="oTUVxFbZ5Vf0" -->

### Latent Domain
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 367} id="kCDNmj0J5XOE" outputId="7ae7ed0d-4020-4520-ceb3-ffce10df26ed"
with torch.no_grad():
    X_ = torch.Tensor(X)
    # X_ = X_.to(device)
    X_r, ldj = learn.model.forward_transform(X_)


fig = corner.corner(torch_2_numpy(X_r))
```

<!-- #region id="qaMRInWJ5c2L" -->
### Inverse
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 367} id="oyqf_7Gc5dtt" outputId="ccf93fa1-f4e4-46de-a919-b3ad52604bdd"
with torch.no_grad():
    # X_ = X_.to(device)
    X_approx = learn.model.inverse_transform(X_r)

fig = corner.corner(torch_2_numpy(X_approx))
```
