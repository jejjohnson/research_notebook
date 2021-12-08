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
# Example - 2D Plane


This is my notebook where I play around with all things normalizing flow with pyro. I use the following packages:

* PyTorch
* Pyro
* PyTorch Lightning
* Wandb
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="hhn17BaNDNPP" outputId="cee79dec-6684-46a6-f61e-812e7c7bc9c9"
#@title Install Packages
# %%capture

!pip install --upgrade --quiet pyro-ppl tqdm wandb corner loguru pytorch-lightning lightning-bolts torchtyping einops plum-dispatch pyyaml==5.4.1 nflows
!pip install --upgrade --quiet scipy
```

```python colab={"base_uri": "https://localhost:8080/"} id="ZHd9hQBx6r5A" outputId="e77655d7-cce5-49b4-c3d6-b4aa8a1b331d"
!git clone https://github.com/jejjohnson/survae_flows_lib.git
!pip install survae_flows_lib/. --use-feature=in-tree-build
```

```python colab={"base_uri": "https://localhost:8080/"} id="5vttXbI6DgSk" outputId="fcb595bb-8722-4fd1-8dee-db7a095f91ac"
#@title Import Packages

# TYPE HINTS
from typing import Tuple, Optional, Dict, Callable, Union
from pprint import pprint

# PyTorch
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from einops import rearrange

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

```python id="iorM0jp8ODHb"
X = get_toy_data(5_000, 123)

```

```python colab={"base_uri": "https://localhost:8080/", "height": 387} id="qKFia9kMOEaE" outputId="f3d089f8-f957-4236-a33c-d4b72f4491ce"
fig = corner.corner(X, color="blue")
fig.suptitle("Sine Wave")
plt.show()
```

<!-- #region id="ZUhElhna-B8f" -->
## Model
<!-- #endregion -->

```python id="bKLdAY9qCZ_U"
from survae.transforms.bijections.elementwise_nonlinear import GaussianMixtureCDF, InverseGaussCDF
from survae.transforms.bijections.linear_orthogonal import LinearHouseholder

from survae.distributions import StandardUniform, StandardNormal
from survae.flows import Flow
import pytorch_lightning as pl
```

<!-- #region id="AwFeXVKRb369" -->
### Naive GF Initialization
<!-- #endregion -->

```python id="qaITuUICCxdL"



    
def init_gf_layers(num_mixtures: int, num_layers: int=5, num_reflections: int=2, **kwargs):

    
    transforms = []

    with trange(num_layers) as pbar:

        for ilayer in pbar:

            # MARGINAL UNIFORMIZATION
            ilayer = GaussianMixtureCDF(shape, num_mixtures=num_mixtures)

            # save layer
            transforms.append(ilayer)

            # ELEMENT-WISE INVERSE GAUSSIAN CDF
            ilayer = InverseGaussCDF()

            # save layer
            transforms.append(ilayer)


            # HOUSEHOLDER TRANSFORM
            ilayer = LinearHouseholder(shape[0], num_householder=num_reflections)

            # save layer
            transforms.append(ilayer)


    return transforms
```

```python colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["302b799190f94b4cb10706d432239544", "4aca89dbeb504a87bd4b7a466a58a39a", "ac1733d07b0e4f92b1640a823f1d2d34", "5b4fea64038a422881ae16d65999783d", "c0d1b842bb4940c6aeb0b15573f580b0", "61e2e7f10739494696d4e026063af010", "b6726be0751a42149e3e72927a6178a0", "27d0031e4a024ae4a0a4b91010c6f5a0", "11340777fdcd49a3b7df23b58b05e448", "1dca1cfa14d5409993ddf6e10d78dcd2", "56aa4a2372d44cffa2b4720b6c178964"]} id="olxMVfTfOvdU" outputId="2c2a61f5-a146-4416-bb59-e3faa6a28945"
shape = (2,)

# base distribution
base_dist = StandardNormal(shape)

# init GF
transforms = init_gf_layers(shape=shape, num_mixtures=6, num_layers=12, num_householder=2)




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
X_train = get_toy_data(10_000, 123)

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
        return torch.optim.Adam(self.model.parameters(), lr=1e-2)

    def train_dataloader(self):
        return train_loader
```

```python id="qDYHXvHjnyZW"
# initialize trainer
learn = Learner2DPlane(model)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Infobxf7nzns" outputId="62632554-5e9b-4491-f1bd-5e36a6f12ab3"
n_epochs = 20
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

```python colab={"base_uri": "https://localhost:8080/", "height": 277, "referenced_widgets": ["d83ef897b8b7449e8b639dec5ba1edbc", "204c156e11cc498d8024437e95468973", "a4ac1d27a01141bca8bbf5f14ca0170f", "5af193090dd24f2895ae38b3ac049652", "cf6dcc27f08a48159f828ce43a3aa007", "691a4cf11e0d4e4ab6adfd0d0d51efe9", "217b0361bafc4d0abffc307007b89142", "8cb5230ee9214374b816ad8fc6c570fd", "7d14af2d0925489abf5a67c8bf18958d", "655338c3585a41fe87f2f4b635f6a76b", "62118daaeb3f4ceea954415420397b83"]} id="S8-cz5qxhyIW" outputId="e4a8c75b-7f6d-4255-d88c-bae4fc29a5b3"
# train model
trainer.fit(learn, )
```

<!-- #region id="ifZDTNsWiZos" -->
## Results
<!-- #endregion -->

<!-- #region id="oTUVxFbZ5Vf0" -->

### Latent Domain
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 367} id="kCDNmj0J5XOE" outputId="7c90eaf4-7d14-40b8-e705-f778bd5942f0"
with torch.no_grad():
    X_ = torch.Tensor(X)
    X_ = X_.to(learn.device)
    X_r, ldj = learn.model.forward_transform(X_)


fig = corner.corner(torch_2_numpy(X_r))
```

<!-- #region id="qaMRInWJ5c2L" -->
### Inverse
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 367} id="oyqf_7Gc5dtt" outputId="44f32161-b064-471a-dd07-9f58689acdc7"
with torch.no_grad():
    # X_ = X_.to(device)
    X_approx = learn.model.inverse_transform(X_r)

fig = corner.corner(torch_2_numpy(X_approx))
```

<!-- #region id="eEGcvC-LP92B" -->
### Samples
<!-- #endregion -->

```python id="btiREZiTP-18"
X_samples = learn.model.sample(5_000)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 379} id="Ti0uifHgQKLC" outputId="adc0f463-d212-40b0-9f8f-98655da89dc9"
fig = corner.corner(torch_2_numpy(X_samples))
```

<!-- #region id="T5y2VQqOP_WD" -->
### Log Probability
<!-- #endregion -->

```python id="wQtYgopKQAQ0"

```

<!-- #region id="YpDz_OSUva-9" -->
## Better Initialization

Notice how we did not actually initialize the layers with the best parameters using the data.
<!-- #endregion -->

```python id="MqfFZZ4ZOOqs"
def init_gf_layers_rbig(shape: Tuple[int], num_mixtures: int, num_reflections: int=2, num_layers: int=5, X=None, **kwargs):

    
    transforms = []

    X = torch.Tensor(X)

    with trange(num_layers) as pbar:

        for ilayer in pbar:

            # MARGINAL UNIFORMIZATION
            ilayer = GaussianMixtureCDF(shape, X=torch_2_numpy(X), num_mixtures=num_mixtures)

            # forward transform
            X, _ = ilayer.forward(X)

            # save layer
            transforms.append(ilayer)

            # ELEMENT-WISE INVERSE GAUSSIAN CDF
            ilayer = InverseGaussCDF()

            # forward transform
            X, _ = ilayer.forward(X)

            # save layer
            transforms.append(ilayer)

            # ELEMENT-WISE INVERSE GAUSSIAN CDF
            ilayer = LinearHouseholder(shape[0], num_householder=num_reflections)

            # forward transform
            X, _ = ilayer.forward(X)

            # save layer
            transforms.append(ilayer)


    return transforms
```

```python colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["76dd37f5bc884b9c83869b50b3441aa8", "af992fa990894743a3effe21eed61b3c", "c38344bebe2947adbf675283027ecd86", "5bcfce53da6d45389bff2aaadf552004", "63bb7a3d8d1542e5b0671cb58c842861", "d1ed04775a3a4d43b995695ec6962dd1", "f34e5e15af8346e9a1077a68059bef74", "0aa90659f4364492acc7806ed2905ef7", "bc37e2f8bd2c4dd59973cacec4a231a7", "23e5f0f5485b411fb2acaba8b563fc01", "2b87159b6f714b22acfee915c241c224"]} id="TDm8TQfFQTKU" outputId="8d6f4ccd-69a1-4b30-b6c4-46886e16f9bf"
shape = (2,)

# base distribution
base_dist = StandardNormal(shape)

# init GF
transforms = init_gf_layers_rbig(shape=shape, X=X, num_mixtures=6, num_layers=12, num_householder=2)




# flow model
model = Flow(
    base_dist=base_dist,
    transforms=transforms
)
```

```python id="jtQH5LVavjLh"



# initialize trainer
learn = Learner2DPlane(model)
```

```python colab={"base_uri": "https://localhost:8080/"} id="3gPiLkYBQc0l" outputId="01574465-ae7c-4823-a16c-03ee9270a3e6"
n_epochs = 20
logger = TensorBoardLogger("tb_logs", name='mg_rbig_init')

# initialize trainer
trainer = pl.Trainer(min_epochs=1, max_epochs=n_epochs, gpus=1, enable_progress_bar=True, logger=logger)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 223, "referenced_widgets": ["c5a7672203ab481d94d0e22b7680c06f", "b7822b81a2d7499c889c8a24859adf52", "2a43cd5ff19949b39c9da2b2e648aed5", "3483c18661cd40a692beb24411e82f56", "4ec72acdb6a64452857ab94cc6ce797c", "262b9afa054d4000bdfdd81cc34675ee", "ad81c30c5b8b4a49a50bca99fcdbf4ae", "cf61ddf9a3704aba913af577f1a0aaf5", "62bea0a6d5c043e3947ca0bc490d608a", "30bd8178b66245f0984a3ea0966305bc", "5c475c72f10c4973b59057d6d6ddac68"]} id="lttpLIhdQfv9" outputId="a46cf3ef-3473-45ae-9b3b-26f471e6b232"
# train model
trainer.fit(learn, )
```

<!-- #region id="JcIujX7MQxvx" -->
## Results
<!-- #endregion -->

<!-- #region id="_HhNCmrnQxvy" -->

### Latent Domain
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 367} id="VvkQCzHLQxvy" outputId="81fc886e-3ddf-4ee5-c53b-977a0af3415f"
with torch.no_grad():
    X_ = torch.Tensor(X)
    X_ = X_.to(learn.device)
    X_r, ldj = learn.model.forward_transform(X_)


fig = corner.corner(torch_2_numpy(X_r))
```

<!-- #region id="IwzFeXp6Qxvy" -->
### Inverse
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 367} id="I-Bg3GV8Qxvy" outputId="f837ecbc-435c-49b2-d6b8-eae06accf995"
with torch.no_grad():
    # X_ = X_.to(device)
    X_approx = learn.model.inverse_transform(X_r)

fig = corner.corner(torch_2_numpy(X_approx))
```

<!-- #region id="wujc9fKMQxvy" -->
### Samples
<!-- #endregion -->

```python id="AT6vio9oQxvy"
X_samples = learn.model.sample(5_000)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 367} id="1ZuH2tBpQxvz" outputId="15f74588-c1a0-44ed-f36d-cddff5bffce4"
fig = corner.corner(torch_2_numpy(X_samples))
```

<!-- #region id="rYvUnHiXQxvz" -->
### Log Probability
<!-- #endregion -->

```python id="HtYuCelLR4b4"
def generate_2d_grid(data: np.ndarray, n_grid: int = 1_000, buffer: float = 0.01) -> np.ndarray:

    xline = np.linspace(data[:, 0].min() - buffer, data[:, 0].max() + buffer, n_grid)
    yline = np.linspace(data[:, 1].min() - buffer, data[:, 1].max() + buffer, n_grid)
    xgrid, ygrid = np.meshgrid(xline, yline)
    xyinput = np.concatenate([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], axis=1)
    return xyinput
```

```python id="Slginl7GQxvz"
# sampled data
xyinput = generate_2d_grid(X, 500, buffer=0.1)
```

```python id="D3dhvWxbRz1Z"
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

```python id="Wmd3BdAiR7yX"
with torch.no_grad():
    X_ = torch.Tensor(xyinput)
    # X_ = X_.to(device)
    X_log_prob = learn.model.log_prob(X_)

X_log_prob = torch_2_numpy(X_log_prob)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 431} id="dvtvOcLzSAUX" outputId="a8236e30-db2c-4a65-b9e9-f36db98b920d"
plot_2d_grid(torch_2_numpy(X), torch_2_numpy(xyinput), torch_2_numpy(X_log_prob))
```
