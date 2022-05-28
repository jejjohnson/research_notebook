# Jupyter Lab Xtras


---
## Notebooks + Scripts


### Environment 

```python
import pyprojroot
```

### Reloading

```
%load_ext autoreload
%autoreload 2
```


---
## Other Environments


So you may be wondering if we need to do this with every conda environment we create. No. We just need to have a general JupyterLab environment that calls other environments. The important thing here is that we have the jupyterlab package installed as well as `nb_conda_kernels` package. This allows the jupyterlab to be able to use any other python kernel that's in your user space (sometimes common shared ones but it depends). 

Now, all other conda environments will need to have the `ipykernel` package installed and it will be visible from your JupyterLab environment.


---
## Extensions



### Installation

This will enable you to have extensions for your Jupyterlab. There are so many cool ones out there. I'm particularly fond of the [variable inspector](https://github.com/lckr/jupyterlab-variableInspector) and the [table of contents](https://github.com/jupyterlab/jupyterlab-toc). JupyterLab has gotten awesome so you can install most new extensions using the JupyterLab GUI.

```yaml
# Install jupyter lab extension maager
jupyter labextension install @jupyter-widgets/jupyterlab-manager
# Enable
jupyter serverextension enable --py jupyterlab-manager
```


### My Favourite Extensions


#### Templates

### Templates

**Google Colab**

**Default**

```python
import sys, os
from pyprojroot import here
root = here(project_files=[".here"])
sys.path.append(str(here()))

import pathlib

# standard python packages
import xarray as xr
import pandas as pd
import numpy as np

from tqdm import tqdm

# NUMPY SETTINGS
import numpy as onp
onp.set_printoptions(precision=3, suppress=True)

# MATPLOTLIB Settings
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

# LOGGING SETTINGS
import sys
import logging
logging.basicConfig(
    level=logging.INFO, 
    stream=sys.stdout,
    format='%(asctime)s:%(levelname)s:%(message)s'
)
logger = logging.getLogger()
#logger.setLevel(logging.INFO)

%load_ext autoreload
```

