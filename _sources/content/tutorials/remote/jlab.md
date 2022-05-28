# Jupyter Lab


---
## TOC

- [TOC](#toc)
- [Install JupyterLab](#install-jupyterlab)
  - [Environment `.yaml` file](#environment-yaml-file)
- [JLab on SLURM](#jlab-on-slurm)
  - [Using `srun`](#using-srun)
  - [Using `sbatch`](#using-sbatch)
    - [Slurm Script](#slurm-script)
  - [SLURM (JeanZay)](#slurm-jeanzay)
    - [SSH](#ssh)
    - [GPU](#gpu)
      - [`SRUN`](#srun)
    - [Procedure](#procedure)
    - [CPU](#cpu)
- [JLab on OAR](#jlab-on-oar)
  - [Using `srun`](#using-srun-1)
  - [Using `oarsh`](#using-oarsh)
- [Extensions](#extensions)
  - [Templates](#templates)
- [2. Setup Your JupyterLab Environment](#2-setup-your-jupyterlab-environment)
  - [2.1 Create a `.yml` file with requirements](#21-create-a-yml-file-with-requirements)
  - [2.2 JupyterLab and other python kernels](#22-jupyterlab-and-other-python-kernels)
  - [2.1 Create a Conda Environment](#21-create-a-conda-environment)
  - [2.3 Install and the Jupyterlab Manager](#23-install-and-the-jupyterlab-manager)
  

---
## Install JupyterLab

Here, we will walk-through how we can install `jupyterlab` on the servers.

---
### Environment `.yaml` file

Below, we have a `.yaml` file for creating a Jupyter Lab environment. To refresh yourself on how this works, have a look at the [conda tutorial](./conda.md) to more details on the installation.


```yaml
name: jlab
channels:
- defaults
- conda-forge
dependencies:
- python=3.9
# GUI
- conda-forge::jupyterlab           # JupyterLab GUI
- conda-forge::nb_conda_kernels     # Access to other conda kernels
- conda-forge::spyder-kernels       # Access via spyder kernels
- conda-forge::nodejs               # for extensions in jupyterlab
- conda-forge::jupyterlab-git
- conda-forge::jupyter-server-proxy
- conda-forge::ipywidgets
- pyviz::holoviews
- bokeh::bokeh
- bokeh::jupyter_bokeh              # Bokeh
- ipykernel
- tqdm                              # For status bars
- pip                               # To install other packages
- pip:
  - dask_labextension
```


:::{note} 
We are making a single *JupyterLab environment to rule them all*. So we are only going to install things that are important for JupyterLab as a jlab *base environment*. It isn't necessary to install things like numpy, scipy or even pytorch. We only need to install things like git and server proxies. 

However, we still will be able to change the `conda_kernel`. To ensure that we can change it to *other environments*, we needed to install  `nb_conda_kernel` and `ipykernels` in the `jlab` base environment as well as any other environment that we want access to.
:::

**Step 1**: Install this as a conda environment. (Use mamba because its faster)

```bash
mamba env create -f environment.yaml --prune
```

**Step 2**: Activate environment

```bash
conda activate jlab
```

**Step 3**: Start JupyterLab

```bash
jupyter-lab --no-browser --ip=0.0.0.0 --port=8888
```

This will give you a

:::{admonition} Automation
We can turn this into a script so that we don't have to keep running these same commands. Add this function to your `.profile` or `.bash_profile`.

```bash
# Launch Jupyter Lab
function jlab(){
    # set port (default)
    port=${1:-8888}
    # activate jupyter-lab
    conda activate jlab
    # Fires-up a Jupyter notebook by supplying a specific port
    jupyter-lab --no-browser --ip=0.0.0.0 --port=$port
}
```

**Example Usage**

```bash
jpt
```
:::


---
## JLab on SLURM

1. Activate Environment
2. Start JLab with port-forwarding
3. Create and SSH session.


### Using `srun`

:::{admonition} Automation
We can turn this into a script so that we don't have to keep running these same commands. Add this function to your `.profile` or `.bash_profile`.

```bash
function jlab_srun(){
    # activate conda environment with jlab
    conda activate jlab
    # run jupyterlab via srun
    srun --nodes=1 --mem=1600 --time=8:00:00 --account=python --job-name=jlab jupyter-lab --no-browser --port=3211 --ip=0.0.0.0
}
```

**Example Usage**

```bash
jpt
```
:::

---

### Using `sbatch`

In this case, we will create a script and then launch the job using the `sbatch` command. 

**Pros**:
* It really allows you to customize the nitty gritty details of the compute node environment.
* It launches in the background.

**Cons**: 
* It's not very easy to do. You have to have access to the nitty-gritty details.
* Sometimes you cannot ssh into the compute node.


---

#### Slurm Script

We need to create a bash script, e.g. `jlab_script.sh`, that will hold all of the commands to really customize the note we run.


```bash
#!/bin/bash

#SBATCH --job-name=jlab                     # name of job
#SBATCH --account=python                    # for statistics
#SBATCH --export=ALL                        # export all environment variables
#SBATCH --nodes=1                           # we ALWAYS request one node
#SBATCH --ntasks-per-node=1                 # number of tasks per node
#SBATCH --cpus-per-task=4                   # number of cpus per task
#SBATCH --time=8:00:00                      # maximum execution time requested (HH:MM:SS)
#SBATCH --memory=1600                       # the amount of memory requested
#SBATCH --output=/mnt/meom/workdir/johnsonj/logs/jlab.log      # name of output file
#SBATCH --error=/mnt/meom/workdir/johnsonj/errs/jlab.err       # name of error file 

# get tunneling info
XDG_RUNTIME_DIR=""
node=$(hostname -s)
user=$(whoami)
cluster="ige-meom-cal1"
cluster_ssh="meom_cal1"
port=3211

# ==================
# Information 4 SSH
# ==================
# Get the compute node
squeue -u $USER -h | grep jlab | awk '{print $NF}' > $LOGDIR/jobs/jlab.node
# get the hostname
hostname -I | awk '{print $1}' > $LOGDIR/jobs/jlab.ip
# get the username
whoami > $LOGDIR/jobs/jlab.user

# Tunneling Info
echo -e "
node=${node}
user=${user}
cluster=${cluster}
port=${port}

Command to create ssh tunnel (manually):
ssh -N -f -L ${port}:${node}:${port} ${user}@${cluster}

Command to create ssh tunnel (ssh config):
ssh -N -f -L ${port}:${node}:${port} ${cluster_ssh}

Command to create ssh tunnel through
ssh -N -f -L ${port}:localhost:${port} $

Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"
 
# loading of modules
conda activate jlab

# go into work directory
cd $WORKDIR
# echo of launched commands
set -x
 
# code execution
jupyter-lab --no-browser --port=${port} --ip=0.0.0.0
```


1. Configure a `.sh` script
2. Run the script using `sbatch` command
3. Create and SSH session.

:::{admonition} Automation

```bash
function jlab_sbatch(){
    # Fires up JLab script in bin using sbatch 
    sbatch jlab_sbatch.sh
    # prints
    cat $LOGDIR/slurm/logs/jlab.log
    # prints jlab
    cat $LOGDIR/slurm/errs/jlab.err
}
```
:::







### SLURM (JeanZay)

We have to do the same as above however there are a few different commands we need to take care of which allow for more customization. Furthermore, we now have access to 

#### SSH


```python
sshuttle -dns -HN -r meom_cal1 130.84.132.0/24
```


:::{admonition} Automation
:class: tip

```bash
sshuttle --dns -HN @meom_cal1.conf
```

The `meom_cal1.conf` file looks like this:

```bash
130.84.132.0/24
--remote
meom_cal1
```

:::




#### GPU

Below we have a `jlab_gpu.slurm` script which is meant to be run on the slurm server.


##### `SRUN`

**Run this Command**

```bash
srun --pty --account=cli@v100 --nodes=1 --ntasks-per-node=1 --cpus-per-task=10 --gres=gpu:1 --hint=nomultithread --time=01:30:00 bash
```

**Activate Bash shell**

```bash
eval "$(conda shell.bash hook)"
```

**Change Environment**

```bash
conda activate jlab
```




```bash
#!/bin/bash

#SBATCH --job-name=jlab_gpu             # name of job
#SBATCH --account=cli@v100              # GPU Account
#SBATCH --export=ALL                    # export all environment variables
#SBATCH --nodes=1                       # we request one node
#SBATCH --ntasks-per-node=1             # with one task per node (= number of GPUs here)
#SBATCH --gres=gpu:1                    # number of GPUs (1/4 of GPUs)
#SBATCH --cpus-per-task=10              # number of cores per task (1/4 of the 4-GPUs node)
#SBATCH --hint=nomultithread            # hyperthreading is deactivated
#SBATCH --time=8:00:00                  # maximum execution time requested (HH:MM:SS)
#SBATCH --output=/gpfswork/rech/cli/uvo53rl/logs/slurm/logs/jlab_gpu.out    # name of output file
#SBATCH --error=/gpfswork/rech/cli/uvo53rl/logs/slurm/errs/jlab_gpu.out     # name of error file

# get tunneling info
XDG_RUNTIME_DIR=""
node=$(hostname -s)
user=$(whoami)
cluster="dahu_ciment"
port=3212

# ==================
# Information 4 SSH
# ==================
# Get the compute node
squeue -u $USER -h | grep jlab_gpu | awk '{print $NF}' > $SLURMDIR/jobs/jlab_gpu.node
# get the hostname
hostname -I | awk '{print $1}' > $SLURMDIR/jobs/jlab_gpu.ip
# get the username
whoami > $SLURMDIR/jobs/jlab_gpu.user

# cleans out the modules loaded in interactive and inherited by default 
module purge

# Tunneling Info
echo -e "
node=${node}
user=${user}
cluster=${cluster}
port=${port}

Command to create ssh tunnel:
ssh -N -f -L ${port}:${node}:${port} ${user}@${cluster}

Command to create ssh tunnel through
ssh -N -f -L ${port}:localhost:${port} $

Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"
 
# loading of modules
module load git
module load cuda/10.2
source activate jlab
 
# echo of launched commands
set -x
 
# code execution
# jupyter-lab --no-browser --port=${port} --ip=0.0.0.0
idrlab --notebook-dir=/gpfswork/rech/cli/uvo53rl
```

:::{note}
make sure that it is executable via the `chmod +x script.slurm` command.
:::


#### Procedure

**Step I**: Run the bash script

```bash
sbatch /path/to/script.slurm
```

**Step 2**: Check


```bash
# JUPYTER NOTEBOOK STUFF
function launch_jlab(){
    # Fires up JLab using sbatch
    sbatch jlab.slurm
    # prints
    cat $LOGDIR/logs/jlab.log
}
```






---
#### CPU


---

## JLab on OAR


:::{warning}
We can't actually launch jupyter-lab on the head node. Well, technically we can but it will kill the process almost immediately. You must use the compute nodes to do all computation. 
:::

---
### Using `srun`


First, take a look at this tutorial to get familiar: https://gricad-doc.univ-grenoble-alpes.fr/notebook/hpcnb/

Below offers a simpler work structure.
 
**In the First Terminal** - Start the JupyterLab session

1. Log into your server


```bash
ssh f-dahu.ciment
```

2. Start an interactive session

```bash
oarsub -I --project data-ocean -l /core=10,walltime=2:00:00 -> it will log automatically on a login node dahuX
```

3. Activate your conda environment with JupyterLab

```
conda activate pangeo
```

4. Start your jupyterlab session


```bash
jupyter notebook/lab
```


**Tip**: It's highly advised to open a session using tmux or screen when running these commands. This will allow you to have things running in the background even if you're not logged into the server.


**In the second terminal** - we will do the ssh tunneling procedure to view jupyterlab on your local machine.

1. Do the Tunneling

```bash
ssh -fNL 8888:dahuX:8888  [ -L 8686:dahuX:8686 for the dashboard] dahu.ciment
```

2. Open `http://localhost:8888/?token=...(see the result of the jupyter notebook command)` on a browser in your laptop.


3. When you're done, make sure you close the tunnel you opened.



```bash
# get the process-ID number for ssh tunneling
lsof -i :portNumber
# kill that process
kill -9 PID
```






---
### Using `oarsh`


Sometimes, it's a bit annoying to have to keep track of everything (launching interactive job + run jupyter notebook/lab + create tunnel, etc). So below is a way to create a simple script that helps automate the process a little bit.

Firstly, we need a bash script which can easily be launched. Below is an example.

<details>
<summary>
jlab_bash.sh
</summary>

```bash
#!/bin/bash

#OAR --name jlab_cpu
#OAR --project pr-data-ocean
#OAR -l /nodes=1,walltime=10:00:00
#OAR --stdout /bettik/username/logs/jupyterlab_cpu.log
#OAR --stderr /bettik/username/errs/jupyterlab_cpu.err

# get tunneling info
XDG_RUNTIME_DIR=""
node=$(hostname -s)
user=$(whoami)
cluster="dahu_ciment"
port=3008

# print tunneling instructions jupyter-log
echo -e "
# Tunneling Info
node=${node}
user=${user}
cluster=${cluster}
port=${port}
Command to create ssh tunnel:
ssh -N -f -L ${port}:${node}:${port} ${cluster}
Command to create ssh tunnel through
ssh -N -f -L ${port}:localhost:${port} $
Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"


# load modules or conda environments here with jupyterlab
source activate jlab

jupyter lab --no-browser --ip=0.0.0.0 --port=${port}
```
</details>

**Note I**: make sure you create the `/logs/` and the `/errs` directory so that you can get the command to run the tunneling procedure as well as be able to debug if the script crashes for whatever reason.


---

Now here are the instructions for using it:


1. Launch the script with `oarsub`

```bash
# set permissions for the script (DO THIS ONE TIME)
chmod +x /path/to/bash/script/bash_script.sh
# launch the script
oarsub -S /path/to/bash/script/bash_script.sh
```

2. Open Port with the server name


```bash
# create a tunnel
ssh -N -f -L port_number:node_name:port_number server_name
```


---
## Extensions


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



---

## 2. Setup Your JupyterLab Environment

**Note**: 

- You only have to do this once!
- Make sure conda is already installed.

### 2.1 Create a `.yml` file with requirements

```yaml
name: jupyterlab
channels:
- defaults
- conda-forge
dependencies:
- python=3.8
# GUI
- conda-forge::jupyterlab           # JupyterLab GUI
- conda-forge::nb_conda_kernels     # Access to other conda kernels
- conda-forge::spyder-kernels       # Access via spyder kernels
- conda-forge::nodejs               # for extensions in jupyterlab
- pyviz::holoviews
- bokeh
- bokeh::jupyter_bokeh              # Bokeh
- tqdm                              # For status bars
- pip                               # To install other packages
- pip:
  - ipykernel
  - ipywidgets
  - jupyter-server-proxy
  - dask_labextension
  - nbserverproxy
```

I've typed it out but you can also check out the file on [github](https://github.com/jejjohnson/dot_files/blob/master/jupyter_scripts/jupyterlab.yml).

### 2.2 JupyterLab and other python kernels

So you may be wondering if we need to do this with every conda environment we create. No. We just need to have a general JupyterLab environment that calls other environments. The important thing here is that we have the jupyterlab package installed as well as `nb_conda_kernels` package. This allows the jupyterlab to be able to use any other python kernel that's in your user space (sometimes common shared ones but it depends). 

Now, all other conda environments will need to have the `ipykernel` package installed and it will be visible from your JupyterLab environment.

### 2.1 Create a Conda Environment

```bash
# create the environment with your .yml file
conda env create --name jupyterlab -f jupyterlab.yml
# activate the environment
source activate jupyterlab
# or perhaps
# conda activate jupyterlab
```

### 2.3 Install and the Jupyterlab Manager

This will enable you to have extensions for your Jupyterlab. There are so many cool ones out there. I'm particularly fond of the [variable inspector](https://github.com/lckr/jupyterlab-variableInspector) and the [table of contents](https://github.com/jupyterlab/jupyterlab-toc). JupyterLab has gotten awesome so you can install most new extensions using the JupyterLab GUI.

```yaml
# Install jupyter lab extension maager
jupyter labextension install @jupyter-widgets/jupyterlab-manager
# Enable
jupyter serverextension enable --py jupyterlab-manager
```