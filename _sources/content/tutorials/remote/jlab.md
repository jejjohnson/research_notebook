# JupyterLab


* Install Jlab (from environment)
* Launch Jlab 
  * Local
  * SLURM
    * Scratch
    * Automation (script, config)
  * Gricad
    * Scratch
    * Automation
* 
  

---
## Organization

1. Create a `project` directory - where our code is living
2. Create a `bin` directory - where we put all of our executables
3. Create `$WORKDIR` - 
4. Create `$LOGSDIR`
5. Create necessary files (`logs`, `jobs`, `errs`)


**Example**

```bash
# ===================
# Custom directories
# ===================
# work directory
export WORKDIR=/mnt/meom/workdir/johnsonj
# log directory
export LOGDIR=$WORKDIR/logs
```

---
**Step 1**: Ensure `$WORKDIR` is set.


Check if it exists in the environments.

```bash
printenv WORKDIR
```

Make sure to add it to the `.bashrc` or `.profile`.

```bash
# add this to the .profile
export WORKDIR=/mnt/meom/workdir/username:$WORKDIR
```

Check again if it exists.

```bash
# check if exists (it should now)
printenv WORKDIR
```


---
**Step 2**: Ensure `$LOGSDIR` is set.

Check if it exists in the environments.

```bash
printenv LOGDIR
```

Make sure to add it to the `.bashrc` or `.profile`.

```bash
# add this to the .profile
export LOGDIR=$WORKDIR/logs
```

Check again if it exists.

```bash
# check if exists (it should now)
printenv LOGDIR
```

---
**Step 3**: Create necessary directories

This is so that we can save logs, errors and job configurations. This will be helpful for automating things later. I like to have these available:


```bash
$LOGDIR/logs
$LOGDIR/jobs
$LOGDIR/errs
```

- `logs` is a subdirectory within logs which will hold all of the slurm log files.
- `errs` - a subdirectory which will hold all of the slurm error log files.
- `jobs` - a subdirectory which will hold all of the current job configurations.

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
- ipykernel
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


---
## JLab on SLURM

1. Activate Environment
2. Start JLab with port-forwarding
3. Create and SSH session.


---
### Automation




---
## JLab on OAR



---
## Automation


1. [ ] Log into server
2. [ ] Run slurm script via `sbatch`.
3. [ ] Create SSH tunnel to allow for `jlab` on local machine.


### SLURM (Cal1)

#### From Scratch

```bash
conda activate jlab
srun -n 1 --time=01:00:00 --mem=1600 --account=python --job-name=jlab jupyter-lab --no-browser --port=3211 --ip=0.0.0.0
```

#### Scripts


```bash
#!/bin/bash

#SBATCH --job-name=jlab                     # name of job
#SBATCH --account=python                    # for statistics
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


### SLURM (JeanZay)


Below we have a `jlab_gpu.slurm` script which is meant to be run on the slurm server.


```bash
#!/bin/bash

#SBATCH --job-name=jlab_gpu          # name of job
#SBATCH --account=cli@gpu
#SBATCH --export=ALL
#SBATCH --nodes=1                    # we request one node
#SBATCH --ntasks-per-node=1          # with one task per node (= number of GPUs here)
#SBATCH --gres=gpu:1                 # number of GPUs (1/4 of GPUs)
#SBATCH --cpus-per-task=10           # number of cores per task (1/4 of the 4-GPUs node)
#SBATCH --hint=nomultithread         # hyperthreading is deactivated
#SBATCH --time=12:00:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --output=$LOGDIR/logs/jlab_gpu.out    # name of output file
#SBATCH --error=$LOGDIR/errs/jlab_gpu.out     # name of error file (here, in common with the output file)

# get tunneling info
XDG_RUNTIME_DIR=""
node=$(hostname -s)
user=$(whoami)
cluster="dahu_ciment"
port=8888

# ==================
# Information 4 SSH
# ==================
# Get the compute node
squeue -u $USER -h | grep jlab_gpu | awk '{print $NF}' > $LOGDIR/jobs/jlab_gpu.node
# get the hostname
hostname -I | awk '{print $1}' > $LOGDIR/jobs/jlab_gpu.ip
# get the username
whoami > $LOGDIR/jobs/jlab_gpu.user

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
jupyter-lab --no-browser --port=${port} --ip=0.0.0.0
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
#### JeanZay (CPU)


---
#### JeanZay (GPU)



---

## Run the jupyter notebook with conda environment (everytime)

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
### Launching JupyterLab Using Jobs


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

3. Close the ssh tunnel when done


```bash
# get the process-ID number for ssh tunneling
lsof -i :portNumber
# kill that process
kill -9 PID
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