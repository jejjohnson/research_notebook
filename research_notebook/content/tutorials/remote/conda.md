# Conda 4 Remote Servers


It's advisable to create personal conda environments in your `/path/to/workdir` directory. Preconfigured environments are great for quickly prototyping. But, inevitably, you always end up needing more fine-tune control over your environments whenever you are prototyping. In addition, too many people using the same environment results in massive environments with unnecessary packages. Having control over your environment will lead to more reproducible settings especially if you keep track.

Look at the table of contents to see what could be of interest. If it's your first time, I would suggest going through all of it first before you start tinkering.

- [Install Miniconda](#install-miniconda)
  - [Testing](#testing)
  - [Install Mamba](#install-mamba)
- [Creating from `environment.yml`](#creating-from-environmentyml)
  - [Updating Env with `environment.yml`](#updating-env-with-environmentyml)
- [Example Environments](#example-environments)
- [Multiple Conda directories](#multiple-conda-directories)
- [Server Installations](#server-installations)
  - [Gricad](#gricad)
  - [JeanZay](#jeanzay)

---
## Install Miniconda


Get the appropriate link via [this webpage](https://docs.conda.io/en/latest/miniconda.html). 

```bash
wget link_to_miniconda_installer
```

This will download a bash script. Then you need to run this bash script to install the

```bash
# change permissions to make it executable
chmod +x path/to/file.sh

# run as bash script
bash .path/to/file.sh
```

:::{warning}
Make sure you install it in the correct directory. Typically we all have a `homedir`, a `workdir` and a `scratchdir`. The first option should always be the `homedir`. However, many times the `homedir` on servers are extremely small. This is a problem because some conda environments can be kind of heavy. So the second option should be the `workdir`. You need to change this when doing the installation; there will be an option which allows one choose which directory to install the conda package. The last option would be the `scratchdir` of course but these are typically erased more frequently. So it makes sense to avoid this if possible.
:::


```bash
# some prompt should appear in the installation
/path/to/workdir
```

Now you should have the conda directory. Continue to do the installation by following the steps given by the prompt and then you should be done. 

:::{note}
Make sure you initialize your miniconda installation so that it adds the appropriate stuff to your `.profile` or `.bashrc`.

```bash
conda init
```
:::

:::{note}
Be sure to restart the terminal (e.g. log out and log back in, rerun the `.profile` and/or `.bashrc`). And your prompt should have an indicator like so:

```bash
(base) user@server:prompt$
```

The `(base)` is a message that lets you know your personal conda environment is active. You can also check to see where it is located:

```bash
conda env list
```
:::



### Testing

If done correctly, you should be able to see the following:

```bash
# conda environments:
#
base     *  /path/to/workdir/miniconda3
```

And now you should be able to create new environments and install new packages.


```bash
# Create environment from scratch
conda create --name myenv python=3.9
# use mamba to create an env from a yaml file
conda env create -f file.yml -n myenv
# activate the created environment
conda activate myenv
# install packages in the environment
conda install numpy scipy matplotlib pandas xarray
```

And if you check using `conda env list`, you should see a new environment listed:

```bash
# conda environments:
#
base        /path/to/workdir/miniconda3
myenv    *  /path/to/workdir/.conda/envs/myenv
```


---
### Install Mamba

Conda (currently) is quite slow to install things. Sometimes it hangs for quite a long time for no apparent reason. So I recommend using `mamba`. You need to install mamba in the base environment. You can find more instructions [here](https://github.com/mamba-org/mamba).

```bash
conda install mamba --name base --channel conda-forge
```

Now you should be able to create, install and remove packages via the `mamba` command.

```diff
# Create environment from scratch
- conda create --name myenv python=3.9
+ mamba create --name myenv python=3.9
# use mamba to create an env from a yaml file
- conda env create -f file.yml -n myenv
+ mamba env create -f file.yml -n myenv
# activate the created environment
conda activate myenv
# install packages in the environment
- conda install numpy scipy matplotlib pandas xarray
+ mamba install numpy scipy matplotlib pandas xarray
# deactivate environment
conda deactivate
# remove environment
- conda remove --name myenv --all
+ mamba remove --name myenv --all
```

**Note**: you only should change the regions where we are *creating*/*removing* environments and  *installing packages* within environments. All other commands should use conda.

---

## Creating from `environment.yml`

Often times we have a preconfigured environment. This allows us to reproduce the conda environments. This comes in the form of an `environment.yml`

```bash
mamba env create -f environment.yml --prefix=/path/to/workdir/.conda/envs/env_name
conda activate env_name
```

Typically we have the `environment.yml`




---
### Updating Env with `environment.yml`

We can also update an existing environment with the packages within the same (or similar) `environment.yml` file. This happens when we may have updated the `environment.yml` file (externally) and we cannot remember which packages we installed or not. 

```bash
mamba env update -f environment.yml --prefix=/path/to/workdir/.conda/envs/env_name
```

This will install any extra packages that are located within the `environment.yml` however **it does not** remove any packages that are already within the `env_name`. If you wish to update the `env_name` with the packages listed in the `environment.yml` **and** remove any excess packages, use the `--prune` flag.

```bash
mamba env update -f environment.yml --prefix=/path/to/workdir/.conda/envs/env_name --prune
```



---
## Example Environments

As I mentioned above, it is useful (and advisable) to install your conda environments using `.yaml` files. This ensures that they are reproducible and it's also easier to install.

As mentioned before, to install the first time, you can use the following command:

```bash
conda env create --file environment.yaml
```

If you already have an environment but you would like to update the environment, use this command:

```bash
conda env update --file environment.yaml
```

**Tip 1**: The `--prune` command ensures that you remove any packages that aren't within the `.yaml` file.


```bash
mamba env update --file environment.yaml --prune
```

**Tip 2**: The `--prefix` allows you to add these packages to an environment with a different name `.yaml` file.


```bash
mamba env update --file environment.yaml --prefix "path/to/env"
```

Below I have included yaml file for using conda and general Earth science packages.



```yaml
name: earthsci_py39
channels:
- defaults
- conda-forge
dependencies:
- python=3.9
# Standard Libraries
- numpy             # Numerical Linear Algebra
- scipy             # Scientific Computing
- xarray            # Data structures
- pandas            # Data structure
- scikit-learn      # Machine Learning
- scikit-image      # Image Processing
- statsmodels       # Statistical Learning
- pymc3             # Probabilistic programming library
# Plotting
- matplotlib
- seaborn
- bokeh
- plotly::plotly>=4.6.0
- pyviz::geoviews
- conda-forge::cartopy
- datashader
- conda-forge::cmocean
- pyviz::hvplot
- conda-forge::xmovie
# Geospatial packages
- geopandas
- conda-forge::regionmask
- conda-forge::xesmf
- conda-forge::xcube
- conda-forge::rioxarray
- conda-forge::shapely
- conda-forge::pooch
- conda-forge::cftime
- conda-forge::pyinterp
# Scale
- numba
- dask              # Out-of-Core processing
- dask-ml           # Out-of-Core machine learning
# Storage
- hdf5              # standard large storage h5
- conda-forge::zarr
# GUI
- conda-forge::papermill
- conda-forge::nb_conda_kernels     # Access to other conda kernels
- conda-forge::nodejs               # for extensions in jupyterlab
- conda-forge::tqdm   
- ipykernel                         # IMPORTANT: allows other environments to see this environment  
- conda-forge::tqdm             # progress bar  
- pip
- pip:
  # Jupyter
  - ipywidgets
  # Formatters
  - black
  - pylint
  - isort
  - flake8
  - mypy
  - pytest
  # Notebook stuff
  - pyprojroot
  # Extra
  -"git+https://github.com/swartn/cmipdata.git"
  - emukit
  - netCDF4
  - shapely
  - affine
  - netCDF4
  - joblib  # Embarssingly parallel
```

:::{adomination} Other Example `environment.yaml`
:class: tip
If you want to see more examples of conda `environment.yml` files, you can check out my [`dot_files` repo](https://github.com/jejjohnson/dot_files). I have plenty of examples and I even distinguish between operating systems like Linux and MacOS.
:::


:::{adomination} Visible Python Kernels 
:class: tip
If you use jupyter-lab a lot, then it is best to try and have all of your conda kernels be visible. This can be done by installing `nb_conda_kernel` and `ipykernels`. So if you run a different environment with jupyter-lab, you should be able to select python kernels from other conda environments.
:::

---

## Multiple Conda directories

Sometimes there may be multiple directories where there are packages available. We have our primary `miniconda3` installation but we also want to have access to the other external environments by other people. We simply need to change the `.condarc` script to include all of the directories which have relevant environments.

```bash
envs_dirs:
    - /path/to/workdir/.conda/envs
    - /path/to/otherdir/.conda/envs
```

You can add as many directories as you want. This just ensures that conda can talk to it. However, the more directories you have, the longer it takes for conda/mamba to spider through all of them.



---
## Server Installations

Often times, `miniconda`/`conda` is already installed. You just need to activate it using the command from the server. However, the above steps allow us to have access to our own miniconda installer which gives us fine-grain control. However, *we can still access* all of the created conda environments by simply adding all of these to the `.condarc` that we showed above. Below are a few servers that I personally have access to and the filenames.

---
### Gricad

For the `gricad` server, there are a few preconfigured environments available. Most of them are for GPU computation so it will be useful for the `bigfoot` cluster. All of the common conda environments are located in the following directory.

```bash
/applis/common/miniconda3/envs
```

So add this to the `.condarc` file as shown above. Now we now have access to all of the environments they have already configured! So we can use them but not necessarily if we don't want to. :)


---
### JeanZay


In the jean-zay server, there are quite a lot of preconfigured environments. Mainly for GPU computation. They are located in following directory: 

```bash
/gpfslocalsup/pub/anaconda-py3/2021.05/envs
```

So by adding this to the `.condarc` file.

Again, now we now have access to all of the environments they have already configured! So we can use them but not necessarily if we don't want to. :D


---

```bash
#!/bin/bash
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh"
MINICONDA_PREFIX="$SCRATCH/miniconda"

install_miniconda(){
  if [ ! -d $SCRATCH/miniconda ]; then
    echo "Installing Miniconda"
    wget $MINICONDA_URL -O $WORK/downloads/miniconda.sh
    bash $WORK/downloads/miniconda.sh -b -p $MINICONDA_PREFIX
    conda init
    eval "$($MINICONDA_PREFIX/condabin/conda shell.bash hook)"
    conda install -y mamba -c conda-forge
    install_mamba
  else
    echo "Miniconda already installed"
  fi

}

install_mamba(){
  eval "$($MINICONDA_PREFIX/condabin/conda shell.bash hook)"
  conda install -y mamba -c conda-forge
}

clone_dotfiles(){
  rm -rf $WORK/projects/dot_files
  git clone https://github.com/jejjohnson/dot_files.git $WORK/projects/dot_files/
}

install_mamba_jlab(){
	wget https://raw.githubusercontent.com/jejjohnson/dot_files/master/jupyter_scripts/jupyterlab.yml -O $WORK/downloads/jlab.yaml
	eval "$($MINICONDA_PREFIX/condabin/conda shell.bash hook)"
	mamba env create -f $WORK/downloads/jlab.yaml
}

install_mamba_dl(){
	install_mamba_jax
  install_conda_pytorch
  install_conda_tensorflow
}

install_mamba_jax(){
	wget https://raw.githubusercontent.com/jejjohnson/dot_files/main/jupyter_scripts/jupyterlab.yaml -O $WORK/downloads/jlab.yaml
	eval "$($MINICONDA_PREFIX/condabin/conda shell.bash hook)"
	conda env create -f $WORK/downloads/jlab.yaml
}

install_conda_pytorch(){
	wget https://raw.githubusercontent.com/jejjohnson/dot_files/main/jupyter_scripts/jupyterlab.yaml -O $WORK/downloads/jlab.yaml
	eval "$($MINICONDA_PREFIX/condabin/conda shell.bash hook)"
	conda env create -f $WORK/downloads/jlab.yaml
}

install_conda_tensorflow(){
	wget https://raw.githubusercontent.com/jejjohnson/dot_files/main/jupyter_scripts/jupyterlab.yaml -O $WORK/downloads/jlab.yaml
	eval "$($MINICONDA_PREFIX/condabin/conda shell.bash hook)"
	conda env create -f $WORK/downloads/jlab.yaml
}


init_conda_env(){
	wget https://raw.githubusercontent.com/quentinf00/dotfiles/main/conda/base_environment.yaml -O conda_ide.yaml
	eval "$($MINICONDA_PREFIX/condabin/conda shell.bash hook)"
	conda install -y mamba -c conda-forge
	mamba env update -f conda_ide.yaml
	rm -f ~/.zshrc ~/.bashrc
	conda init && mv ~/.bashrc ~/.condainitrc
	conda init zsh && mv ~/.zshrc ~/.condainitzshrc
}



reinstall_everything(){
        rm -rf $MINICONDA_PREFIX
        bash ~/miniconda.sh -b -p $MINICONDA_PREFIX
        eval "$($MINICONDA_PREFIX/condabin/conda shell.bash hook)"
        conda install -y mamba -c conda-forge
        mamba env update -f conda_ide.yaml
        mamba env create -f jlab.yaml
        conda init
}

```