# Conda 4 Remote Servers

It's advisable to create personal conda environments in your `/path/to/workdir` directory. Preconfigured environments are great for quickly prototyping. But, inevitably, you always end up needing more fine-tune control over your environments whenever you are prototyping. In addition, too many people using the same environment results in massive environments with unnecessary packages. Having control over your environment will lead to more reproducible settings especially if you keep track.

In this tutorial we will learn how to:

1. Install miniconda
2. Create personal conda environments
3. JupyterLab Environments
4. Preconfigured environments (server specific)


---
## Preconfigured Conda Environments 

Conda is already installed. You just need to activate it using the command from the server.


#### Gricad

There is a command which will activate the conda environment already preconfigured.

```
source /applis/environments/conda.sh
```



#### JeanZay

The conda environments should already be available within the server modules. You just need to activate it with the following command:

```bash
module load Anaconda
```

Now you should have the preconfigured environments.

```
conda activate env
```

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

Make sure you install it in the correct directory. The first option should always be the `homedir`. However, many times the `homedir` on servers are extremely small. This is a problem because some conda environments can be kind of heavy. So the second option should be the `workdir`. You need to change this when doing the installation; there will be an option which allows one choose which directory to install the conda package.

```bash
# some prompt should appear in the installation
/path/to/workdir
```

Now you should have the conda directory. Continue to do the installation by following the steps given by the prompt and then you should be done. Be sure to restart the terminal (e.g. log out and log back in, rerun the `.profile` and/or `.bashrc`). And your prompt should have an indicator like so:

```bash
(base) user@server:prompt$
```

The `(base)` is a message that lets you know your personal conda environment is active. You can also check to see where it is located:

```bash
conda env list
```

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
+ conda remove --name myenv --all
```

**Note**: you only should change the regions where we are *creating*/*removing* environments and  *installing packages* within environments. All other commands should use conda.

---

### Creating from preconfigured

Often times we have a preconfigured environment. This allows us to reproduce the conda environments. This comes in the form of an `environment.yml`

```bash
mamba env create -f environment.yml --prefix=/path/to/workdir/.conda/envs/env_name
conda activate env_name
```

Typically we have the `environment.yml`





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
#### Example Environments

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

<details>
<summary>General Earth Science Packages</summary>

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

</details>



---

### Multiple Conda directories

Sometimes there may be multiple directories where there are packages available. We have our primary `miniconda3` installation but we also want to have access to the other external environments by other people. We simply need to change the `.condarc` script to include all of the directories which have relevant environments.

```bash
envs_dirs:
    - /path/to/workdir/.conda/envs
    - /path/to/otherdir/.conda/envs
```

You can add as many directories as you want. This just ensures that conda can talk to it. However, the more directories you have, the longer it takes for conda/mamba to spider through all of them.

**Example**: In the jean-zay server, there are quite a lot of preconfigured environments located in following directory: `/gpfslocalsup/pub/anaconda-py3/2021.05/envs/`. So by adding this to the `.condarc` file, we now have access to all of the environments they have already configured! So we can use them but not necessarily if we don't want to. :D



