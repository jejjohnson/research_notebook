# Jupyter Lab 4 Remote Servers

---
## Intro

Jupyter Lab (`jlab`) is one of the most popular IDEs for science, data science and machine learning. Firstly, it offers an interactive development environment that greatly speeds up the coding process. Additionally, it also has a lower barrier to entry so many beginners can get started right away. Finally, also serves as an amazing method to document the thought process as it allows others to "walk-through" your code with text, equations, code blocks and visualizations.

Unfortunately/Fortunately, data is getting bigger and computation is getting expensive. So nowadays, most people are **required** to use some sort of lab server `jlab` because laptops do not have the capacity nor the bandwidth to process such large volumes of data efficiently. There exists some built-in solutions like [`JupyterHub`]() which just needs a log-in and then your whole Jupyter suite is available. But, I find that they do not offer the necessary level of granularity that is required for many researchers; especially researchers that really want to make their products reproducible. For example, often times it requires images that are spun up which means the users don't have control over their python environments and also have to keep re-loading their scripts. This is cumbersome and doesn't promote good practices outside of simple prototyping. You could [argue for](https://www.youtube.com/watch?v=9Q6sLbz37gk) or [argue against](https://www.youtube.com/watch?v=7jiPeIFXb6U) the notion that Jupyter notebooks are good enough to do good coding practices and there are [certainly tools](https://nbdev.fast.ai/tutorial.html) to get around that. I personally find that the ecosystem doesn't lend itself well to that kind of development.

This guide is meant to demonstrate how one can setup `jlab` oneself. Setting `jlab` up by yourself is definitely more work than the prebuilt solutions. However, once you know how to do (and you know how to automate some of the tedious commands), these skills are transferable to many different setups. As long as there is a remote server and you can have an ssh connection, you can have a `jlab` environment.

---
## TLDR


Below is **What You Will Do** in this tutorial:

* Install JupyterLab on a remote server
* Learn how to launch jobs on SLURM and OAR
* Automate some things to make life easier (e.g. `bash` scripts, `tmux`, `tmuxp`)


:::{admonition} MEOM-ers
:class: info

For my current lab (2022), we have access to 3 servers. Each of them correspond to a server type with varying levels of power and customizability.

* `cal1` -> SLURM
* `gricad` -> OAR
* `jean-zay` -> SLURM

:::

---

**Table of Contents**
- [Intro](#intro)
- [TLDR](#tldr)
- [Install JupyterLab](#install-jupyterlab)
  - [Environment `.yaml` file](#environment-yaml-file)
- [Running JupyterLab](#running-jupyterlab)
  - [Advanced Users](#advanced-users)
    - [Bash Script](#bash-script)
    - [`tmuxp`](#tmuxp)
- [JLab on Remote Server](#jlab-on-remote-server)
  - [Advanced Users](#advanced-users-1)
    - [SSH Config](#ssh-config)
    - [**TMUXP** Config](#tmuxp-config)
    - [Bash Script](#bash-script-1)
- [JLab on SLURM](#jlab-on-slurm)
  - [Using `srun`](#using-srun)
  - [Using `sbatch`](#using-sbatch)
  - [Advanced Users](#advanced-users-2)
    - [Demo Setup](#demo-setup)
    - [**TMUXP**](#tmuxp-1)
  - [SLURM on JeanZay](#slurm-on-jeanzay)
    - [SSH](#ssh)
      - [Advanced Usage](#advanced-usage)
    - [Advanced Users](#advanced-users-3)
      - [Config](#config)
      - [**TMUXP**](#tmuxp-2)
- [JLab on OAR](#jlab-on-oar)
  - [Using `oarsub`](#using-oarsub)
  - [Using `oarsh`](#using-oarsh)
  

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

---
## Running JupyterLab

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
### Advanced Users

This will allow users to not have to go through all of those above steps each time we want to log in. Instead, we will create a function that automates this for us which will greatly reduce the amount of steps. We will also

#### Bash Script

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

Add this function to your `.profile` or `.bash_profile`.

**Example Usage**

Since everything is within the function, we just need to run the following command:

```bash
# activate conda environment and start jlab
jlab 8888
```

**That was easy**! Just a little bit of customization goes a long way.

#### `tmuxp`

This is the demo for `tmuxp` users.

Below is my demo **`jlab.yaml`** file which has the automated commands for `tmuxp`.

```bash
session_name: cal1_jlab
windows:
  - window_name: jlab
    panes:
    - shell_command: |
        cd $WORKDIR
        jlab 8888
```


**Demo Usage**:

```bash
tmuxp load $HOME/.tmuxp/jlab.yaml
```

As you can see, it is **super simple** to use! Just this command, we get a nice `tmux` window with `jlab` running and even an extra window for tinkering. But of course, we need to add a function to our local `.profile` so that we don't have to type all of this out.

```bash
function tmux_jlab(){
  tmuxp load $HOME/.tmuxp/jlab.yaml
}
```

**Demo Usage**:

```bash
tmux_jlab
```

**Even easier**! Now we just type this command and we get a full tmux session where the ssh command has been run and the jlab session should already be activated on the remote server. We just need to go to `localhost:8888` in our browser and we are good to go.



---
## JLab on Remote Server

So the **only difference** between what we did above and what we do now

So the actual commands are quite similar. But they have different steps:

```diff
# log into remote server
+ ssh username@server -L 8888:localhost:8888
# launch jlab via script
jlab 8888
```

---
### Advanced Users

There are some shortcuts we can apply based on our previous stuff. We will do the following:

1. Setup the `.ssh/config` to streamline the `ssh` flags
2. Use `tmuxp` to streamline the `ssh` and `jlab` execution
3. Write a `bash` script to execute the `tmuxp` commands

---
#### SSH Config

If we have the server information already preconfigured in our `.ssh/config` file, then we can remove a lot of these redundant commands. 

```bash
Host server_name
    HostName server
    User username
    # Allow for Ids
    ForwardX11 yes
    # SSH Identity File
    IdentityFile ~/.ssh/id_key_file
    # Jupyter Notebooks
    LocalForward 8888 localhost:8888
```

The most important are:
* `Host` - makes an easier to remember ssh command
* `IdentityFile` - removes the need to authenticate
* `LocalForward`/`localhost` - does the tunneling automatically

Take a look at the changes when we have this setup. 

```diff
# log into remote server
- ssh username@server -L 8888:localhost:8888
+ ssh server_name
# launch jlab via script
jlab 8888
```

**Note**: We could also remove the exact port number since this port number is configured to be the same within the `.ssh/config` file. So the result is the following command:

```bash
ssh server_name
jlab
```

**Voila**! Super simple! :)


---
#### **TMUXP** Config

We can create a `tmux` environment where we create windows to run commands where everything has already been executed. Below is my demo **`jlab_cal1.yaml`** file which has the automated commands for `tmuxp`.

```bash
session_name: cal1_jlab
windows:
  - window_name: jlab
    panes:
    - shell_command: |
        ssh meom_cal1
        cd $WORKDIR
        sbatch $HOMEDIR/bin/jlab_sbatch.sh

  - window_name: git
    panes:
    - shell_command: ssh meom_cal1
```

**Demo Usage**:

```bash
tmuxp load jlab_cal1.yaml
```

As you can see, it is **super simple** to use! Just this command, we get a nice `tmux` window with `jlab` running and even an extra window for tinkering.


---
#### Bash Script

Now, the final step, we want to not have to write all of those commands. So we create a nice little bash script that does everything. Remember to add it to your `.profile` and restart your terminal.

```bash
function jlab_cal1(){
  tmuxp load $HOME/.tmuxp/jlab_cal1.yaml
}
```

**Demo Usage**:

```bash
jlab_cal1
```

**It's as easy as it gets!**


---
## JLab on SLURM

In this section, we will take what we did above and apply it to a cluster managed by SLURM. The biggest difference in steps is that we first need to log into the remote server, then we need to start a compute node, and then we can apply the steps for the jlab:

```diff
# ssh into the remote server
+ ssh username@server -L 8888:localhost:8888
# start an interactive compute session
+ srun ... or + sbatch
# start jupyter lab environment
conda activate jlab
# Fires-up a Jupyter notebook by supplying a specific port
jupyter-lab --no-browser --ip=0.0.0.0 --port=8888
```

Now we can open our browser to `localhost:8888` and we will have access to `jlab`. We have two ways to do this: 1) we can use `srun` which will start an interactive node or 2) `sbatch` which will start a node based on a `bash` script with the configurations. We will go over both options below.


---
### Using `srun`


```bash
# log into remote server
ssh username@server -L 8888:localhost:8888
# activate conda environment
conda activate jlab
# start an interactive node (apply configs)
srun --nodes=1 --cpus-per-task=8 --mem=1600 --account=python --pty jupyter-lab --no-browser --ip=0.0.0.0 --port=8888
```

Again, now we can open our browser to `localhost:8888` and we will have access to `jlab`. 


---

### Using `sbatch`

In this case, we will create a script and then launch the job using the `sbatch` command. 

**Pros**:
* It really allows you to customize the nitty gritty details of the compute node environment.
* It launches in the background.

**Cons**: 
* It's not very easy to do. You have to have access to the nitty-gritty details.
* Sometimes you cannot ssh into the compute node.


Below is an example of a `bash` script, titled `jlab_sbatch.sh` which has the appropriate commands.


```bash
#!/bin/bash

#SBATCH --job-name=jlab                     # name of job
#SBATCH --account=python                    # for statistics
#SBATCH --export=ALL                        # export all environment variables
#SBATCH --nodes=1                           # we ALWAYS request one node
#SBATCH --cpus-per-task=8                   # number of cpus per task
#SBATCH --time=8:00:00                      # maximum execution time requested (HH:MM:SS)
#SBATCH --memory=1600                       # the amount of memory requested

# get tunneling info
XDG_RUNTIME_DIR=""
port=8888

# loading of modules
conda activate jlab
# code execution
jupyter-lab --no-browser --port=${port} --ip=0.0.0.0 --notebook-dir=$WORKDIR
```


---
### Advanced Users

For interactive sessions with `srun`, I strongly recommend the use of functions to ensure that your configurations are correct. We can turn the above steps on the server into a script so that we don't have to keep running these same commands. 


#### Demo Setup

Add this function to your `.profile` or `.bash_profile` **on the remote server**.

:::{tabbed} srun
An example using the SLURM `srun` system:
```bash
function jlab_srun(){
    # activate conda environment with jlab
    conda activate jlab
    # run jupyterlab via srun
    srun --nodes=1 --mem=1600 --time=8:00:00 --account=python --job-name=jlab --pty jupyter-lab --no-browser --port=8888 --ip=0.0.0.0
}
```

:::

:::{tabbed} sbatch
An example using the SLURM `sbatch` system:
```bash
function jlab_sbatch(){
    # run jupyterlab via srun
    sbatch jlab_sbatch.sh
}
```

:::



**Demo Usage**:

I am assume that we have already configured our `.ssh/config` file as done above.

:::{tabbed} srun
```bash
# ssh into
ssh meom_cal1
# run function
jlab_srun
```
:::

:::{tabbed} sbatch
```bash
# ssh into
ssh meom_cal1
# run function
jlab_sbatch
```
:::

This is a low easier and less cumbersome than typing out all of the commands from scratch.

#### **TMUXP**

We can make this even easier with `tmuxp`. We need to create a `.yaml` file which will store all the commands to be executed. Here is an example for `jlab_cal1.yaml`.


:::{tabbed} srun
```bash
session_name: jlab_cal1
windows:
  - window_name: jlab
    panes:
    - shell_command: |
      ssh meom_cal1
      conda activate jlab
      srun --nodes=1 --cpus-per-task=8 --mem=1600 --account=python --pty jupyter-lab --no-browser --ip=0.0.0.0 --port=8888

  - window_name: git
    panes:
    - shell_command: ssh meom_cal1
```
:::

:::{tabbed} sbatch
```bash
session_name: jlab_cal1
windows:
  - window_name: jlab
    panes:
    - shell_command: |
      ssh meom_cal1
      sbatch jlab_sbatch.sh

  - window_name: git
    panes:
    - shell_command: ssh meom_cal1
```
:::


**Demo Usage**:


```bash
tmuxp load jlab_cal1.yaml
```

And as we did before, we can create a function in our local `.profile` that will allow us to run these commands with even greater brevity.

```bash
function jlab_cal1(){
  tmuxp load jlab_cal1.yaml
}
```

So now it's even easier to run:

```bash
jlab_cal1
```

**Super easy**! :)


---
### SLURM on JeanZay

We have to do the same as above however there are a few different commands we need to take care of which allow for more customization. Furthermore, we now have access to 

#### SSH


```python
sshuttle -dns -HN -r meom_cal1 130.84.132.0/24
```

**Demo Usage**:

```bash
sshuttle
```

##### Advanced Usage



---
#### Advanced Users


##### Config



The `meom_cal1.conf` file looks like this:

```bash
130.84.132.0/24
--remote
meom_cal1
```

**Demo Usage**:

```bash
sshuttle --dns -HN @meom_cal1.conf
```

And of course, we will create a nice little function for it.

```bash
function sshuttle_cal1(){
  sshuttle --dns -HN @meom_cal1.conf
}
```

**Demo Usage**:

```bash
sshuttle_cal1
```


##### **TMUXP**


```bash
session_name: jlab_jz_gpu
windows:
  - window_name: jlab
    panes:
    - shell_command: |
        ssh jean_zay
        cd $WORKDIR
        sbatch $HOMEDIR/bin/jlab_gpu.sh

  - window_name: vpn
    panes:
    - shell_command: sshuttle --dns -HN @$HOME/.sshuttle/meom_cal1.conf
  
```

**Demo Usage**:

```bash
tmuxp load jlab_jz.yaml
```


And again, a function:

```bash
function jlab_jz(){
  tmuxp load jlab_jz.yaml
}
```

**Demo Usage**:

```bash
jlab_jz
```

**As easy as it gets**!



---

## JLab on OAR


:::{warning}
We can't actually launch jupyter-lab on the head node. Well, technically we can but it will kill the process almost immediately. You must use the compute nodes to do all computation. 
:::

---
### Using `oarsub`


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




