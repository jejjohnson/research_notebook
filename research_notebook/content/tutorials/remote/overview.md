# Remote Computing

Remote computing is the future. It is

All of the following tutorials are for remote servers:

* [SSH Config](./ssh.md)
* [Conda ](./conda.md)
* [Jupyter Lab](./jlab.md)
* [Organization](./organize.md) (TODO)
* [Visual Studio Code]() (TODO)
* [Tips n Tricks for ML]() (TODO)

These tutorials should be followed in sequential order because:

* that's the path of increasing difficulty
* that's the path of increasing usefulness.

---
## [**SSH Configuration**](./ssh.md)

This tutorial will demonstrate how to log in using `ssh` and then how to set up your `.ssh/config` file. `ssh` is probably the most important tool you need to be relatively familiar with in order to have a good workflow. The `.ssh/config` file will automate some of the commands and permissions and it will make things **a lot** easier. For example:

* Permissions
* Tunneling
* Proxy Jumping

---
## [**Conda**](./conda.md)

This tutorial will demonstrate how we can set up personal `conda` as our package management on remote servers. We will show how we can have full control of our package environments but still be able to see some of the pre-configured packages that the server admins may have installed.

---
## [**Jupyter Lab**](./jlab.md)

This tutorial will demonstrate how to install, start and use `jupyter-lab` / `jupyter-notebook` effectively on the servers while still being in the comforts of our own home. It will also showcase the different scenarios of how to get it to work using:

* `cal1` - a small server that uses the `slurm` management system.
* `jeanzay` - a security-heavy server that uses the `slurm` management system.
* `gricad` - a server that uses the `oar` management system.

---
## [**Jupyter Lab Xtras**](./jlab_xtras.md) (TODO)


---
## [**Organization**](./organize.md) (TODO)

This tutorial will give some tips and tricks for my workflow. It might be useful to showcase how I code and how I use all of the servers and the tools at my disposal to be efficient and effective. There will also be some extra tips and tricks:

* `git` + `github` - local and remote version control
* `zsh` - a more advanced cli prompt 
* `tmux` - for running scripts in the background
* `tmuxp` - for automating the navigation pane depending upon the project

---
## [**Visual Studio Code**]() (TODO)

This tutorial will demonstrate how we can use a fully-fledged editor for programming via `ssh`-remote-computing. It's possible but it does require a bit of manipulation.

* Useful extensions
* JupyterLab embedded
* Debugging on a remote node


**Note**: I have not been able to use `vscode` on servers that don't allow dynamic IPs, e.g. `jean-zay`.

---
## [**Tips n Tricks 4 ML**]() (TODO)



* logging with *weights and biases*
* syncing files across servers

<!-- 
**Language**: `Python`

**Package Manager**: `Conda`

**Research**: `JupyterLab`

**IDE**: `Visual Studio Code`

**Dissemination**: `Weights & Biases`, `JupyterBook`, `Docs` -->