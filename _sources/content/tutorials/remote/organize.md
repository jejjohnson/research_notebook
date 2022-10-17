# Getting Organized


* Automating Everything
* 


---
## WorkFlow

### Coding

* Prototype: `jupyter-lab`
* Packaging: `vscode`


### Big Jobs

1. Setup Codes on My machine
2. Prototype pipeline with `jupyter-lab` + `vscode`
3. Setup organization on `cal1` server.
4. Prototype job scripts on `slurm`
5. Sync files across `cal1` <---> `jean-zay` and/or `gricad`
6. Run mega-jobs on `slurm`
7. Sync files across `cal1` <---> `jean-zay` and/or `gricad`
8. Visualize, Play, Host Results on `cal1` server

---

## File Structure

```
data/project
logs/project
project/
config/
credentials/
```


**`project`**: this is where all of your source code lives. It should be an environment under version control (e.g. git) so that you can track all of the changes.

**`data/project`**: this should contain all of your data. It can be a symbolic link, or a mounted drive. It's important that all of the subsequent files within the data-drive have the same file structure across 


---
## Syncing Files Across Servers

:::{tip}
Be sure to have the `.ssh/config` ready to ease the process. Otherwise the commands will be fairly cumbersome.
:::


### `Projects`

You should be using `git` and `github`. This is the best way to make sure all changes are being captured and you have the entire history.

```bash
# add files to be commited
git add file1 file2

# create a commit message
git commit -m "commit message"

# push to remote server
git push origin master

# pull from remote server
git pull origin master
```



---
### PreProcessing


```bash
salloc --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --account=cli@cpu 
```

### `Data`


#### `cal1` <---> `gricad`

**Big Data Transfer**

```bash
rsync -avxH /path/to/project/data login@scp.univ-grenoble-alpes.fr:/path/to/project/data
```

**Light Data Transfer**

```bash
rsync -avxH /path/to/project/data login@cargo.univ-grenoble-alpes.fr:/path/to/project/data
```

#### `cal1` <---> `jean-zay`

### `Logs`

**Light Data Transfer**

#### RSYNC

```bash
rsync -avxH jean_zay:/gpfswork/rech/cli/uvo53rl/test_logs/wandb/ /mnt/meom/workdir/johnsonj/test_logs/wandb/
```


##### Functions

```bash
function pull_wandb_changes(){
    rsync -avxH jean_zay:/gpfswork/rech/cli/uvo53rl/logs/wandb/ /mnt/meom/workdir/johnsonj/logs/wandb/
}
function push_wandb_changes(){
    rsync -avxH /mnt/meom/workdir/johnsonj/logs/wandb/ jean_zay:/gpfswork/rech/cli/uvo53rl/logs/wandb/
}
function sync_wandb_changes(){
    wandb sync
}
```

```bash
# sync offline runs
wandb sync --include-offline /mnt/meom/workdir/johnsonj/logs/wandb/offline-*
```

```bash
wandb sync --include-offline /mnt/meom/workdir/johnsonj/logs/wandb/offline-run-20220601_065448-2m11j69u
```

```bash
# make directory for wandb logs
if [ ! -d mkdir /gpfsscratch/rech/cli/uvo53rl/logs ]; then
  mkdir -p mkdir /gpfsscratch/rech/cli/uvo53rl/logs;
fi

# make directory for wandb logs
if [ ! -d /gpfsscratch/rech/cli/uvo53rl/wandb ]; then
  mkdir -p /gpfsscratch/rech/cli/uvo53rl/wandb;
fi

# make directory for wandb logs
if [ ! -d /gpfsscratch/rech/cli/uvo53rl/errs ]; then
  mkdir -p /gpfsscratch/rech/cli/uvo53rl/errs;
fi

# make directory for wandb logs
if [ ! -d /gpfsscratch/rech/cli/uvo53rl/jobs ]; then
  mkdir -p /gpfsscratch/rech/cli/uvo53rl/jobs;
fi

# make dot files
if [ ! -d /gpfsscratch/rech/cli/uvo53rl/.conda ]; then
  mkdir -p /gpfsscratch/rech/cli/uvo53rl/.conda &&
  conda create --prefix=/gpfsscratch/rech/cli/uvo53rl/.conda/envs/jaxtf_gpu_py39 --clone jax_gpu_py39 &&
  conda create --prefix=/gpfsscratch/rech/cli/uvo53rl/.conda/envs/jaxtftorch_gpu_py39 --clone jax_gpu_py39;
fi
if [ ! -d /gpfsscratch/rech/cli/uvo53rl/.cache ]; then
  mkdir -p /gpfsscratch/rech/cli/uvo53rl/.cache;
fi
```

#### SCP

A lot of times you'll get coworkers who can't access or they don't use (or don't want to learn) how to use the server effectively. So they might ask you to help them transfer some files. One way to do it is to use the scp package. The command I use is below.


**Forward Transfer**

```bash
scp -r test jean_zay:/gpfswork/rech/cli/uvo53rl/logs/wandb/
```

**Inverse Transfer**

```bash
scp -r jean_zay:/gpfswork/rech/cli/uvo53rl/logs/wandb/test ./ 
```

Other Resources:

* [Jean-Zay Docs]()
* [SCP Guide]()


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


### Space

**Check Project Space**

```bash
# summary
idr_quota_user
# more detail
idr_quota_project 
```

**Check Home Space**

```bash
# summary
idrquota -m -t Gio
# more detail
du -h --max-depth=1 $HOME
```

**Check Work Space**

```bash
# summary
idrquota -w -t Gio
# more detail
du -h --max-depth=1 $WORK
```
### Symbolic Links

We need to move **everything** to the other drive. Otherwise, we run out of disk space really quickly...
* `workdir`
* `.cache`
* `.local`
* `.ipython`
* `.keras`