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


### `Project`

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

```bash
rsync -avxH /path/to/project/data login@cargo.univ-grenoble-alpes.fr:/path/to/project/data
```

**SCP**

A lot of times you'll get coworkers who can't access or they don't use (or don't want to learn) how to use the server effectively. So they might ask you to help them transfer some files. One way to do it is to use the scp package. The command I use is below.


```bash
scp -r user@your.server.example.com:/path/to/foo /home/user/Desktop/
```


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