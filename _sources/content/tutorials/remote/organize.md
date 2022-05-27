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