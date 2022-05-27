# Getting Organized


* Automating Everything
* 

---

## File Structure

```
data/project
logs/project
project/
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