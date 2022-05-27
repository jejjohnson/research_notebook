# SSH Config



---

We will go over 2 cases:

1. Remote Server
2. Remote Server with a ProxyJump

---
## Table of Conents

- [Table of Conents](#table-of-conents)
- [SSH](#ssh)
  - [IdentityFile](#identityfile)
  - [Port Forwarding](#port-forwarding)
    - [Background](#background)
      - [Closing The Background Process](#closing-the-background-process)
- [SSH Config File](#ssh-config-file)
- [Remote Server (Cal1)](#remote-server-cal1)
  - [Important Arguments](#important-arguments)
    - [`IdentityFile`](#identityfile-1)
    - [`ForwardX11`](#forwardx11)
    - [`LocalForward`](#localforward)
    - [Testing the Setup](#testing-the-setup)
- [Remote Server + ProxyJump](#remote-server--proxyjump)
  - [`JeanZay`](#jeanzay)
    - [Testing the Setup](#testing-the-setup-1)
  - [`Gricad`](#gricad)
    - [Testing the Setup](#testing-the-setup-2)


---
## SSH

```bash
ssh username@server
```
---
### IdentityFile

```bash
ssh username@server -i /path/to/id_key_file
```

**Forward ID**

```bash
ssh-copy-id -i /path/to/id_key username@server
```

---
### Port Forwarding

```bash
ssh username@server 8888:localhost:8888
```

---
#### Background

Often times we don't want to have the connection open occupying our terminal. So we can put it in the background where we don't need to worry about it.

```bash
# create a tunnel
ssh -fN username@server -L 8888:localhost:8888 
```

The `-f` flag ensures that it is executed to the background.

---
##### Closing The Background Process

When you're done, make sure you close the tunnel you opened. First we need to find all of the background processes that have an ssh.

```bash
ps aux | grep ssh | grep "localhost"
```

We can narrow the search to only those that have a local host attached.

```bash
ps aux | grep ssh | localhost assigned
```

Then we can kill the process assigned here

```bash
# kill that process manually by looking
kill -9 PID
# kill it automatically
ps aux | grep ssh | grep "localhost" | awk '{print $2}' | xargs kill
```

:::{admonition} Automation
:class: tip

Now we want to automate this. I would suggest adding these functions to your `.profile` or `.bash_profile` so that you can just use these easier to find and kill ssh background processes.

```bash
# syntatic sugar for launching ssh tunnel given a server and a port
function ssh_tunnel(){
    server="$1"
    port="$2"
    echo "server:" $1
    echo "port:" $2
    ssh -fN $server -L "$port":localhost:"$port"
}

# function to show all background ssh processes
# demo:
# > show_all_bg_ssh
#
function show_all_bg_ssh(){
    ps aux | grep ssh | grep "localhost"
}
# function to show background ssh processes given a localport
# demo:
# > show_bg_ssh 8888
#
function show_bg_ssh(){
    ps aux | grep ssh | grep $1
}

# function to kill background ssh process given a localport number
# demo:
# > kill_bg_ssh 8888
#
function kill_bg_ssh(){
    ps aux | grep ssh | grep $1 | awk '{print $2}' | xargs kill
}

# function to kill all background ssh processes
# demo:
# > kill_all_bg_ssh
#
function kill_all_bg_ssh(){
    ps aux | grep ssh | grep "localhost" | awk '{print $2}' | xargs kill
}
```

**Demo Usage** (Specific Port):

```bash
# launch ssh
ssh_tunnel cal1 8888

# find the background process
show_bg_sh 8888

# kill the background process
kill_bg_sh 8888
```

**Demo Usage** (Everything):

```bash
# launch ssh
ssh_tunnel cal1 8888

# find the background process
show_all_bg_sh

# kill the background process
kill_all_bg_sh
```

:::


---
## SSH Config File

It ups your game to the next level if you use an ssh config file. 

More resources:

* [Gricad Docs](https://gricad-doc.univ-grenoble-alpes.fr/en/hpc/connexion/)
* [Blog](https://linuxize.com/post/using-the-ssh-config-file/) - Using the SSH Config File

We typically store this here:

```bash
/home/user/.ssh/config
```
where it's visible from anywhere in on your computer.


---
## Remote Server (Cal1)

In this example, we will showcase how to automate the connection to

We can automate this using using

```bash
Host cal1
    HostName ige-meom-cal1.u-ga.fr
	User username
	# Allow for Ids
	ForwardX11 yes
	# SSH Identity File
	IdentityFile ~/.ssh/id_key_file
	# Jupyter Notebooks
	LocalForward 8888 localhost:8888

Host *
  AddKeysToAgent yes
  UseKeychain yes
  IdentityFile ~/.ssh/id_key_file
```

---
### Important Arguments

#### `IdentityFile`

This will point to the `identity_file` found in your local server. This removes the need to authenticate every time you log in. It's hard to remember passwords all of the time so this removes that necessity. It's also more secure due to the encryption.

**Note**: Never share this `.pub` with anyone! It should be kept on your personal computer only.

#### `ForwardX11`

This allows us to have some visual bits and pieces. This is important for matlab GUIs and even GUIs for `spyder`. 

#### `LocalForward`

This takes care of all of the port forwarding. You can also forward multiple ports if there is a reason, e.g. 1 for tensor board, 1 for JupyterLab, 1 for some visualization software, etc.

---
#### Testing the Setup

**Step 1**: Test the connection to `cal1`.

```bash
ssh cal1
```

Now it is much easier to ssh into the remote server without having to do all of the extra stuff.


---

## Remote Server + ProxyJump

In this case, we will look at the configuration for `gricad` and `jeanzay`. In this case, we have a head node (not really useful except to enter into the server). And then later we *proxy jump* to the actual head node for computing.

---
### `JeanZay`

```bash
Host cal1
    HostName ige-meom-cal1.u-ga.fr
	User username
	# Allow for Ids
	ForwardX11 yes
	# SSH Identity File
	IdentityFile ~/.ssh/id_key_file
	# Jupyter Notebooks
	LocalForward 8888 localhost:8888

# JEANZAY Head Node
Host jean_zay
    User username
    HostName jean-zay.idris.fr
    ProxyJump cal1
    IdentityFile ~/.ssh/id_key_file
    # Jupyter Notebooks
    LocalForward 8880 localhost:8880

Host *
  AddKeysToAgent yes
  UseKeychain yes
  IdentityFile ~/.ssh/id_key_file
```

---
#### Testing the Setup


**Step 1**: Test access to the `cal1` node.

```bash
ssh cal1
```

**Step 2**: Test access the mega head node (`jeanzay`)

```bash
ssh jean_zay
```

---
### `Gricad`

```bash
# GRIDCAD Entry Node Access
Host gricad
    User username
    HostName access-gricad.univ-grenoble-alpes.fr
    IdentityFile ~/.ssh/id_key_file

# DAHU Head Node (CPUs)
Host dahu_gricad
    User username
    HostName dahu
    ForwardX11 yes
    ProxyJump gricad
    IdentityFile ~/.ssh/id_key_file
    # Jupyter Notebooks
    LocalForward 8888 localhost:8888

# BIGFOOT Head Node (GPUs) 
Host bigfoot_gricad
    User username
    HostName bigfoot
    ForwardX11 yes
    ProxyJump gricad
    IdentityFile ~/.ssh/id_key_file
    # Jupyter Notebooks (CPU)
    LocalForward 8889 localhost:8889

Host *
  AddKeysToAgent yes
  UseKeychain yes
  IdentityFile ~/.ssh/id_key_file
```

---
#### Testing the Setup

**Step 1**: Test access the compute node (`gricad`)

```bash
ssh gricad
```

**Note**: There is almost never any reason to do this...

**Step 2**: Test access the head node (`dahu`)

```bash
ssh dahu_gricad
```

**Step 3**: Test access the GPU compute node (`Bigfoot`)

```bash
ssh bigfoot_gricad
```



