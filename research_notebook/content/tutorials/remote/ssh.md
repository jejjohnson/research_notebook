# SSH Configuration



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
  - [Background SSH Tunneling](#background-ssh-tunneling)
  - [Closing Background SSH Tunnels](#closing-background-ssh-tunnels)
  - [Advanced Users - Bash Script](#advanced-users---bash-script)
    - [SSH Process Background](#ssh-process-background)
    - [Closing Ports](#closing-ports)
- [SSH Config File](#ssh-config-file)
- [Remote Server (Cal1)](#remote-server-cal1)
  - [Important Arguments](#important-arguments)
    - [`IdentityFile`](#identityfile-1)
    - [`ForwardX11`](#forwardx11)
    - [`LocalForward`](#localforward)
- [Remote Server + ProxyJump](#remote-server--proxyjump)
  - [`ProxyJump`](#proxyjump)
  - [Example: `Gricad`](#example-gricad)
  - [Example: `JeanZay`](#example-jeanzay)


---
## SSH

The simplest thing to do would be to just ssh into the remote server. You just need a username and server.

```bash
ssh username@server
```

This should work with your password. However, there are some essential things missing that we will need in order to use more features. Some include:
* Permissions - no more passwords + more security
* Tunneling - to be able to see applications that are hosted on the server
* Proxy Jumping - to be able to ssh into a remote server on a remote server 

The remainder of this section will talk about the extra flags we need to in order to accomplish this.

---
### IdentityFile

This is important so we don't have to keep typing import our password all of the time... This can be alleviated with an ssh key. We can generate this with `ssh-keygen` and then copy it over to the remote server using `ssh-copy-id`.

Please see the detailed [**Instructions**](https://www.ssh.com/academy/ssh/keygen) for how to use `ssh-keygen` to generate a new SSH key **and** copy it over to your remote server. Below are some brief instructions for how to do it.

**Step 1**: Generate the Key

```bash
ssh-keygen -t rsa -b 4096 -f /path/to/id_key
```

**Step 2**: Forward Key to remote server. You will have to enter your password once - and then never again :).

```bash
ssh-copy-id -i /path/to/id_key username@server
```


Now we should be able to log in with a specified key and we won't have to enter our password.

```bash
ssh username@server -i /path/to/id_key_file
```

---
### Port Forwarding

We need to create a tunnel especially if we have applications running on the remote server and we want access to the content. This can be done using the `-L` flag along with the appropriate ports.

```bash
ssh username@server -L 8888:localhost:8888
```

**Note**: You may have to change the port number from `8888` to something else.

Check this [blog](https://bytexd.com/what-is-ssh-tunneling-and-how-does-it-work/) for details about how **port forwarding** (ssh-tunneling) works.

---
### Background SSH Tunneling

Often times we don't want to have the connection open occupying our terminal. So we can put it in the background where we don't need to worry about it. This can be done with the `-f` to ensure that it is executed but stays in the background.

```bash
# create a tunnel
ssh -fN username@server -L 8888:localhost:8888 
```

---
### Closing Background SSH Tunnels

When you're done, it's good practice to make sure you close the ssh-tunnel that you opened. Below, I have some steps to do this in a relatively efficient way. It is a bit tedious to type everything out from scratch but later we can automate this so that we don't have to keep typing in all of the commands. First we need to find all of the background processes that have an ssh.

```bash
ps aux | grep ssh | grep "localhost"
```

We can narrow the search to only those that have a local host attached.

```bash
ps aux | grep ssh | grep "8888:localhost:8888"
```

Then we can kill the process assigned here

```bash
# kill that process manually by looking
kill -9 PID
# kill all processes with localhost
ps aux | grep ssh | grep "localhost" | awk '{print $2}' | xargs kill
# kill a specific one
ps aux | grep ssh | grep "8888:localhost:8888" | awk '{print $2}' | xargs kill
```

These commands are a cumbersome but they get the job done. I would strongly suggest you continue to the next section and try to use some of the scripts in your own workflow. It will make for a much more pleasant experience.

---

### Advanced Users - Bash Script


We want to automate a few of the things listed above. It's too much to type all of that every time. In particular, we will automate:

* Opening up the SSH and Tunneling in the background
* Closing specific and all background ports

#### SSH Process Background

We want to automate the ssh tunnel process in the background. 


```bash
# syntatic sugar for launching ssh tunnel given a server and a port
function ssh_tunnel_bg(){
    server="$1"
    port="$2"
    echo "server:" $1
    echo "port:" $2
    ssh -fN $server -L "$port":localhost:"$port"
}
```

**Demo Usage**:

```bash
ssh_tunnel_bg 8888 meom_cal1
```

**Much simpler** and we don't have to remember all of the commands. It is a bit of *syntatic sugar* but it's more understandable.

#### Closing Ports

Now we want to automate the removal of ports. I would suggest adding these functions to your `.profile` or `.bash_profile` so that you can just use these easier to find and kill ssh background processes.

```bash
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

**Demo Usage**: Kill a Specific Port

```bash
# find the background process
show_bg_sh 8888

# kill the background process
kill_bg_sh 8888
```

**Demo Usage**: Kill Everything

```bash

# find the background process
show_all_bg_sh

# kill the background process
kill_all_bg_sh
```



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

**Demo Usage**:


Test the connection to `cal1`.

```bash
ssh cal1
```

Now it is much easier to ssh into the remote server without having to do all of the extra stuff.

---
### Important Arguments

There are many important arguments that make our lives a lot easier if they are defined properly.

#### `IdentityFile`

This will point to the `identity_file` found in your local server. This removes the need to authenticate every time you log in. It's hard to remember passwords all of the time so this removes that necessity. It's also more secure due to the encryption.

**Note**: Never share this `.pub` with anyone! It should be kept on your personal computer only.

#### `ForwardX11`

This allows us to have some visual bits and pieces. This is important for matlab GUIs and even GUIs for `spyder`. 

#### `LocalForward`

This takes care of all of the port forwarding. You can also forward multiple ports if there is a reason, e.g. 1 for tensor board, 1 for JupyterLab, 1 for some visualization software, etc.



---

## Remote Server + ProxyJump

In this case, we will look at the configuration for `gricad` and `jeanzay`. In this case, we have a head node (not really useful except to enter into the server). And then later we *proxy jump* to the actual head node for computing.


```bash
Host server1
  HostName base_server
	User username

Host server2
  HostName server
  User username
  ProxyJump server1
```

---
### `ProxyJump`

This is the only extra command is the `ProxyJump` command. This allows

---
### Example: `Gricad`

In `gricad` they have a weird system where they have a *entry node* which you ssh into. And then you need to ssh into a compute node, e.g. `dahu`, `bigfoot`, etc. So this means we need to *proxyjump* to `dahu` via `entrynode`.

Below is an example of the `.ssh/config` file with these arguments already in place.

```bash
Host gricad
	User username
	HostName access-gricad.univ-grenoble-alpes.fr
	IdentityFile ~/.ssh/id_rsa_gricad

# DAHU 
Host dahu_gricad
	User johnsonj
	HostName dahu
	ProxyJump gricad
	IdentityFile ~/.ssh/id_rsa_gricad
```

**Example Usage**:

```bash
ssh dahu_gricad
```

This is **much simpler** than having to do the proxy jumps. 


On the [`gricad` docs](https://gricad-doc.univ-grenoble-alpes.fr/en/hpc/connexion/#4-transparent-ssh-connexion-configuration) they have a similar command which uses your `<pereseus-login>`. I have included it below for a side-by-side comparison with my version.

:::{tabbed} gricad Docs
```bash
Host *
  ServerAliveInterval 30

Host *.ciment
  ProxyCommand ssh -q username@access-gricad.univ-grenoble-alpes.fr "nc -w 60 `basename %h .ciment` %p"

Host dahu luke froggy bigfoot
  User username
  ProxyJump username@access-gricad.univ-grenoble-alpes.fr:22
```
**Example Usage**:

```bash
ssh dahu
```
:::

:::{tabbed} My Version
```bash
Host gricad
	User username
	HostName access-gricad.univ-grenoble-alpes.fr
	IdentityFile ~/.ssh/id_rsa_gricad

# DAHU 
Host dahu
	User username
	HostName dahu
	ProxyJump gricad
	IdentityFile ~/.ssh/id_rsa_gricad
```

**Example Usage**:

```bash
ssh dahu
```
:::

It's apparently *transparent* but I really don't understand the `ProxyCommand` they put in place. I think it is much harder to understand compared to my version. I have yet to see any differences in performance from their command and my own.

---
### Example: `JeanZay`

In `jeanzay` the permissions are a bit crazy. It doesn't work to just `ssh` directly to `jeanzay` without some sort of hassle and special permissions with custom ips registered in their system. Instead, it's much easier to just `ssh` into the `cal1` server. Then we can `ssh` from there into the `jeanzay` server. So this means we need to *proxyjump* to `jeanzay` via `cal1`. 


Below is an example of the `.ssh/config` file with these arguments already in place.

```bash
Host cal1
  HostName ige-meom-cal1.u-ga.fr
	User username
	IdentityFile ~/.ssh/id_key_file
	LocalForward 8888 localhost:8888

# JEANZAY Head Node
Host jean_zay
    User username
    HostName jean-zay.idris.fr
    ProxyJump cal1
    IdentityFile ~/.ssh/id_key_file
    LocalForward 8880 localhost:8880
```

**Demo Usage**:

```bash
ssh jean_zay
```

As you can see, this allows us to log into `jeanzay` in a much simpler manner. 

