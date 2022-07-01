# Server Access



---
## Profile

```bash

# CHANGE PROFILE


# LOAD DEFAULT MODULES
module load cuda/11.2
module load cudnn/10.1-v7.5.1.10
```

---
## Github Profiles




---
## GriCad

### Configs

#### Profile

```bash
# JOB KEY FILE
export OAR_JOB_KEY_FILE=/home/johnsonj/.ssh/id_my_job_key

# SHORTCUTS
jobs_check(){
    oarstat -u johnsonj
}
check_nodes(){
    recap.py
}
jobs_submit(){
    oarsub -S $1
}
jobs_check(){
    oarstat -u johnsonj
}


# LANDING DIRECTORY
export $WORK /bettik/johnsonj
cd $WORK

# INTERACTIVE JOBS (CPU)
jobs_cpu_interactive() {
    time=${1:-2:00:00}
    oarsub -l /nodes=1,walltime=time, --project pr-data-ocean -I
}

# INTERACTIVE JOBS (GPU)
jobs_gpu_interactive() {
    time=${1:-2:00:00}
    oarsub -l /nodes=1/gpu=1,walltime=time -p "gpumodel='V100'" --project=pr-data-ocean -I
    source /applis/environments/cuda_env.sh bigfoot 10.2
}

```

#### Conda

```bash

```


---
### Add SSH Keys


### Interactive Nodes


#### CPU

```bash
alias jobs_interactive_cpu="oarsub -I -l /nodes=1,walltime=1:00:00 --project pr-data-ocean"
```


#### GPU


##### DAHU

```bash
alias jobs_interactive_cpu="oarsub -I -l /nodes=1,walltime=1:00:00, --project pr-data-ocean"
```

##### BigFoot

```bash
alias jobs_interactive_cpu="oarsub -l /nodes=1/gpu=1,walltime=2:00:00 --project pr-data-ocean -I"
jobs_interactive_gpu(){
    oarsub -l /nodes=1/gpu=1,walltime=2:00:00 -p "gpumodel='V100'" --project pr-data-ocean -I
}


```




---
### JupyterLab

#### Common Commands

```bash
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
ssh -N -f -L ${port}:${node}:${port} ${user}@${cluster}

Command to create ssh tunnel through server
ssh -N -f -L ${port}:localhost:${port} $

Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"


# load modules or conda environments here
source activate jlab

jupyter lab --no-browser --ip=0.0.0.0 --port=${port}
```


#### CPU

```bash
#!/bin/bash

#OAR --name jlab_cpu
#OAR --project pr-data-ocean
#OAR -l /nodes=1,walltime=2:00:00
#OAR --stdout /bettik/johnsonj/logs/jupyterlab_cpu.log
#OAR --stderr /bettik/johnsonj/errs/jupyterlab_cpu.log
#OAR -e /home/johnsonj/.ssh/id_my_job_key
```



#### CPU (GPU)

```bash
#!/bin/bash

#OAR --name jlab_gpu
#OAR --project pr-data-ocean
#OAR -l /nodes=1,walltime=12:00:00
#OAR --stdout /bettik/johnsonj/logs/jupyterlab_gpu.log
#OAR --stderr /bettik/johnsonj/errs/jupyterlab_gpu.log
#OAR -e /home/johnsonj/.ssh/id_my_job_key
```



---
## JeanZay


---
### IP Addresses


**Start sshuttle** 

We want to start a tunnel to our proxy

```bash
sshuttle --dns -vHN -r meom_cal1 130.84.132.0/24
# sshuttle --dns -vHN @vpn_jz.conf
```

**File**: `vpn_jz.conf`

```bash
130.84.132.16/32
130.84.132.17/32
--remote
meom_cal1
```

**Start a Node**

```bash
salloc --ntasks=1 --cpus-per-task=10 --gres=gpu:1 --hint=nomultithread -C v100-16g --qos=qos_gpu-dev -A cli@gpu --time=02:00:00 --job-name=repl
```

**Start SSHuttle**

We want to start a tunnel from our local computer to the node

```
sshuttle --dns -vHN -r jean_zay 10.159.36.43/32
```


**Start JLab**

```bash
jupyter lab --port 8881
```


**SSH Into Node**

```bash
ssh uvo53rl@10.159.36.43 -L 8881:localhost:8881
# ssh username@node_ip -L 8881:localhost:8881
```

---
### Conda

```bash
envs_dirs:
    - /gpfswork/rech/cli/username/.conda/envs
```


### Interactivate Nodes


#### CPU


```bash
srun --pty --account cli@cpu --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --hint=nomultithread bash
```

#### GPU

**Single GPU**

```bash
srun --pty --account cli@gpu --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --gres=gpu:1 --hint=nomultithread bash
```

**Multiple GPUs**

```bash
srun --pty --account cli@gpu --nodes=1 --ntasks-per-node=4 --cpus-per-task=16 --gres=gpu:4 --hint=nomultithread bash
```


```python

import torch
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch: {torch.cuda.is_available()}")
print(f"PyTorch: {torch.cuda.device_count()}")
```

```python
import jax
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
print(jax.devices())
```


```bash
# save hostname
hostname --ip > $HOME/jlab_configs/jlab_gpu.ip
# save node
hostname > $HOME/jlab_configs/jlab_gpu.node
# save user
whoami > $HOME/jlab_configs/jlab_gpu.user
```

---
### JLAB (CPU)

```yaml
session_name: repl_jz_test
windows:
  - window_name: repl
    panes:
    - shell_command: |
        salloc --ntasks=1 --cpus-per-task=10 --gres=gpu:1 --hint=nomultithread -C v100-16g
        squeue -u $USER -h | grep repl_test | awk '{print $NF}' > $HOME/jlab_configs/alloc_gpu.node
        ssh $(cat $HOME/jlab_configs/alloc_gpu.node) -o StrictHostKeyChecking=no
        hostname -I | awk '{print $1}' > $HOME/jlab_configs/alloc_gpu.ip
        whoami > $HOME/jlab_configs/alloc_gpu.user
```

```bash
allocate_node_test(){
    salloc --ntasks=1 --cpus-per-task=10 --gres=gpu:1 --hint=nomultithread -C v100-16g --qos=qos_gpu-dev -A cli@gpu --time=2:00:00 --job-name=repl_test
    squeue -u $USER -h | grep repl_test | awk '{print $NF}' > $HOME/jlab_configs/alloc_gpu.node
    ssh $(cat $HOME/jlab_configs/alloc_gpu.node) -o StrictHostKeyChecking=no
    hostname -I | awk '{print $1}' > $HOME/jlab_configs/alloc_gpu.ip
}
```




```bash
allocate_node_gpu(){
    salloc --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --hint=nomultithread -C v100-16g -A cli@gpu --time=10:00:00 --job-name=repl
    squeue -u $USER -h | grep repl | awk '{print $NF}' > $HOME/jlab_configs/alloc_gpu.node
    ssh $(cat $HOME/jlab_configs/alloc_gpu.node) -o StrictHostKeyChecking=no
    hostname -I | awk '{print $1}' > $HOME/jlab_configs/alloc_gpu.ip
    whoami > $HOME/jlab_configs/alloc_gpu.user
}
```

```bash
allocate_node_cpu(){
    salloc --nodes=1 --ntasks-per-node=1 -C v100-16g -A cli@cpu --time=10:00:00 --job-name=repl_cpu
    squeue -u $USER -h | grep repl_cpu | awk '{print $NF}' > $HOME/jlab_configs/alloc_cpu.node
    ssh $(cat $HOME/jlab_configs/alloc_gpu.node) -o StrictHostKeyChecking=no
    hostname -I | awk '{print $1}' > $HOME/jlab_configs/alloc_cpu.ip
    whoami > $HOME/jlab_configs/alloc_cpu.user
}
```


```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=cli@gpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-dev
#SBATCH --constraint=v100-16g
#SBATCH --exclude=nodo17
#SBATCH --job-name jlab_gpu
#SBATCH --output /gpfswork/rech/cli/uvo53rl/logs/jupyterlab-%J.log
#SBATCH --error /gpfswork/rech/cli/uvo53rl/logs/jupyterlab-%J.log

# get tunneling info
XDG_RUNTIME_DIR=""
node=$(hostname -s)
ip=$(hostname -i)
user=$(whoami)
port=8881

# SAVE Tunneling Info
user > $WORK/jlab_configs/jlab_gpu.user
node > $WORK/jlab_configs/jlab_gpu.node
ip > $WORK/jlab_configs/jlab_gpu.ip

# print tunneling instructions jupyter-log
echo -e "
# Tunneling Info
node=${node}
user=${user}
cluster=${cluster}
port=${port}

Command to create ssh tunnel:
ssh -N -f -L ${port}:${node}:${port} ${user}@${cluster}

Command to create ssh tunnel through server
ssh -N -f -L ${port}:localhost:${port} $

Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"


# load modules or conda environments here
conda activate jlab

jupyter lab --no-browser --ip=0.0.0.0 --port=${port}
```

```python
tmuxp load .tmuxp/jz_alloc_cpu.yaml
sshuttle --dns -vNr jean_zay $(ssh jean_zay 'cat $HOME/jlab_configs/alloc_cpu.ip')
```