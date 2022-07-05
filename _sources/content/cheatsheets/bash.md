# Bash

## Functions in `.sh` files


```bash
function srun_cpu(){
    cd $WORKDIR &
    # do srun
    srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --time=02:00:00 --pty bash
}
```

### Arguments

```bash
function srun_cpu(){
    cores={$1:-16}
    # go to work directory
    cd $WORKDIR &
    # do srun
    srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=$cores --time=02:00:00 --pty bash
}
```