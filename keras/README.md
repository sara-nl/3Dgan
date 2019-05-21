# Installation
## Stampede2
In order to replicate the run environment on Stampede2 you should run the ```init_env_stampede2.sh``` script:

```
#!/bin/bash
module load gcc

wget https://surfdrive.surf.nl/files/index.php/s/hm6EAx3j39H5FFh

virtualenv cp27_gcc71_impi17
source cp27_gcc71_impi17/bin/activate

pip install tensorflow-1.13.1-cp27-cp27mu-linux_x86_64.whl --ignore-installed --no-cache
pip install horovod --no-cache

if [ $(python -c "import horovod.tensorflow as hvd;hvd.init();print(hvd.size())") == "1" ]; then
  echo "Test passed, Horovod initialised"
else
  echo "Integrity check failed"
fi
```

This will load the environment, install the Python packages needed and activate the virtual environment.

## Cartesius
There is a job script for Cartesius [```init_env_cartesius```](init_env_cartesius.sh). Estimated ```RunTime=00:05:19```.

## Usage
The order should be:
1. ```module load``` ...
2. ```source virtualenv_folder/bin/activate```

like in the ```job_*.sh``` job submission files. This sets the correct paths, otherwise Python packages outside the virtual environment are used.

# Run the Keras code
The script [```job_stampede_4w_1n_16bs.sh```](job_stampede_4w_1n_16bs.sh) will launch a one node run with four workers.
The shell script is configured for a system with 24C/socket or 48 Cores/Node and 4 workers/node. Our recommendation is that the following equation hold true:

```Inter_op * OMP_NUM_THREADS <= Physical Cores/Worker```

So, for a 48 cores node and 4 ```Workers/Node``` we have 12 ```Phyical_Cores/Worker```.

If ```inter_op``` is 2 then ```OMP_NUM_THREADS``` is ```Phyical_Cores/Worker``` divided by ```inter_op```, therefore ```12/2 = 6```. In this case ```intra_op``` is ```48/4 = 12```.
