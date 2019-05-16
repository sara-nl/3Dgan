#Installation
In order to replicate the run environment on Stampede2 you should run the ```init_env_stampede2.sh``` script:

```
#!/bin/bash
module load gcc

wget https://surfdrive.surf.nl/files/index.php/s/hm6EAx3j39H5FFh

virtualenv cp27_gcc71_impi17
source /work/04653/damianp/stampede2/cern_1.3/cp27_gcc71_impi17/bin/activate

pip install tensorflow-1.13.1-cp27-cp27mu-linux_x86_64.whl --ignore-installed --no-cache
pip install horovod --no-cache

if [ $(python -c "import horovod.tensorflow as hvd;hvd.init();print(hvd.size())") == "1" ]; then
  echo "Test passed, Horovod initialised"
else
  echo "Integrity check failed"
fi
```

This will load the environment, install the Python packages needed and activate the virtual environment

#Run
The script [```job_stampede_4w_1n_16bs.sh```](job_stampede_4w_1n_16bs.sh) will launch a one node run with four workers.
The shell script is configured for a system with 24C/socket or 48 Cores/Node and 4 workers/node. Our recommendation is that the following equation hold true:

Inter_op * OMP_NUM_THREADS <= Physical Cores/Worker

So, for a 48 Cores/Node and 4 Workers/Node => 12 Phy Cores/Workers.

If inter_op = 2 then

                OMP_NUM_THREADS = Physical Cores/worker / inter_op = 12/2 = 6

                Intra_op = 48/4 = 12