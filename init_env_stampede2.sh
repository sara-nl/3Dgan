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