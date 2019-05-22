#!/bin/bash -l
#SBATCH --job-name SetENV 
#SBATCH --nodes 1 
#SBATCH --partition broadwell 
#SBATCH --time 01:00:00

module load Python/2.7.14-foss-2017b

wget https://surfdrive.surf.nl/files/index.php/s/hm6EAx3j39H5FFh/download --output-document "tensorflow-1.13.1-cp27-cp27mu-linux_x86_64.whl"

virtualenv cp27_gcc63_omp211
source cp27_gcc63_omp211/bin/activate

pip install \
tensorflow-1.13.1-cp27-cp27mu-linux_x86_64.whl \
scikit-learn keras --no-cache

CC=mpicc CXX=mpicxx \
pip install horovod \
--no-cache

if [ $(python -c "import horovod.tensorflow as hvd;hvd.init();print(hvd.size())") == "1" ]; then
  echo "Test passed, Horovod initialised"
else
  echo "Initialisation check failed"
fi
