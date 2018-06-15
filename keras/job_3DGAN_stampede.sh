#!/bin/bash -l
#SBATCH -J 3DGAN_1w_1n_bs128_47_1_n16all_RMSprop_200_newwheel
#SBATCH -o 3DGAN_1w_1n_bs128_47_1_n16all_RMSprop_200_newwheel_output.txt
#SBATCH -e 3DGAN_1w_1n_bs128_47_1_n16all_RMSprop_200_newwheel_errors.txt

#SBATCH -t 2:00:00

#SBATCH --partition skx-dev

#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1

module load gcc python

export OMP_NUM_THREADS=47

# export MKL_VERBOSE=1

ulimit -l unlimited
ulimit -s unlimited

export PATH=/work/04653/damianp/stampede2/root/install/bin:$PATH
export LD_LIBRARY_PATH=/work/04653/damianp/stampede2/root/install/lib:$LD_LIBRARY_PATH
export CPATH=/work/04653/damianp/stampede2/root/install/include:$CPATH
export PYTHONPATH=/work/04653/damianp/stampede2/root/install/lib:$PYTHONPATH

mkdir /scratch/04653/damianp/3DGAN_1w_1n_bs128_47_1_n16all_RMSprop_200_newwheel

# export HOROVOD_TIMELINE=/scratch/04653/damianp/3DGAN_16w_4n_bs128_11_1_n16all_RMSprop/timeline.json

mpirun -np 1 -ppn 1 -genvlist HOROVOD_TIMELINE python /work/04653/damianp/stampede2/3Dgan/keras/EcalEnergyTrain_hvd_o.py --datapath=/scratch/04653/damianp/eos/project/d/dshep/LCD/V1/*scan/*.h5 --weightsdir=/scratch/04653/damianp/3DGAN_1w_1n_bs128_47_1_n16all_RMSprop_200_newwheel --batchsize 128 -lr 0.001 --latentsize 200 --intraop 47 --interop 1 --warmup 5 --nbepochs 25 --optimizer=RMSprop

module list

# COMMAND="mpirun -np 1 -ppn 1 python /work/04653/damianp/stampede2/3Dgan/keras/EcalEnergyTrain.py"

# echo $COMMAND

# $COMMAND