#!/bin/bash -l
#SBATCH -J 3DGAN_512w_128n_bs4_10_1_x16f_Adam_200_TF19
#SBATCH -o 3DGAN_512w_128n_bs4_10_1_x16f_Adam_200_TF19_output.txt
#SBATCH -e 3DGAN_512w_128n_bs4_10_1_x16f_Adam_200_TF19_errors.txt

#SBATCH -t 48:00:00
#SBATCH --partition iq

#SBATCH --nodes=128
#SBATCH --ntasks=512
#SBATCH --ntasks-per-node=4
#SBATCH --exclude=skl182, skl130
set -x

ulimit -l unlimited
ulimit -s unlimited

module load hdf5/1.10.1 intel/mpi/2018u2

# export MKL_VERBOSE=1

export RUNDIR=/mnt/lustrefs/damian.surfsara/3DGAN_512w_128n_bs4_10_1_x16f_Adam_200_TF19
export HOROVOD_TIMELINE=$RUNDIR/timeline.json

mpirun -np 512 -ppn 4 -genvlist HOROVOD_TIMELINE \
  python EcalEnergyTrain_hvd.py \
  --model=EcalEnergyGan \
  --datapath=/mnt/lustrefs/damian.surfsara/data/zip/eos/project/d/dshep/LCD/V1/*scan/*.h5 \
  --weightsdir=$RUNDIR \
  --batchsize 4 \
  --learningRate 0.001 --optimizer=Adam \
  --latentsize 200 \
  --intraop 10 --interop 1 \
  --warmupepochs 0 --nbepochs 25

# mpirun -np 1 -ppn 1 -genvlist HOROVOD_TIMELINE python $HOME/3Dgan/keras/EcalEnergyTrain_hvd.py --datapath=/mnt/lustrefs/damian.surfsara/data/zip/eos/project/d/dshep/LCD/V1/*scan/*.h5 --weightsdir=/mnt/lustrefs/damian.surfsara/cnn_1w_1n_bs2048__0_001__RMSprop_38_2_timeline1 --batchsize 2048 -lr 0.001 --intraop 38 --interop 2 --warmup 5 --nbepochs 25 --optimizer=RMSprop