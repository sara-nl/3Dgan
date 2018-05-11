#!/bin/bash -l
#SBATCH -J cnn_64w_32n_bs32__0_001__RMSprop_19_1_timeline
#SBATCH -o cnn_3DGAN_hvd_64w_32n_bs32__0_001__RMSprop_19_1_timeline_output.txt
#SBATCH -e cnn_3DGAN_hvd_64w_32n_bs32__0_001__RMSprop_19_1_timeline_errors.txt

#SBATCH -t 48:00:00

#SBATCH --partition iq

#SBATCH --nodes 32
#SBATCH --ntasks-per-node=2

ulimit -l unlimited
ulimit -s unlimited

module load hdf5/1.10.1 intel/mpi/2018u2

cat job_3DGAN_hvd.sh

export HOROVOD_TIMELINE=/mnt/lustrefs/damian.surfsara/cnn_3DGAN_hvd_64w_32n_bs32__0_001__RMSprop_19_1_timeline/timeline.json

mpirun -np 64 -ppn 2 -genvlist HOROVOD_TIMELINE python $HOME/3Dgan/keras/EcalEnergyTrain_hvd.py --datapath=/mnt/lustrefs/damian.surfsara/data/zip/eos/project/d/dshep/LCD/V1/*scan/*.h5 --weightsdir=/mnt/lustrefs/damian.surfsara/cnn_3DGAN_hvd_64w_32n_bs32__0_001__RMSprop_19_1_timeline --batchsize 32 -lr 0.001 --intraop 19 --interop 1 --warmup 5 --nbepochs 25 --optimizer=RMSprop --verbose=True