#!/bin/bash -l
#SBATCH -J 3DGAN_1w_1n_bs128__RMSprop
#SBATCH -o 3DGAN_1w_1n_bs128__RMSprop_output.txt
#SBATCH -e 3DGAN_1w_1n_bs128__RMSprop_errors.txt

#SBATCH -t 2:00:00

#SBATCH --partition skx-dev

#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1

ulimit -l unlimited
ulimit -s unlimited

module load gcc

module list

COMMAND="mpirun -np 1 -ppn 1 python /work/04653/damianp/stampede2/3Dgan/keras/EcalEnergyTrain.py"

echo $COMMAND

$COMMAND