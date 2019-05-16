#!/bin/bash -l
#SBATCH -J 3DGAN_4w_1n_bs16_sun
#SBATCH -o 3DGAN_4w_1n_bs16_sun_output.txt
#SBATCH -e 3DGAN_4w_1n_bs16_sun_errors.txt

#SBATCH -t 1:00:00

#SBATCH --partition broadwell_short

#SBATCH --nodes 1
#SBATCH --ntasks-per-node=4

module load Python/2.7.14-foss-2017b

ulimit -l unlimited
ulimit -s unlimited

source "/home/damian/cern_1.3/cp27_gcc63_omp211/bin/activate"

mkdir -p "/scratch/shared/damian/CERN/3DGAN_4w_1n_bs16_sun"

export KERAS_BACKEND="tensorflow"

NUM_NODES=1
WORKERS_PER_SOCKET=2

INTER_T=2

NUM_SOCKETS=`lscpu | grep "Socket(s)" | cut -d':' -f2 | xargs`
CORES_PER_SOCKET=`lscpu | grep "Core(s) per socket" | cut -d':' -f2 | xargs`

CORES_PER_WORKER=$((CORES_PER_SOCKET / WORKERS_PER_SOCKET))
INTRA_T=$CORES_PER_WORKER
OMP_NUM_THREADS=4 #32 physical cores

WORKERS_PER_NODE=$((WORKERS_PER_SOCKET * NUM_SOCKETS))
TOTAL_WORKERS=$((NUM_NODES * WORKERS_PER_NODE))


echo "CORES_PER_WORKER: $CORES_PER_WORKER"
echo "NUM_INTRA_THREADS: $INTRA_T"
echo "NUM_INTER_THREADS: $INTER_T"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "WORKERS_PER_NODE: $WORKERS_PER_NODE"
echo "TOTAL_WORKERS: $TOTAL_WORKERS"

export OMP_NUM_THREADS=$OMP_NUM_THREADS
export KMP_AFFINITY="granularity=fine,compact,1,0"

which mpirun

mpirun -np ${TOTAL_WORKERS} \
 --map-by socket: \
 --map-by socket:pe=${INTRA_T} \
 --bind-to core \
 --report-bindings \
 --oversubscribe \
 -x LD_LIBRARY_PATH \
 -x HOROVOD_FUSION_THRESHOLD \
 -x HOROVOD_CYCLE_TIME \
 -x OMP_NUM_THREADS \
 -x KMP_AFFINITY \
 numactl -l python tf_gan.py \
 --model EcalEnergyGan \
 --nbepochs 25 \
 --batchsize 16 \
 --datapath='/scratch/shared/damian/CERN/EleScan/*.h5' \
 --weightsdir='/scratch/shared/damian/CERN/3DGAN_4w_1n_bs16_sun' \
 --intraop ${INTRA_T} \
 --interop ${INTER_T}
