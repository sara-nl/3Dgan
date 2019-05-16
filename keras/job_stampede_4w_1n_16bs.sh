#!/bin/bash -l
#SBATCH -J 3DGAN_4w_1n_bs16_sun
#SBATCH -o 3DGAN_4w_1n_bs16_sun_output.txt
#SBATCH -e 3DGAN_4w_1n_bs16_sun_errors.txt

#SBATCH -t 2:00:00

#SBATCH --partition skx-dev

#SBATCH --nodes 1
#SBATCH --ntasks-per-node=4

module load gcc 

ulimit -l unlimited
ulimit -s unlimited

source /work/04653/damianp/stampede2/cern_1.3/cp27_gcc71_impi17/bin/activate

export PATH=/work/04653/damianp/stampede2/root/install/bin:$PATH
export LD_LIBRARY_PATH=/work/04653/damianp/stampede2/root/install/lib:$LD_LIBRARY_PATH
export CPATH=/work/04653/damianp/stampede2/root/install/include:$CPATH
export PYTHONPATH=/work/04653/damianp/stampede2/root/install/lib:$PYTHONPATH

mkdir /scratch/04653/damianp/3DGAN_4w_1n_bs16_sun

NUM_NODES=1
WORKERS_PER_SOCKET=2

INTER_T=2

NUM_SOCKETS=`lscpu | grep "Socket(s)" | cut -d':' -f2 | xargs`
CORES_PER_SOCKET=`lscpu | grep "Core(s) per socket" | cut -d':' -f2 | xargs`

CORES_PER_WORKER=$((CORES_PER_SOCKET / WORKERS_PER_SOCKET))
INTRA_T=$CORES_PER_WORKER
OMP_NUM_THREADS=6

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

mpirun -np ${TOTAL_WORKERS} \
 --map-by socket \
 -cpus-per-proc ${INTRA_T} \
 --report-bindings \
 --oversubscribe \
 -x LD_LIBRARY_PATH \
 -x HOROVOD_FUSION_THRESHOLD \
 -x HOROVOD_CYCLE_TIME \
 -x OMP_NUM_THREADS \
 -x KMP_AFFINITY \
 numactl -l python EcalEnergyTrain_hvd.py \
 --model EcalEnergyGan \
 --nbepochs 25 \
 --batchsize 16 \
 --datapath='/scratch/04653/damianp/eos/project/d/dshep/LCD/V1/*scan/*.h5' \
 --weightsdir='/scratch/04653/damianp/3DGAN_4w_1n_bs16_sun' \
 --intraop ${INTRA_T} \
 --interop ${INTER_T}
