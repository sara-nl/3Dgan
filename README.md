# 3Dgan

## How to run

First set up the rundir (generator-discriminator models and horovod timeline are saved here)
```bash
export RUNDIR=/scratch/3DGAN_1w_1n_bs32_11_1_x16f_Adam_200_TF19
export HOROVOD_TIMELINE=$RUNDIR/timeline.json
mkdir -p $RUNDIR
echo $(hostname) > $RUNDIR/hostname
```

Then run (should change datapath to dataset location)
### With 1 worker 1 node
```bash
python EcalEnergyTrain_hvd.py \
		--datapath=/scratch/eos/project/d/dshep/LCD/V1/*scan/*.h5 \
		--weightsdir=$RUNDIR \
		--batchsize 16 \
		--lr 0.001 --optimizer=Adam \
		--latentsize 200 \
		--intraop 11 --interop 1 \
		--warmup 0 --nbepochs 25 
```

### With 4 workers 1 node
```bash
mpirun -np 4 -ppn 4 -f $RUNDIR/hostname \
  -genvlist HOROVOD_TIMELINE \
  python EcalEnergyTrain_hvd.py \
  --model=EcalEnergyGan \
  --datapath=/scratch/eos/project/d/dshep/LCD/V1/EleEscan/EleEscan_2_8.h5 \
  --weightsdir=$RUNDIR \
  --batchsize 16 \
  --learningRate 0.001 --optimizer=Adam \
  --latentsize 200 \
  --intraop 11 --interop 1 \
  --warmupepochs 0 --nbepochs 25
```

### Stampede submission script
```bash
#!/bin/bash -l
#SBATCH -J 3DGAN_wheel
#SBATCH -o 3DGAN_wheel_output.txt
#SBATCH -e 3DGAN_wheel_errors.txt

#SBATCH -t 2:00:00

#SBATCH --partition skx-dev

#SBATCH --nodes 2
#SBATCH --ntasks-per-node=4

set -x

module load gcc python

export OMP_NUM_THREADS=11

# export MKL_VERBOSE=1

ulimit -l unlimited
ulimit -s unlimited

export PATH=/work/04653/damianp/stampede2/root/install/bin:$PATH
export LD_LIBRARY_PATH=/work/04653/damianp/stampede2/root/install/lib:$LD_LIBRARY_PATH
export CPATH=/work/04653/damianp/stampede2/root/install/include:$CPATH
export PYTHONPATH=/work/04653/damianp/stampede2/root/install/lib:$PYTHONPATH

mkdir /scratch/04653/damianp/3DGAN_wheel

export HOROVOD_TIMELINE=/scratch/04653/damianp/3DGAN_wheel/timeline.json

mpirun -np 8 -ppn 4 -genvlist HOROVOD_TIMELINE python /work/04653/damianp/stampede2/3Dgan/keras/EcalEnergyTrain_hvd.py --datapath=/scratch/04653/damianp/eos/project/d/dshep/LCD/V1/*scan/*.h5 --weightsdir=/scratch/04653/damianp/3DGAN_wheel --batchsize 8 -lr 0.001 --latentsize 200 --intraop 11 --interop 1 --warmup 5 --nbepochs 25 --optimizer=RMSprop

module list
