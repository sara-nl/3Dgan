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
		-lr 0.001 --optimizer=Adam \
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