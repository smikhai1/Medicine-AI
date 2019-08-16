#!/bin/bash
#PBS -N unet-crossentr
#PBS -l nodes=gpu1:ppn=8:gpus=4
#PBS -l pmem=24gb
#PBS -l walltime=08:00:00
#PBS -q gpgpu
#PBS -e /home/Mikhail.Sidorenko/logs/errors.txt
#PBS -o /home/Mikhail.Sidorenko/logs/output3.txt
 

cd $PBS_O_WORKDIR
singularity exec --nv ~/pytorch_19.04-py3.sif python ./src/train.py --config ./configs/config.yml --paths ./configs/path.yml --fold 1
