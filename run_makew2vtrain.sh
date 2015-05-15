#!/bin/bash
#PBS -N makew2vtrain
#PBS -l nodes=1:ppn=1
#PBS -l walltime=02:00:00

$HOME/anaconda/bin/python $HOME/langchange/run_makew2vtrain.py $PBS_ARRAYID

