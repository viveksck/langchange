#!/bin/bash
#PBS -N makew2vtrain
#PBS -l nodes=1:ppn=1
#PBS -l walltime=02:00:00

/dfs/scratch0/wleif/anaconda/bin/python /dfs/scratch0/wleif/langchange/run_makew2vtrain.py $PBS_ARRAYID
