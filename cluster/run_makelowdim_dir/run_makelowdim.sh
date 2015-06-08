#!/bin/bash
#PBS -N makelowdim
#PBS -l nodes=1:ppn=1
#PBS -l walltime=02:00:00

cd /dfs/scratch0/wleif/langchange
/dfs/scratch0/wleif/anaconda/bin/python -m vecanalysis.scripts.make_low_dim $PBS_ARRAYID
