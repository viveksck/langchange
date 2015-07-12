#!/bin/bash
#PBS -N bootstrapstats
#PBS -l nodes=1:ppn=32
#PBS -l walltime=12:00:00

cd /dfs/scratch0/wleif/langchange
/dfs/scratch0/wleif/anaconda/bin/python -m /dfs/scratch0/googlengrams/2012-eng-fic/5grams_sym/ /dfs/scratch0/googlengrams/2012-eng-fic/info/freqnonstop_peryear-1900-2000-7.pkl /dfs/scratch0/googlengrams/2012-eng-fic/info/samplesizes-top20000.pkl 25 --num-words 20000 --num-boots 4 --smooth 10 --id $PBS_ARRAYID
