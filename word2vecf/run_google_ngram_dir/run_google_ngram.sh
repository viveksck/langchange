#!/bin/bash
#PBS -N ngramw2v
#PBS -l nodes=1:ppn=30
#PBS -l walltime=05:00:00

/dfs/scratch0/wleif/anaconda/bin/python /dfs/scratch0/wleif/langchange/run_google_ngram.py $PBS_ARRAYID
