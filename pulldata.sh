#!/bin/bash

NUM_PROCS=75
python -m googlengrams.pullscripts.newgrabscript $1/5grams_unmerged $2 $NUM_PROCS
python -m googlengrams.pullscripts.runmerge $1/5grams_unmerged $1/5grams_unmerged/$2/20120701/5grams $NUM_PROCS
mkdir $1/5grams_merged
python -m googlengrams.pullscripts.indexmerge $1/5grams_merged $1/5grams_unmerged
python -m googlengrams.pullscripts.indexmerge $1/5grams_merged $1/5grams_unmerged

