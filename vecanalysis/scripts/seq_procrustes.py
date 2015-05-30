import numpy as np

from vecanalysis import alignment
from vecanalysis.representations import embedding

DATA_DIR = '/dfs/scratch0/google_ngrams/'
INPUT_DIR = DATA_DIR + '/sglove-vecs-smallrel-np/'
OUTPUT_DIR = DATA_DIR + '/sglove-vecs-smallrel-aligned-seq/'
INPUT_FILE = INPUT_DIR + '{year}-300vecs'
OUTPUT_FILE = OUTPUT_DIR + '{year}-300vecs'

def align_years(years):
    first_iter = True
    base_embed = None
    for year in years:
        print "Loading year:", year
        year_embed = embedding.Embedding.load(INPUT_FILE.format(year=year))
        print "Aligning year:", year
        if first_iter:
            aligned_embed = year_embed
            first_iter = False
        else:
            aligned_embed = alignment.smart_procrustes_align(base_embed, year_embed)
        base_embed = aligned_embed
        print "Writing year:", year
        foutname = OUTPUT_FILE.format(year=year)
        np.save(foutname+".npy",aligned_embed.m)
        with file(foutname+".vocab","w") as outf:
           print >> outf, " ".join(aligned_embed.iw)

