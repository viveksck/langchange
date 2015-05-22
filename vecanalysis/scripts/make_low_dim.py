import sys

import numpy as np

from vecanalysis.representations.embedding import Embedding
from vecanalysis.alignment import smart_procrustes_align
from vecanalysis.dimreduce import reduce_dim

INPUT_DIR = "/dfs/scratch0/google_ngrams/vecs-fixed-np/"
OUTPUT_DIR = "/dfs/scratch0/google_ngrams/vecs-fixed-lowdim/"
BASE_YEAR = 2008
INPUT_PATH = INPUT_DIR + '{year}-300vecs'
OUTPUT_PATH = OUTPUT_DIR + '{year}-vecs'

if __name__ == '__main__':
    year = sys.argv[1]
    print "Loading embeddings..."
    base_embed = Embedding.load(INPUT_PATH.format(year=BASE_YEAR))
    other_embed = Embedding.load(INPUT_PATH.format(year=year))
    print "Reducing dimensionalities..."
    base_embed = reduce_dim(base_embed)
    other_embed = reduce_dim(other_embed)
    print "Aligning..."
    aligned_embed = smart_procrustes_align(base_embed, other_embed)
    print "Writing..."
    foutname = OUTPUT_PATH.format(year=year)
    np.save(foutname+".npy",aligned_embed.m)
    with file(foutname+".vocab","w") as outf:
       print >> outf, " ".join(aligned_embed.iw)


