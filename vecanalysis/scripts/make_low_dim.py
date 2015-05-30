import sys

import numpy as np
from sklearn.decomposition import TruncatedSVD

from googlengram import util
from vecanalysis.representations.explicit import Explicit

INPUT_DIR = "/dfs/scratch0/google_ngrams/5grams_ppmi_lsmooth_fixed/"
OUTPUT_DIR = "/dfs/scratch0/google_ngrams/vecs-svd/"
INPUT_PATH = INPUT_DIR + '{year}.bin'
OUTPUT_PATH = OUTPUT_DIR + '{year}-300vecs'

if __name__ == '__main__':
    year = sys.argv[1]
    print "Loading embeddings for year", year
    words = util.load_pickle("/dfs/scratch0/google_ngrams/info/interestingwords.pkl")
    base_embed = Explicit.load(INPUT_PATH.format(year=year), restricted_context=words)
    print "SVD for year", year
    pca = TruncatedSVD(n_components=300)
    new_mat = pca.fit_transform(base_embed.m)
    print "Saving year", year
    np.save(OUTPUT_PATH.format(year=year) + ".npy", new_mat)
    vocab_outfp = open(OUTPUT_PATH.format(year=year) + ".vocab", "w")
    vocab_outfp.write(" ".join(base_embed.iw))


