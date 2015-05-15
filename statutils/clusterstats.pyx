import random
import os
import time
import random
from multiprocessing import Process, Lock
from sklearn.preprocessing import normalize
import scipy as sp

from googlengram import matstore, util

import numpy as np
cimport numpy as np

DATA_DIR = '/dfs/scratch0/google_ngrams/'
INPUT_DIR = DATA_DIR + '/5grams_ppmi_smooth/'
OUTPUT_PREFIX = DATA_DIR + "/info/fullfixed"
TMP_DIR = '/lfs/madmax5/0/will/google_ngrams/tmp/'
TARGET_WORD_FILE = DATA_DIR + "info/interestingwords.pkl"
CONTEXT_WORD_FILE = DATA_DIR + "info/interestingwords.pkl"
#CONTEXT_WORD_FILE = DATA_DIR + "info/relevantwords.pkl"

NUM_KEEP = 1000
ACCEPT_PROB = 0.3
random.seed(10)

## LOAD FUNCTIONS ##
def get_words():
    words = []
    count = 0
    for word in TARGET_WORDS:
        count += 1
        rand = random.random()
        if rand < ACCEPT_PROB:
            words.append(word)
        if len(words) > NUM_KEEP:
            break
    return words

def get_context_indices():
    cdef int i
    context_indices = []
    for i in xrange(len(CONTEXT_WORDS)):
        context_indices.append(MERGED_INDEX[CONTEXT_WORDS[i]])
    return np.array(context_indices)

TARGET_WORDS = util.load_pickle(TARGET_WORD_FILE) 
#WORDS = get_words()
WORDS = TARGET_WORDS
CONTEXT_WORDS = util.load_pickle(CONTEXT_WORD_FILE)
MERGED_INDEX = util.load_pickle(DATA_DIR + "5grams_merged/merged_index.pkl")
CONTEXT_INDICES = get_context_indices()
YEARS = range(1850, 2009)

def compute_word_stats(mat, word, context_indices):
    word_i = MERGED_INDEX[word]
    if word_i >= mat.shape[0]:
        return -1, -1 
    vec = mat[word_i, :]
    indices = vec.nonzero()[1]
    indices = np.intersect1d(indices, context_indices, assume_unique=True)
    if len(indices) <= 1:
        return 0, 0
    weights = (vec/vec.sum())[:, indices]
    reduced = mat[indices, :]
    reduced = reduced[:, indices]
    return (weights * reduced).sum() / (float(len(indices)) - 1), float(reduced.nnz) / (len(indices) * (len(indices) - 1))

def merge():
    binary_yearstats = {}
    weighted_yearstats = {}
    for word in WORDS:
        binary_yearstats[word] = {}
        weighted_yearstats[word] = {}
    for year in YEARS:
        binary_yearstat = util.load_pickle(TMP_DIR + str(year) + "-binary-dist.pkl")
        weighted_yearstat = util.load_pickle(TMP_DIR + str(year) + "-weighted-dist.pkl")
        for word in WORDS:
            binary_yearstats[word][year] = binary_yearstat[word]
            weighted_yearstats[word][year] = weighted_yearstat[word]
        os.remove(TMP_DIR + str(year) + "-binary-dist.pkl")
        os.remove(TMP_DIR + str(year) + "-weighted-dist.pkl")
    util.write_pickle(binary_yearstats, OUTPUT_PREFIX + "-binary-dist.pkl")
    util.write_pickle(weighted_yearstats, OUTPUT_PREFIX + "-weighted-dist.pkl")

def main(proc_num, lock):
    cdef int i
    years = range(YEARS[0], YEARS[-1] + 1)
    random.shuffle(years)
    print proc_num, "Start loop"
    while True:
        lock.acquire()
        work_left = False
        for year in years:
            dirs = set(os.listdir(TMP_DIR))
            if str(year) + "-binary-dist.pkl" in dirs:
                continue
            work_left = True
            print proc_num, "year", year
            fname = TMP_DIR + str(year) + "-binary-dist.pkl"
            with open(fname, "w") as fp:
                fp.write("")
            fp.close()
            break
        lock.release()
        if not work_left:
            print proc_num, "Finished"
            break

        print proc_num, "Retrieving mat for year", year
        mat = matstore.retrieve_cooccurrence_as_coo(INPUT_DIR + str(year) + ".bin")
        mat.setdiag(0)
        context_indices = CONTEXT_INDICES[CONTEXT_INDICES < min(mat.shape[1], mat.shape[0])]
        mat = mat.tocsr()
        mat.eliminate_zeros()
        weighted_word_stats = {}
        binary_word_stats = {}
        print proc_num, "Getting stats for year", year
        for word in WORDS:
            weighted, binary = compute_word_stats(mat, word, context_indices)
            weighted_word_stats[word] = weighted
            binary_word_stats[word] = binary

        print proc_num, "Writing stats for year", year
        util.write_pickle(weighted_word_stats, TMP_DIR + str(year) + "-weighted-dist.pkl")
        util.write_pickle(binary_word_stats, TMP_DIR + str(year) + "-binary-dist.pkl")

def run_parallel(num_procs):
    lock = Lock()
    procs = [Process(target=main, args=[i, lock]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print "Merging"
    merge()
