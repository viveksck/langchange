import random
import os
from multiprocessing import Process, Lock

from googlengram import matstore, util

import numpy as np
cimport numpy as np

DATA_DIR = '/dfs/scratch0/google_ngrams/'
INPUT_DIR = DATA_DIR + '/5grams_ppmi_lsmooth_fixed9/'
OUTPUT_PREFIX = DATA_DIR + "/stats/interesting-clust-lfixed9"
TMP_DIR = '/lfs/madmax5/0/will/google_ngrams/tmp/'
WORD_FILE = DATA_DIR + "info/interestingwords.pkl"

def get_word_indices(word_list):
    common_indices = [MERGED_INDEX[word] for word in word_list]
    common_indices = sorted(common_indices)
    return np.array(common_indices)

WORDS = util.load_pickle(WORD_FILE) 
MERGED_INDEX = util.load_pickle(DATA_DIR + "5grams_merged/merged_index.pkl")
CONTEXT_INDICES = get_word_indices(WORDS)
YEARS = range(1900, 2001)

def compute_word_stats(mat, word, context_indices):
    word_i = MERGED_INDEX[word]
    if word_i >= mat.shape[0]:
        return -1, -1 
    vec = mat[word_i, :]
    indices = vec.nonzero()[1]
    indices = np.intersect1d(indices, context_indices, assume_unique=True)
    if len(indices) <= 1:
        return 0, 0
    weights = vec[:, indices]
    weights /= weights.sum()
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
        binary_yearstat = util.load_pickle(TMP_DIR + str(year) + "-binary.pkl")
        weighted_yearstat = util.load_pickle(TMP_DIR + str(year) + "-weighted.pkl")
        for word in WORDS:
            binary_yearstats[word][year] = binary_yearstat[word]
            weighted_yearstats[word][year] = weighted_yearstat[word]
        os.remove(TMP_DIR + str(year) + "-binary.pkl")
        os.remove(TMP_DIR + str(year) + "-weighted.pkl")
    util.write_pickle(binary_yearstats, OUTPUT_PREFIX + "-binary.pkl")
    util.write_pickle(weighted_yearstats, OUTPUT_PREFIX + "-weighted.pkl")

def main(proc_num, lock):
    years = range(YEARS[0], YEARS[-1] + 1)
    random.shuffle(years)
    print proc_num, "Start loop"
    while True:
        lock.acquire()
        work_left = False
        for year in years:
            dirs = set(os.listdir(TMP_DIR))
            if str(year) + "-binary.pkl" in dirs:
                continue
            work_left = True
            print proc_num, "year", year
            fname = TMP_DIR + str(year) + "-binary.pkl"
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
        util.write_pickle(weighted_word_stats, TMP_DIR + str(year) + "-weighted.pkl")
        util.write_pickle(binary_word_stats, TMP_DIR + str(year) + "-binary.pkl")

def run_parallel(num_procs):
    lock = Lock()
    procs = [Process(target=main, args=[i, lock]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print "Merging"
    merge()
