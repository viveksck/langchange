import random
import os
from multiprocessing import Process, Lock

from googlengram import matstore, util

import numpy as np

DATA_DIR = '/dfs/scratch0/google_ngrams/'
INPUT_DIR = DATA_DIR + '/5grams_ppmi_lsmooth_fixed/'
OUTPUT_PREFIX = DATA_DIR + "/stats/interesting-neigh-fixed"
TMP_DIR = '/lfs/madmax7/0/will/google_ngrams/tmp/'
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
    return vec.nnz, vec.sum(), vec.sum() / (float(vec.nnz)) 

def merge():
    degree_yearstats = {}
    sum_yearstats = {}
    avg_yearstats = {}
    for word in WORDS:
        degree_yearstats[word] = {}
        sum_yearstats[word] = {}
        avg_yearstats[word] = {}
    for year in YEARS:
        degree_yearstat = util.load_pickle(TMP_DIR + str(year) + "-deg.pkl")
        sum_yearstat = util.load_pickle(TMP_DIR + str(year) + "-sum.pkl")
        avg_yearstat = util.load_pickle(TMP_DIR + str(year) + "-avg.pkl")
        for word in WORDS:
            degree_yearstats[word][year] = degree_yearstat[word]
            sum_yearstats[word][year] = sum_yearstat[word]
            avg_yearstats[word][year] = avg_yearstat[word]
        os.remove(TMP_DIR + str(year) + "-deg.pkl")
        os.remove(TMP_DIR + str(year) + "-sum.pkl")
        os.remove(TMP_DIR + str(year) + "-avg.pkl")
    util.write_pickle(degree_yearstats, OUTPUT_PREFIX + "-deg.pkl")
    util.write_pickle(sum_yearstats, OUTPUT_PREFIX + "-sum.pkl")
    util.write_pickle(avg_yearstats, OUTPUT_PREFIX + "-avg.pkl")

def main(proc_num, lock):
    years = range(YEARS[0], YEARS[-1] + 1)
    random.shuffle(years)
    print proc_num, "Start loop"
    while True:
        lock.acquire()
        work_left = False
        for year in years:
            dirs = set(os.listdir(TMP_DIR))
            if str(year) + "-deg.pkl" in dirs:
                continue
            work_left = True
            print proc_num, "year", year
            fname = TMP_DIR + str(year) + "-deg.pkl"
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
        degree_word_stats = {}
        sum_word_stats = {}
        avg_word_stats = {}
        print proc_num, "Getting stats for year", year
        for word in WORDS:
            degree, sum, avg = compute_word_stats(mat, word, context_indices)
            degree_word_stats[word] = degree
            sum_word_stats[word] = sum
            avg_word_stats[word] = avg

        print proc_num, "Writing stats for year", year
        util.write_pickle(degree_word_stats, TMP_DIR + str(year) + "-deg.pkl")
        util.write_pickle(sum_word_stats, TMP_DIR + str(year) + "-sum.pkl")
        util.write_pickle(avg_word_stats, TMP_DIR + str(year) + "-avg.pkl")

def run_parallel(num_procs):
    lock = Lock()
    procs = [Process(target=main, args=[i, lock]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print "Merging"
    merge()
