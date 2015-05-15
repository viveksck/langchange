import random
import os
import matstore
import collections
import operator
import util
from multiprocessing import Process, Lock

import numpy as np
cimport numpy as np

DATA_DIR = '/dfs/scratch0/google_ngrams/'
INPUT_DIR = DATA_DIR + '/5grams_ppmi/'
OUTPUT_FILE = DATA_DIR + "/info/densitysorted.tsv"
TMP_DIR = '/lfs/madmax5/0/will/google_ngrams/tmp/'
COMMON_WORD_FILE = DATA_DIR + "info/commonwords-90.pkl"

def get_common_word_indices():
    common_words = util.load_pickle(COMMON_WORD_FILE) 
    cdef np.ndarray common_indices = np.empty((len(common_words),), dtype=np.int64) 
    cdef int i
    for i in xrange(len(common_words)):
        common_indices[i] = MERGED_INDEX[common_words[i]]
    return common_indices

MERGED_INDEX = util.load_pickle(DATA_DIR + "5grams_merged/merged_index.pkl")
MERGED_LIST = list(MERGED_INDEX)
COMMON_INDICES = get_common_word_indices()
YEARS = range(1820, 2009)

def compute_word_stats(mat, word_i):
    if word_i >= mat.shape[0]:
        return 0
    vec = mat[word_i, :]
    return vec.nnz

def merge():
    yearstats = {}
    avgs = collections.defaultdict(float)
    for year in YEARS:
        yearstat = util.load_pickle(TMP_DIR + str(year) + ".pkl")
        for word, count in yearstat.iteritems():
            avgs[word] += count 
        os.remove(TMP_DIR + str(year) + ".pkl")
    for word in avgs:
        avgs[word] /= float(len(YEARS))
    sorted_avg = sorted(avgs.items(), key = operator.itemgetter(1), reverse=True)
    out_fp = open(OUTPUT_FILE, "w")
    for word, avg in sorted_avg:
        out_fp.write(word.encode('utf-8') + '\t' + str(avg) + "\n")

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
            if str(year) + ".pkl" in dirs:
                continue
            work_left = True
            print proc_num, "year", year
            fname = TMP_DIR + str(year) + ".pkl"
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
        word_stats = {}
        mat = mat.tocsr()
        mat.prune()
        mat = mat[:, COMMON_INDICES]
        print "Getting stats for year.."
        for word_i in COMMON_INDICES: 
            word_stats[MERGED_LIST[word_i]] = compute_word_stats(mat, word_i)

        print proc_num, "Writing stats for year", year
        util.write_pickle(word_stats, TMP_DIR + str(year) + ".pkl")


def run_parallel(num_procs):
    lock = Lock()
    procs = [Process(target=main, args=[i, lock]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print "Merging"
    merge()
