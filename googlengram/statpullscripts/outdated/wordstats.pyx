import random
import os
import matstore
import util
from multiprocessing import Process, Lock

import numpy as np
cimport numpy as np

DATA_DIR = '/dfs/scratch0/google_ngrams/'
INPUT_DIR = DATA_DIR + '/5grams_ppmi/'
OUTPUT_FILE = DATA_DIR + "/info/basicwordstats.pkl"
TMP_DIR = '/lfs/madmax4/0/will/google_ngrams/tmp/'
COMMON_WORD_FILE = DATA_DIR + "info/commonwords-90.pkl"

WORDS = ["maid", "skyline", "hound", "flesh", "shit", "campaign", "seed", "princess", "blood", "ocean", "trend", "like", "transmit", "rough", "maiden", "twisted", "thermos", "money", "voyage"]

def get_common_word_indices():
    common_words = util.load_pickle(COMMON_WORD_FILE) 
    cdef np.ndarray common_indices = np.empty(len(common_words), dtype=np.int64) 
    cdef int i
    for i in xrange(len(common_words)):
        common_indices[i] = MERGED_INDEX[common_words[i]]
    return common_indices

MERGED_INDEX = util.load_pickle(DATA_DIR + "5grams_merged/merged_index.pkl")
COMMON_INDICES = get_common_word_indices()
YEARS = range(1700, 2009)

def compute_word_stats(mat, word):
    word_i = MERGED_INDEX[word]
    if word_i >= mat.shape[0]:
        return {'avg' : 0, 'avg_c' : 0, 'nnzprop' : 0, 'nnzprop_c' : 0}
    mat = mat.tocsr()
    mat.prune()
    vec = mat[word_i, :]
    common_indices = COMMON_INDICES[COMMON_INDICES < min(mat.shape[0], mat.shape[1])]
    c_vec = vec[:, common_indices]
    avg = vec.mean()
    avg_c = c_vec.mean()
    nnzprop = vec.getnnz() / float(vec.shape[1])
    nnzprop_c = c_vec.getnnz() / float(c_vec.shape[1])
    return {'avg' : avg, 'avg_c' : avg_c, 'nnzprop' : nnzprop, 'nnzprop_c' : nnzprop_c}

def merge():
    yearstats = {}
    for word in WORDS:
        yearstats[word] = {'avg' : {}, 'avg_c' : {}, 'nnzprop' : {}, 'nnzprop_c' : {}}
    for year in YEARS:
        yearstat = util.load_pickle(TMP_DIR + str(year) + ".pkl")
        for word in WORDS:
            yearstats[word]['avg'][year] = yearstat[word]['avg']
            yearstats[word]['avg_c'][year] = yearstat[word]['avg_c']
            yearstats[word]['nnzprop'][year] = yearstat[word]['nnzprop']
            yearstats[word]['nnzprop_c'][year] = yearstat[word]['nnzprop_c']
        os.remove(TMP_DIR + str(year) + ".pkl")
    util.write_pickle(yearstats, OUTPUT_FILE)

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
        print "Getting stats for year.."
        for word in WORDS:
            word_stats[word] = compute_word_stats(mat, word)

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
