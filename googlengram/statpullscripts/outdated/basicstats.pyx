import random
import os
import matstore
import util
from multiprocessing import Process, Lock

import numpy as np
cimport numpy as np

DATA_DIR = '/dfs/scratch0/google_ngrams/'
PPMI_DIR = DATA_DIR + '/5grams_ppmi/'
COUNT_DIR = DATA_DIR + '/5grams_merged/'
OUTPUT_DIR = DATA_DIR + "/info/"
TMP_DIR = '/lfs/madmax4/0/will/google_ngrams/tmp/'
COMMON_WORD_FILE = DATA_DIR + "info/commonwords-90.pkl"

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

def get_basic_stats(ppmi_mat, count_mat):
    ppmi_mat = count_mat.tocsr()
    count_mat = count_mat.tocsr()
    cdef float dim = (ppmi_mat.getnnz(1) > 0).sum()
    common_indices = COMMON_INDICES[COMMON_INDICES < min(ppmi_mat.shape[0], ppmi_mat.shape[1])]
    c_ppmi_mat = ppmi_mat[common_indices]
    c_ppmi_mat = c_ppmi_mat[:, common_indices]
    c_ppmi_mat.prune()
    c_count_mat = count_mat[common_indices]
    c_count_mat = c_count_mat[:, common_indices]
    c_count_mat.prune()

    cdef float c_dim = (c_ppmi_mat.getnnz(1) > 0).sum()
    avg = ppmi_mat.sum() / (dim * dim)
    avg_c = c_ppmi_mat.sum() / (c_dim * c_dim)
    mi = (ppmi_mat.multiply(count_mat).sum()) / count_mat.sum()
    mi_c = (c_ppmi_mat.multiply(c_count_mat).sum()) / count_mat.sum()
    nnzprop = ppmi_mat.getnnz() / (dim * dim)
    nnzprop_c = c_ppmi_mat.getnnz() / (c_dim * c_dim)
    stats =  {'avg' : avg, 'avg_c' : avg_c, 'nnzprop' : nnzprop, 'nnzprop_c' : nnzprop_c, 'mi' : mi, 'mi_c' : mi_c}
    return stats

def merge(k):
    yearstats = {'avg' : {}, 'avg_c' : {}, 'nnzprop' : {}, 'nnzprop_c' : {}, 'mi' : {}, 'mi_c' : {}}
    for year in YEARS:
        yearstat = util.load_pickle(TMP_DIR + str(year) + ".pkl")
        yearstats['avg'][year] = yearstat['avg']
        yearstats['avg_c'][year] = yearstat['avg_c']
        yearstats['mi'][year] = yearstat['mi']
        yearstats['mi_c'][year] = yearstat['mi_c']
        yearstats['nnzprop'][year] = yearstat['nnzprop']
        yearstats['nnzprop_c'][year] = yearstat['nnzprop_c']
        os.remove(TMP_DIR + str(year) + ".pkl")
    util.write_pickle(yearstats, OUTPUT_DIR + "basicyearstats-" + str(k) + ".pkl")

def main(proc_num, lock, k):
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

        print proc_num, "Getting stats for year", year
        ppmi_mat = matstore.retrieve_cooccurrence_as_coo_thresh(PPMI_DIR + str(year) + ".bin", np.log(float(k)))
        count_mat = matstore.retrieve_cooccurrence_as_coo_thresh(COUNT_DIR + str(year) + ".bin", np.log(float(k)))
        stats = get_basic_stats(ppmi_mat, count_mat)

        print proc_num, "Writing stats for year", year
        util.write_pickle(stats, TMP_DIR + str(year) + ".pkl")


def run_parallel(num_procs, k):
    lock = Lock()
    procs = [Process(target=main, args=[i, lock, k]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print "Merging"
    merge(k)
