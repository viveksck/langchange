import random
import os
from multiprocessing import Process, Lock

import numpy as np

from googlengram import matstore, util

DATA_DIR = '/dfs/scratch0/google_ngrams/'
PPMI_DIR = DATA_DIR + '/5grams_ppmi_lsmooth/'
OUTPUT_DIR = DATA_DIR + "/info/"
TMP_DIR = '/lfs/madmax5/0/will/google_ngrams/tmp/'
COMMON_WORD_FILE = DATA_DIR + "info/interestingwords.pkl"

def get_word_indices(word_file):
    common_words = util.load_pickle(word_file) 
    common_indices = [MERGED_INDEX[word] for word in common_words]
    common_indices = sorted(common_indices)
    return np.array(common_indices)

MERGED_INDEX = util.load_pickle(DATA_DIR + "5grams_merged/merged_index.pkl")
COMMON_INDICES = get_word_indices(COMMON_WORD_FILE)
YEARS = range(1850, 2001)

def get_basic_stats(ppmi_mat):
    ppmi_mat = ppmi_mat.tocsr()
    dim = (ppmi_mat.getnnz(1) > 0).sum()
    common_indices = COMMON_INDICES[COMMON_INDICES < min(ppmi_mat.shape[0], ppmi_mat.shape[1])]
    c_ppmi_mat = ppmi_mat[common_indices, :]
    c_ppmi_mat = c_ppmi_mat[:, common_indices]
    c_ppmi_mat.prune()
    c_dim = (c_ppmi_mat.getnnz(1) > 0).sum()

    avg = ppmi_mat.sum() / (dim * dim)
    avg_c = c_ppmi_mat.sum() / (c_dim * c_dim)
    nnzprop = float(ppmi_mat.getnnz()) / (dim * dim)
    nnzprop_c = float(c_ppmi_mat.getnnz()) / (c_dim * c_dim)
    stats =  {'avg' : avg, 'avg_c' : avg_c, 'nnzprop' : nnzprop, 'nnzprop_c' : nnzprop_c}
    return stats

def merge():
    yearstats = {'avg' : {}, 'avg_c' : {}, 'nnzprop' : {}, 'nnzprop_c' : {}}
    for year in YEARS:
        yearstat = util.load_pickle(TMP_DIR + str(year) + ".pkl")
        yearstats['avg'][year] = yearstat['avg']
        yearstats['avg_c'][year] = yearstat['avg_c']
        yearstats['nnzprop'][year] = yearstat['nnzprop']
        yearstats['nnzprop_c'][year] = yearstat['nnzprop_c']
        os.remove(TMP_DIR + str(year) + ".pkl")
    util.write_pickle(yearstats, OUTPUT_DIR + "basicyearstats.pkl")

def main(proc_num, lock):
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
        ppmi_mat = matstore.retrieve_cooccurrence_as_coo(PPMI_DIR + str(year) + ".bin")
        stats = get_basic_stats(ppmi_mat)
        print proc_num, "Writing stats for year", year
        util.write_pickle(stats, TMP_DIR + str(year) + ".pkl")


def run_parallel(num_procs):
    lock = Lock()
    procs = [Process(target=main, args=[i, lock]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print "Merging"
    merge()
