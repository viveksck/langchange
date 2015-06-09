import random
import os
import collections
import numpy as np
from multiprocessing import Process, Lock

from googlengram import matstore, util

DATA_DIR = '/dfs/scratch0/google_ngrams/'
INPUT_DIR = DATA_DIR + '/5grams_merged/'
OUTPUT_FILE = DATA_DIR + "/info/samplesizes.pkl"
TMP_DIR = '/lfs/madmax5/0/will/google_ngrams/tmp/'
INTER_WORD_FILE = DATA_DIR + "info/interestingwords.pkl"
REL_WORD_FILE = DATA_DIR + "info/relevantwords.pkl"
MERGED_INDEX = util.load_pickle(DATA_DIR + "5grams_merged/merged_index.pkl")

def get_word_indices(word_file):
    common_words = util.load_pickle(word_file) 
    common_indices = []
    for i in xrange(len(common_words)):
        common_indices.append(MERGED_INDEX[common_words[i]])
    return np.array(common_indices)
INTER_INDICES = get_word_indices(INTER_WORD_FILE)
REL_INDICES = get_word_indices(REL_WORD_FILE)

YEARS = range(1850, 2009)

def merge():
    yearstats = collections.defaultdict(dict)
    for year in YEARS:
        yearstat = util.load_pickle(TMP_DIR + str(year) + "-sizes.pkl")
        for stat in yearstat:
            yearstats[stat][year] = yearstat[stat]
        os.remove(TMP_DIR + str(year) + "-sizes.pkl")
    util.write_pickle(yearstats, OUTPUT_FILE)

def main(proc_num, lock):
    years = range(YEARS[0], YEARS[-1] + 1)
    random.shuffle(years)
    print proc_num, "Start loop"
    while True:
        lock.acquire()
        work_left = False
        for year in years:
            dirs = set(os.listdir(TMP_DIR))
            if str(year) + "-sizes.pkl" in dirs:
                continue
            work_left = True
            print proc_num, "year", year
            fname = TMP_DIR + str(year) + "-sizes.pkl"
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
        print proc_num, "Making inverse freq mat", year
        mat = mat.tocsr()
        single_year_stats = {}
        print proc_num, "Getting stats for year", year
        single_year_stats["total"] = mat.sum()
        rel_col_mat = mat[:, REL_INDICES]
        single_year_stats["rel-rel"] = rel_col_mat[REL_INDICES, :].sum()
        single_year_stats["inter-rel"] = rel_col_mat[INTER_INDICES, :].sum()
        inter_row_mat = mat[INTER_INDICES, :]
        single_year_stats["inter-inter"] = inter_row_mat[:, INTER_INDICES].sum()

        print proc_num, "Writing stats for year", year
        util.write_pickle(single_year_stats, TMP_DIR + str(year) + "-sizes.pkl")


def run_parallel(num_procs):
    lock = Lock()
    procs = [Process(target=main, args=[i, lock]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print "Merging"
    merge()
