import random
import os
import random
from multiprocessing import Process, Lock
from sklearn.preprocessing import normalize
import scipy as sp

from googlengram import matstore, util

import numpy as np
cimport numpy as np

DATA_DIR = '/dfs/scratch0/google_ngrams/'
INPUT_DIR = DATA_DIR + '/5grams_merged/'
OUTPUT_FILE = DATA_DIR + "/info/interestingfreqs.pkl"
TMP_DIR = '/dfs/scratch0/wleif/tmp/'
WORD_FILE = DATA_DIR + "info/interestingwords.pkl"

WORDS = util.load_pickle(WORD_FILE) 
MERGED_INDEX = util.load_pickle(DATA_DIR + "5grams_merged/merged_index.pkl")
YEARS = range(1850, 2009)

def compute_word_stats(mat, word):
    word_i = MERGED_INDEX[word]
    if word_i >= mat.shape[0]:
        return 0
    return mat[word_i, :].sum()

def merge():
    yearstats = {}
    for word in WORDS:
        yearstats[word] = {}
    for year in YEARS:
        yearstat = util.load_pickle(TMP_DIR + str(year) + "-freqs.pkl")
        for word in WORDS:
            yearstats[word][year] = yearstat[word]
        os.remove(TMP_DIR + str(year) + "-freqs.pkl")
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
            if str(year) + "-freqs.pkl" in dirs:
                continue
            work_left = True
            print proc_num, "year", year
            fname = TMP_DIR + str(year) + "-freqs.pkl"
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
        mat = mat / mat.sum()
        word_stats = {}
        print proc_num, "Getting stats for year", year
        for word in WORDS:
            word_stats[word] = compute_word_stats(mat, word)

        print proc_num, "Writing stats for year", year
        util.write_pickle(word_stats, TMP_DIR + str(year) + "-freqs.pkl")


def run_parallel(num_procs):
    lock = Lock()
    procs = [Process(target=main, args=[i, lock]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print "Merging"
    merge()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get frequency information")
