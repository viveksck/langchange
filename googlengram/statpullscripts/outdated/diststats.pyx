import random
import os
import matstore
import util
import time
import random
from multiprocessing import Process, Lock
from sklearn.preprocessing import normalize
import scipy as sp

import numpy as np
cimport numpy as np

DATA_DIR = '/dfs/scratch0/google_ngrams/'
INPUT_DIR = DATA_DIR + '/5grams_ppmi/'
INPUT_DIR_2 = DATA_DIR + '/5grams_merged/'
OUTPUT_FILE = DATA_DIR + "/info/distwordstats-random1000.pkl"
TMP_DIR = '/lfs/madmax2/0/will/google_ngrams/tmp/'
COMMON_WORD_FILE = DATA_DIR + "info/interestingwords.pkl"
#WORD_FILE = DATA_DIR + "info/freqsorted.tsv"

#WORDS = ["maid", "skyline", "hound", "flesh", "shit", "campaign", "seed", "princess", "blood", "ocean", "trend", "like", "transmit", "rough", "maiden", "twisted", "thermos", "money", "voyage"]
#WORDS.extend(["homozygous", "viral", "heterozygous", "cardiac", "commutative", "orthogonal", "embedding", "crystallography", "algebraic", "calculus", "euclidean", "phenotype"])
START_SKIP = 100
NUM_KEEP = 1000
ACCEPT_PROB = 0.1
COMMON_WORDS = util.load_pickle(COMMON_WORD_FILE) 

def get_words(word_file):
    word_fp = open(word_file, "r")
    count = 0
    words = []

    for line in word_fp:
        count += 1
        if count < START_SKIP:
            continue
        word = line.strip().split("\t")[0].decode('utf-8')
        if not word.isalpha():
            continue
        rand = random.random()
        if rand < ACCEPT_PROB:
            words.append(word)
        if len(words) >= NUM_KEEP:
            break
    return words

def get_words_2():
    words = []
    count = 0
    for word in COMMON_WORDS:
        count += 1
        rand = random.random()
        if rand < ACCEPT_PROB:
            words.append(word)
        if len(words) > NUM_KEEP:
            break

WORDS = get_words_2()

def get_common_word_indices():
    cdef int i
    common_indices = []
    for i in xrange(len(COMMON_WORDS)):
#        if COMMON_WORDS[i] in WORDS:
#            continue
        common_indices.append(MERGED_INDEX[COMMON_WORDS[i]])
#    for word in WORDS:
#        common_indices.append(MERGED_INDEX[word])
    return np.array(common_indices)

MERGED_INDEX = util.load_pickle(DATA_DIR + "5grams_merged/merged_index.pkl")
COMMON_INDICES = get_common_word_indices()
YEARS = range(1820, 2009)

def compute_dist_stats(indices, weights, mat):
    cdef int i, index
    cdef float avgdist = 0
    if len(indices) == 0:
        return 0
    for i in xrange(len(indices)):
        index = indices[i]
        if index >= mat.shape[0]:
            continue
        row = mat[index, :] 
        avgdist += weights[0,index] * ((row[0,indices]).sum() - row[0, index]) / float(row.shape[1] - 1) 
    avgdist = avgdist / float(len(indices))
    return avgdist
        
def compute_word_stats(mat, mat2, word):
    word_i = MERGED_INDEX[word]
    if word_i >= mat.shape[0]:
        return -1 
    vec = mat[word_i, :]
    indices = vec.nonzero()[1]
    if len(indices) <= 1:
        return 0
    weights = (vec/vec.sum())[:, indices]
#    weights = vec[:,indices]
    reduced = mat[:, indices]
    reduced = reduced[indices, :]
    return (weights * reduced).sum() / (float(len(indices)))
#    return reduced.nnz / (float(len(indices)) * (float(len(indices)) - 1))

def merge():
    yearstats = {}
    for word in WORDS:
        yearstats[word] = {}
    for year in YEARS:
        yearstat = util.load_pickle(TMP_DIR + str(year) + "-dist.pkl")
        for word in WORDS:
            yearstats[word][year] = yearstat[word]
        os.remove(TMP_DIR + str(year) + "-dist.pkl")
    util.write_pickle(yearstats, OUTPUT_FILE)

def make_inverse_freq_mat(freq_mat):
    cdef int i
    cdef float val
    cdef float sum = freq_mat.sum()
    cdef np.ndarray[np.float64_t, ndim=1] new_data = np.empty((len(freq_mat.data),))
    for i in xrange(freq_mat.data.shape[0]): 
        val = np.log(freq_mat.data[i] / sum) 
        new_data[i] = -1*val**(-1.0)
    return new_data

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
            if str(year) + "-dist.pkl" in dirs:
                continue
            work_left = True
            print proc_num, "year", year
            fname = TMP_DIR + str(year) + "-dist.pkl"
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
        mat2 = matstore.retrieve_cooccurrence_as_coo(INPUT_DIR_2 + str(year) + ".bin")
        print proc_num, "Making inverse freq mat", year
        new_data = make_inverse_freq_mat(mat2)
        mat2.data = new_data
        mat2 = mat2.tocsr()
        mat2.prune()
        common_indices = COMMON_INDICES[COMMON_INDICES < min(mat.shape[0], mat.shape[1])]
        mat = mat.tocsr()
        mat2 = mat2[:, common_indices]
        mat = mat[:, common_indices]
        mat.prune()
        mat = mat.multiply(mat2)
#        mat = normalize(mat, copy=False, norm='l1')
        word_stats = {}
        print proc_num, "Getting stats for year", year
        for word in WORDS:
            word_stats[word] = compute_word_stats(mat, mat2, word)

        print proc_num, "Writing stats for year", year
        util.write_pickle(word_stats, TMP_DIR + str(year) + "-dist.pkl")


def run_parallel(num_procs):
    lock = Lock()
    procs = [Process(target=main, args=[i, lock]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print "Merging"
    merge()
