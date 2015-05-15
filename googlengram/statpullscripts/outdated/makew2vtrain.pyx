from .. import matstore, util
import collections
import random
import os
from multiprocessing import Process, Lock
import numpy as np
cimport numpy as np

random.seed(10)

DATA_DIR = '/dfs/scratch0/google_ngrams/'
INPUT_DIR = DATA_DIR + '5grams_merged/'
OUTPUT_DIR = DATA_DIR + 'w2vtrain-tmp/'
SAMPLES = 1000000000
BUFF_SAMPLES = 1000000000 
SAMPLE_BUFFER = 50000000
MIN_COUNT = 10
YEARS = range(2008, 2009)
MERGED_INDEX = util.load_pickle(INPUT_DIR + "merged_index.pkl")
TARGET_WORD_FILE = DATA_DIR + "info/relevantwords-100000.pkl"
CONTEXT_WORD_FILE = DATA_DIR + "info/relevantwords.pkl"
cdef float SUBSAMPLE = 10.0**(-5.0)

def get_word_indices(word_file):
    common_words = util.load_pickle(word_file) 
    common_indices = set([])
    for i in xrange(len(common_words)):
        common_indices.add(MERGED_INDEX[common_words[i]])
    return common_indices

TARGET_INDICES = get_word_indices(TARGET_WORD_FILE)
CONTEXT_INDICES = get_word_indices(CONTEXT_WORD_FILE)
MERGED_INDEX = list(MERGED_INDEX)

def main(proc_num, lock):
    years = range(YEARS[0], YEARS[-1] + 1)
    random.shuffle(years)
    print proc_num, "Start loop"
    while True:
        lock.acquire()
        work_left = False
        for year in years:
            dirs = set(os.listdir(OUTPUT_DIR))
            if str(year) + "-train.txt" in dirs:
                continue
            work_left = True
            print proc_num, "year", year
            fname = OUTPUT_DIR + str(year) + "-train.txt"
            with open(fname, "w") as fp:
                fp.write("")
            fp.close()
            break
        lock.release()
        if not work_left:
            print proc_num, "Finished"
            break
        train_data_for_year(year)

def train_data_for_year(year):
    print "Getting data for year", year
    year_mat_coo = matstore.retrieve_cooccurrence_as_coo(INPUT_DIR + str(year) + ".bin")
    og_sum = year_mat_coo.sum()
    subsampler = collections.defaultdict(float)
    print "Getting rows to remove..."
    year_mat_csr = year_mat_coo.tocsr()
    year_mat_csr.prune()
    rows_to_remove = set([])
    year_mat_csr = year_mat_csr.sum(1)
    cdef row_i
    num_subsampled = 0
    for row_i in xrange(year_mat_csr.shape[0]):
        if not row_i in TARGET_INDICES:
            rows_to_remove.add(row_i)
        else:
            subsampler[row_i] = max(1 - np.sqrt(SUBSAMPLE / (max(year_mat_csr[row_i, 0], 1) / og_sum)), 0)
            if subsampler[row_i] > 0:
                num_subsampled += 1
    print "Num subsampled: ", num_subsampled 
    print "Number rows to remove", len(rows_to_remove), " of", year_mat_coo.shape[0]

    print "Getting columns to remove..."
    year_mat_csc = year_mat_coo.tocsc()
    cols_to_remove = set([])
    year_mat_csc = year_mat_csc.sum(0)
    cdef col_i
    for col_i in xrange(year_mat_csc.shape[1]):
        if not col_i in CONTEXT_INDICES:
            cols_to_remove.add(col_i)
    print "Number columns to remove:", len(cols_to_remove)

    print "Getting sum and writing count files..."
    cdef int i
    cdef sum = 0, nnz = 0
    for i in xrange(year_mat_coo.data.shape[0]):
        if year_mat_coo.col[i] in cols_to_remove:
            year_mat_coo.data[i] = 0
            continue
        if year_mat_coo.row[i] in rows_to_remove:
            year_mat_coo.data[i] = 0
            continue
        sum += year_mat_coo.data[i]
        nnz += 1
    print "Initial sum of elements:", og_sum, " Sum of remaining elements:", sum, " Number remaining elements", nnz

    print "Making temp array for randomness..."
    cdef np.ndarray[np.int32_t, ndim=2] tmp_array = np.empty((SAMPLE_BUFFER + BUFF_SAMPLES, 2), dtype=np.int32) 
    cdef int num_samples, count
    cdef int num_added = 0
    for i in xrange(year_mat_coo.data.shape[0]):
        count = (year_mat_coo.data[i] / sum) * (BUFF_SAMPLES + SAMPLE_BUFFER)
        num_samples = count - (count * (subsampler[year_mat_coo.col[i]] + subsampler[year_mat_coo.row[i]] - 
            subsampler[year_mat_coo.col[i]] * subsampler[year_mat_coo.row[i]]))
        num_samples = int(max(num_samples, 0))
        for j in xrange(num_added, num_added + num_samples):
            tmp_array[j, 0] = year_mat_coo.row[i] 
            tmp_array[j, 1] = year_mat_coo.col[i] 
        num_added += num_samples
    
    train_fp = open(OUTPUT_DIR + str(year) + "-train.txt", "w")
    row_sums = collections.defaultdict(int)
    col_sums = collections.defaultdict(int)
    cdef rand_i
    print "Writing data..."
    for i in xrange(SAMPLES):
        rand_i = np.random.randint(0, num_added)
        pair = tmp_array[rand_i, :]
        row_sums[pair[0]] += 1
        col_sums[pair[1]] += 1
        train_fp.write(MERGED_INDEX[pair[0]].encode('utf-8') + " " + MERGED_INDEX[pair[1]].encode('utf-8') + "\n")
        
    wv_fp = open(OUTPUT_DIR + str(year) + "-wv.txt", "w")
    for row_i, count in row_sums.iteritems():
        wv_fp.write(MERGED_INDEX[row_i].encode('utf-8') + " " + str(count) + "\n")
    cv_fp = open(OUTPUT_DIR + str(year) + "-cv.txt", "w")
    for col_i, count in col_sums.iteritems():
        cv_fp.write(MERGED_INDEX[col_i].encode('utf-8') + " " + str(count) + "\n")
        
def run_parallel(num_procs):
    lock = Lock()
    procs = [Process(target=main, args=[i, lock]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

