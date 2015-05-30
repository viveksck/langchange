import random
import os
from multiprocessing import Process, Lock

from googlengram import matstore, util

import numpy as np
cimport numpy as np

DATA_DIR = '/dfs/scratch0/google_ngrams/'
INPUT_DIR = DATA_DIR + '/5grams_merged/'
OUTPUT_DIR = DATA_DIR + '/5grams_ppmi_lsmooth_fixed/'

DYTPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef float SMOOTH = 10.0**(-8.0)

def compute_rowcol_probs(csr_mat, smooth):
    cdef np.ndarray row_probs
    row_probs = csr_mat.sum(1)
    row_probs = row_probs + row_probs.shape[0] * smooth
    row_probs /= row_probs.sum()
    return row_probs

def main(proc_num, lock):
    cdef int i
    cdef np.ndarray data_d
    cdef np.ndarray row_d, col_d
    cdef float prob_norm

    years = range(1900, 2001)
    random.shuffle(years)
    print proc_num, "Start loop"
    while True:
        lock.acquire()
        work_left = False
        for year in years:
            dirs = set(os.listdir(OUTPUT_DIR))
            if str(year) + ".bin" in dirs:
                continue
            work_left = True
            print proc_num, "year", year
            fname = OUTPUT_DIR + str(year) + ".bin"
            with open(fname, "w") as fp:
                fp.write("")
            fp.close()
            break
        lock.release()
        if not work_left:
            print proc_num, "Finished"
            break

        print proc_num, "Making PPMIs for year", year
        old_mat = matstore.retrieve_cooccurrence_as_coo(INPUT_DIR + str(year) + ".bin")
        old_mat = old_mat.tocsr()
        old_mat = (old_mat + old_mat.T)/2.0
        smooth = old_mat.sum() * SMOOTH
        print smooth, old_mat.sum(), SMOOTH
        row_probs = compute_rowcol_probs(old_mat, smooth)
        old_mat = old_mat.tocoo()

        row_d = old_mat.row
        col_d = old_mat.col
        data_d = old_mat.data
        
        prob_norm = old_mat.sum() + (old_mat.shape[0] ** 2) * smooth
        for i in xrange(len(old_mat.data)):
            joint_prob = (data_d[i] + smooth) / prob_norm
            data_d[i] = np.log(joint_prob / (row_probs[row_d[i], 0] * row_probs[col_d[i], 0]))
            data_d[i] = max(data_d[i], 0)
            data_d[i] /= -1.0 * np.log(joint_prob)

        print proc_num, "Writing counts for year", year
        matstore.export_cooccurrence_eff(row_d, col_d, data_d, year, OUTPUT_DIR)
            

def run_parallel(num_procs):
    lock = Lock()
    procs = [Process(target=main, args=[i, lock]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
