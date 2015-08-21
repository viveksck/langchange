import random
import os
import argparse
from Queue import Empty
from multiprocessing import Process, Queue
from scipy.sparse import coo_matrix
import scipy.stats as stat

import ioutils
from cooccurrence import matstore
from cooccurrence.wilsonconf import check_conf
from cooccurrence.bootstrapping import bootstrap_count_mat

import numpy as np
cimport numpy as np

DYTPE = np.float64
ctypedef np.float64_t DTYPE_t

def compute_rowcol_probs(csr_mat, smooth):
    cdef np.ndarray row_probs
    row_probs = csr_mat.sum(1)
    row_probs += row_probs.shape[0] * smooth
    row_probs /= row_probs.sum()
    return row_probs

def make_conf_mat(old_mat, alpha, eff_sample_size, min_val, fwer_control=False, smooth = 0):
    row_probs = compute_rowcol_probs(old_mat, smooth)
    old_mat = old_mat.tocoo()

    row_d = old_mat.row
    col_d = old_mat.col
    data_d = old_mat.data
    sample_size = eff_sample_size
    prob_norm = old_mat.sum() + smooth * old_mat.shape[0] ** 2.0
    if fwer_control:
        alpha = alpha / old_mat.nnz
    z = stat.norm.ppf(1 - alpha / 2.0)
    z2 = z ** 2.0
    for i in xrange(len(old_mat.data)):
        val = (old_mat.data[i] + smooth) / prob_norm
        if val < min_val:
            data_d[i] = 0.0
            continue
        if check_conf(val, row_probs[row_d[i], 0], row_probs[col_d[i], 0], z, z2, sample_size):
            data_d[i] = 1.0
        else:
            data_d[i] = 0.0
    return coo_matrix((data_d, (row_d, col_d)))

def main(proc_num, queue, out_dir, in_dir, alpha, eff_sample_size, year_index_infos, fwer_control=False):
    cdef int i
    cdef np.ndarray data_d
    cdef np.ndarray row_d, col_d
    cdef float prob_norm
    while True:
        try: 
            year = queue.get(block=False)
        except Empty:
            print proc_num, "Finished"
            break

        old_mat = matstore.retrieve_mat_as_coo(in_dir + str(year) + ".bin", min_size=250000)
        old_mat = old_mat.tocsr()
        if year_index_infos != None:
            word_indices = year_index_infos[year]["indices"]
            old_mat = old_mat[word_indices, :]
            old_mat = old_mat[:, word_indices]
        print "Making conf mat for year", year
        if fwer_control:
            alpha = alpha / float(len(old_mat.nonzero()[1]))
        conf_mat = make_conf_mat(old_mat, alpha, eff_sample_size, 0)
        matstore.export_mat_eff(conf_mat.row_d, conf_mat.col_d, conf_mat.data_d, year, out_dir)
            

def run_parallel(num_procs, out_dir, in_dir, alpha, eff_sample_size, year_index_infos, fwer_control): 
    years = year_index_infos.keys()
    queue = Queue()
    for year in years:
        queue.put(year)
    procs = [Process(target=main, args=[i, queue, out_dir, in_dir, alpha, eff_sample_size, year_index_infos, fwer_control]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
