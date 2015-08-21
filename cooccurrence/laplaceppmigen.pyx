import random
import os
import argparse
import collections
from Queue import Empty 
from multiprocessing import Process, Queue
from scipy.sparse import coo_matrix

import ioutils
from cooccurrence import matstore

import numpy as np
cimport numpy as np

DYTPE = np.float64
ctypedef np.float64_t DTYPE_t

def compute_rowcol_probs(csr_mat, prob_norm, smooth):
    cdef np.ndarray row_probs
    row_probs = csr_mat.sum(1)
    row_probs = row_probs + csr_mat.shape[0] * smooth
    row_probs /= prob_norm
    return row_probs

def make_ppmi_mat(old_mat, conf_mat, smooth):
    smooth = old_mat.sum() * smooth
    prob_norm = old_mat.sum() + (old_mat.shape[0] ** 2) * smooth
    row_probs = compute_rowcol_probs(old_mat, prob_norm, smooth)
    old_mat = old_mat.tocoo()
    row_d = old_mat.row
    col_d = old_mat.col
    data_d = old_mat.data
    for i in xrange(len(old_mat.data)):
        joint_prob = (data_d[i] + smooth) / prob_norm
        data_d[i] = np.log(joint_prob / (row_probs[row_d[i], 0] * row_probs[col_d[i], 0]))
        if conf_mat != None:
            if conf_mat[row_d[i], col_d[i]] <= 0:
                data_d[i] = 0
        data_d[i] = max(data_d[i], 0)
        data_d[i] /= -1.0 * np.log(joint_prob)

    return coo_matrix(data_d, (row_d, col_d))

def worker(proc_num, queue, out_dir, in_dir, smooth, year_index_infos, conf_dir):
    cdef int i
    cdef np.ndarray data_d
    cdef np.ndarray row_d, col_d
    cdef float prob_norm
    print proc_num, "Start loop"
    while True:
        try: 
            year = queue.get(block=False)
        except Empty:
            print proc_num, "Finished"
            break

        print proc_num, "Making PPMIs for year", year
        old_mat = matstore.retrieve_mat_as_coo(in_dir + str(year) + ".bin", min_size=250000)
        old_mat = old_mat.tocsr()
        if conf_dir != None:
            conf_mat = matstore.retrieve_mat_as_coo(conf_dir + str(year) + ".bin", min_size=250000)
            conf_mat = conf_mat.tocsr()
        else:
            conf_mat = None
        if year_index_infos != None:
            word_indices = year_index_infos[year]["indices"]
            old_mat = old_mat[word_indices, :]
            old_mat = old_mat[:, word_indices]
            
        ppmi_mat  = make_ppmi_mat(old_mat, conf_mat, smooth)
        print proc_num, "Writing counts for year", year
        matstore.export_mat_eff(ppmi_mat.row, ppmi_mat.col, ppmi_mat.data, year, out_dir)
        year_index = collections.OrderedDict()
        word_list = year_index_infos[year]["list"]
        for i in xrange(len(word_list)):
            year_index[word_list[i]] = i
        ioutils.write_pickle(year_index, out_dir + str(year) + "-index.pkl")

def run_parallel(num_procs, out_dir, in_dir, smooth, year_index_infos, conf_dir):
    queue = Queue()
    years = year_index_infos.keys()
    for year in years:
        queue.put(year)
    procs = [Process(target=worker, args=[i, queue, out_dir, in_dir, smooth, year_index_infos, conf_dir]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
