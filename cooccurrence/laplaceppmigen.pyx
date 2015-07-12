import random
import os
import argparse
import collections
from multiprocessing import Process, Lock
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

def make_ppmi_mat(old_mat, conf_mat, smooth, eff_sample_size):
    smooth = old_mat.sum() * smooth
    print smooth
    prob_norm = old_mat.sum() + (old_mat.shape[0] ** 2) * smooth
#    temp = old_mat / old_mat.sum()
#    old_sum = old_mat.sum()
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
#            else:
#                if data_d[i] < 0:
#                    print "Old: ", temp[row_d[i], col_d[i]], temp[row_d[i], :].sum(), temp[col_d[i], :].sum()
#                    print "New: ", joint_prob, row_probs[row_d[i], 0], row_probs[col_d[i], 0]
#                    print "Smooth: ", smooth, old_mat.shape[0], old_sum
        data_d[i] = max(data_d[i], 0)
        data_d[i] /= -1.0 * np.log(joint_prob)

    return row_d, col_d, data_d

def main(proc_num, lock, out_dir, in_dir, years, smooth, year_word_indices, conf_dir, eff_sample_size):
    cdef int i
    cdef np.ndarray data_d
    cdef np.ndarray row_d, col_d
    cdef float prob_norm

    random.shuffle(years)
    print proc_num, "Start loop"
    while True:
        lock.acquire()
        work_left = False
        for year in years:
            dirs = set(os.listdir(out_dir))
            if str(year) + ".bin" in dirs:
                continue
            work_left = True
            print proc_num, "year", year
            fname = out_dir + str(year) + ".bin"
            with open(fname, "w") as fp:
                fp.write("")
            fp.close()
            break
        lock.release()
        if not work_left:
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
        if year_word_indices != None:
            word_indices = year_word_indices[year][1]
#            tmp_word_indices = list(word_indices[word_indices < min(old_mat.shape[0], old_mat.shape[1])])
#            for i in range(len(tmp_word_indices), len(word_indices)):
#               tmp_word_indices.append(0) 
            old_mat = old_mat[word_indices, :]
            old_mat = old_mat[:, word_indices]
#            for i in range(len(tmp_word_indices), len(word_indices)):
#                old_mat[i, :] = 0
#                old_mat[:,i] = 0
            
        row_d, col_d, data_d = make_ppmi_mat(old_mat, conf_mat, smooth, eff_sample_size)
        print proc_num, "Writing counts for year", year
        matstore.export_mat_eff(row_d, col_d, data_d, year, out_dir)
        year_index = collections.OrderedDict()
        word_list = year_word_indices[year][0]
        for i in xrange(len(word_list)):
            year_index[word_list[i]] = i
        ioutils.write_pickle(year_index, out_dir + str(year) + "-index.pkl")
            

def run_parallel(num_procs, out_dir, in_dir, years, smooth, word_indices, conf_dir, eff_sample_size):
    lock = Lock()
    procs = [Process(target=main, args=[i, lock, out_dir, in_dir, years, smooth, word_indices, conf_dir, eff_sample_size]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
