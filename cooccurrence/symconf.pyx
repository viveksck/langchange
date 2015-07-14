import random
import os
import argparse
from multiprocessing import Process, Lock
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

def make_conf_mat(old_mat, alpha, eff_sample_size, min_val, fwer_control=False):
    print "alpha:", alpha
#    smooth = old_mat.sum() * 10 ** -10.0
    smooth = 0
    print smooth
    row_probs = compute_rowcol_probs(old_mat, smooth)
    print "Row sums:", old_mat.sum()
    old_mat = old_mat.tocoo()

    row_d = old_mat.row
    col_d = old_mat.col
    data_d = old_mat.data
   
    sample_size = eff_sample_size
    print "Eff sample size: ", sample_size
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
    return row_d, col_d, data_d

def main(proc_num, lock, out_dir, in_dir, years, alpha, eff_sample_size, year_word_indices, fwer_control):
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

        old_mat = matstore.retrieve_mat_as_coo(in_dir + str(year) + ".bin", min_size=250000)
        old_mat = old_mat.tocsr()
        if year_word_indices != None:
            word_indices = year_word_indices[year]
            old_mat = old_mat[word_indices, :]
            old_mat = old_mat[:, word_indices]
#            tmp_word_indices = list(word_indices[word_indices < min(old_mat.shape[0], old_mat.shape[1])])
#            for i in range(len(tmp_word_indices), len(word_indices)):
#               tmp_word_indices.append(0) 
#            old_mat = old_mat[tmp_word_indices, :]
#            old_mat = old_mat[:, tmp_word_indices]
#            for i in range(len(tmp_word_indices), len(word_indices)):
#                old_mat[i, :] = 0
#                old_mat[:,i] = 0
       # print "Old sum:", old_mat.sum()
       # old_mat.eliminate_zeros()
       # print "Old num occurr", len(old_mat.data)
        num_occur = np.empty((10,))
        for i in xrange(len(num_occur)):
            rand_sample = np.random.multinomial(eff_sample_size, old_mat.data/old_mat.data.sum())
            num_sampled = (rand_sample > 0).sum() 
            num_occur[i] = num_sampled
        print "Num occur:", year, num_occur.mean(), num_occur.std()
        num_occur = (int(num_occur.mean()))
        min_val = np.sort(old_mat.data)[-num_occur]
        permed_indices = np.random.permutation(len(old_mat.data))
        for i in permed_indices:
            if num_occur < 0:
                break
            if old_mat.data[i] < min_val:
                old_mat.data[i] = 0
            else:
                num_occur -= 1

       # print "New sum:", old_mat.sum()
       # old_mat.eliminate_zeros()
       # old_mat = old_mat.tocoo()
       # print "Number insiginficant:", np.sum(old_mat.data / old_mat.sum() <  10.0 ** -7.0), year
       # if fwer_control:
       #     alpha = alpha / float((old_mat.data > 0).sum())
        print "Bootstrapping year", year
        #old_mat.data = np.random.multinomial(eff_sample_size, old_mat.data/old_mat.data.sum())
        #old_mat.data = old_mat.data.astype(np.float64, copy=False)
        #old_mat = (old_mat + old_mat.T) / 2.0
        print "New sum", old_mat.sum()
        print "Making conf mat for year", year
        row_d, col_d, data_d = make_conf_mat(old_mat, alpha, eff_sample_size, 0)
        
       # print proc_num, "Writing counts for year", year
        matstore.export_mat_eff(row_d, col_d, data_d, year, out_dir)
            

def run_parallel(num_procs, out_dir, in_dir, years, alpha, eff_sample_size, word_indices, fwer_control): 
    np.random.seed(10)
    lock = Lock()
    procs = [Process(target=main, args=[i, lock, out_dir, in_dir, years, alpha, eff_sample_size, word_indices, fwer_control]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
