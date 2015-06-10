import random
import os
import argparse
from multiprocessing import Process, Lock

import ioutils
from cooccurrence import matstore

import numpy as np
cimport numpy as np

DYTPE = np.float64
ctypedef np.float64_t DTYPE_t

SMOOTH = -8.0
START_YEAR = 1900
END_YEAR = 2000

def compute_rowcol_probs(csr_mat, smooth):
    cdef np.ndarray row_probs
    row_probs = csr_mat.sum(1)
    row_probs = row_probs + row_probs.shape[0] * smooth
    row_probs /= row_probs.sum()
    return row_probs

def make_ppmi_mat(old_mat, smooth):
    old_mat = (old_mat + old_mat.T)/2.0
    smooth = old_mat.sum() * smooth
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

    return row_d, col_d, data_d

def main(proc_num, lock, out_dir, in_dir, years, smooth):
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
        old_mat = matstore.retrieve_mat_as_coo(in_dir + str(year) + ".bin")
        old_mat = old_mat.tocsr()
        row_d, col_d, data_d = make_ppmi_mat(old_mat, smooth)
        
        print proc_num, "Writing counts for year", year
        matstore.export_mat_eff(row_d, col_d, data_d, year, out_dir)
            

def run_parallel(num_procs, out_dir, in_dir, years, smooth):
    lock = Lock()
    procs = [Process(target=main, args=[i, lock, out_dir, in_dir, years, smooth]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merges years of raw 5gram data.")
    parser.add_argument("out_dir", help="directory where data will be stored")
    parser.add_argument("in_dir", help="path to unmerged data")
    parser.add_argument("num_procs", type=int, help="number of processes to spawn")
    parser.add_argument("--start-year", type=int, help="start year (inclusive)", default=START_YEAR)
    parser.add_argument("--end-year", type=int, help="start year (inclusive)", default=END_YEAR)
    parser.add_argument("--smooth", type=int, help="smoothing factor", default=SMOOTH)
    args = parser.parse_args()
    years = range(args.start_year, args.end_year + 1)
    smooth = 10.0**(float(args.smooth))
    run_parallel(args.num_procs, args.out_dir, args.in_dir, years, smooth)       


