import random
import os
import argparse
from multiprocessing import Process, Lock
from scipy.sparse import coo_matrix

import ioutils
import numpy as np
from cooccurrence import matstore

START_YEAR = 1900
END_YEAR = 2000
K = 5

def make_knn_mat(old_mat, k):
    old_mat = old_mat.tocsr()
    old_mat.setdiag(0)
    row_keeps = np.empty((old_mat.shape[0],))
    count = 0
    for row_i in xrange(old_mat.shape[0]):
        row_data = (old_mat[row_i, :]).data
        eff_k = min(k, len(row_data))
        row_keeps[row_i] = np.sort(row_data)[-eff_k]
        if (row_data >= row_keeps[row_i]).sum() > 5:
            count += 1
    print "FUckered:", count
    old_mat = old_mat.tocoo()
    for i in xrange(len(old_mat.data)):
        if old_mat.data[i] < row_keeps[old_mat.row[i]]:
            old_mat.data[i] = 0
    test = old_mat.tocsr()
    for row_i in xrange(test.shape[0]):
        if len(test[row_i, :].nonzero()[1]) > 5:
            pass
            #print "WHAT"
    return old_mat.row, old_mat.col, old_mat.data

def main(proc_num, lock, in_dir, years, k):
    random.shuffle(years)
    print proc_num, "Start loop"
    tmp_pref = in_dir + "dknn-" + str(k) + "/"
    while True:
        lock.acquire()
        work_left = False
        for year in years:
            dirs = set(os.listdir(tmp_pref))
            if str(year) + ".bin" in dirs:
                continue
            work_left = True
            print proc_num, "year", year
            fname = tmp_pref + str(year) + ".bin"
            with open(fname, "w") as fp:
                fp.write("")
            fp.close()
            break
        lock.release()
        if not work_left:
            print proc_num, "Finished"
            break

        print proc_num, "Making knn net for year", year
        old_mat = matstore.retrieve_mat_as_coo(in_dir + str(year) + ".bin")
        row_d, col_d, data_d = make_knn_mat(old_mat, k)
        
        print proc_num, "Writing counts for year", year
        matstore.export_mat_eff(row_d, col_d, data_d, year, tmp_pref)

def run_parallel(num_procs, in_dir, years, k):
    lock = Lock()
    procs = [Process(target=main, args=[i, lock, in_dir, years, k]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merges years of raw 5gram data.")
    parser.add_argument("in_dir", help="path to unmerged data")
    parser.add_argument("num_procs", type=int, help="number of processes to spawn")
    parser.add_argument("--start-year", type=int, help="start year (inclusive)", default=START_YEAR)
    parser.add_argument("--end-year", type=int, help="start year (inclusive)", default=END_YEAR)
    parser.add_argument("--k", type=int, help="k nn thresh", default=K)
    args = parser.parse_args()
    years = range(args.start_year, args.end_year + 1)
    ioutils.mkdir(args.in_dir + "/dknn-" + str(args.k))
    run_parallel(args.num_procs, args.in_dir + "/", years, args.k) 
