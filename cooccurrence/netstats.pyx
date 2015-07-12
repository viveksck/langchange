import random
import os
import argparse
from multiprocessing import Process, Lock
from scipy.sparse import coo_matrix

import ioutils
from cooccurrence import matstore

import numpy as np 
cimport numpy as np

NAN = float('nan')

def compute_word_stats(mat, word, index_set, word_index):
    if not word in word_index:
        return NAN, NAN, NAN, NAN 
    word_i = word_index[word] 
    if not word_i in index_set or word_i >= mat.shape[0]: 
        return NAN, NAN, NAN, NAN
    vec = mat[word_i, :]
    indices = vec.nonzero()[1]
    indices = np.intersect1d(indices, index_set, assume_unique=True)
    vec = vec[:, indices]
    if len(indices) <= 1:
        return 0, 0, len(indices), vec.sum()
    weights = vec/vec.sum()
    reduced = mat[indices, :]
    reduced = reduced[:, indices]
    reduced.eliminate_zeros()
    weighted = (weights * reduced).sum() / (float(len(indices)) - 1)
    binary = float(reduced.nnz) / (len(indices) * (len(indices) - 1)) 
    deg = len(indices)
    sum = vec.sum()
    return (weighted, binary, deg, sum)

def merge(out_pref, tmp_out_pref, years, word_list):
    binary_yearstats = {}
    weighted_yearstats = {}
    deg_yearstats = {}
    sum_yearstats = {}
    for word in word_list:
        binary_yearstats[word] = {}
        weighted_yearstats[word] = {}
        deg_yearstats[word] = {}
        sum_yearstats[word] = {}
    for year in years:
        binary_yearstat = ioutils.load_pickle(tmp_out_pref + str(year) + "-binary.pkl")
        weighted_yearstat = ioutils.load_pickle(tmp_out_pref + str(year) + "-weighted.pkl")
        deg_yearstat = ioutils.load_pickle(tmp_out_pref + str(year) + "-deg.pkl")
        sum_yearstat = ioutils.load_pickle(tmp_out_pref + str(year) + "-sum.pkl")
        for word in word_list:
            binary_yearstats[word][year] = binary_yearstat[word]
            weighted_yearstats[word][year] = weighted_yearstat[word]
            deg_yearstats[word][year] = deg_yearstat[word]
            sum_yearstats[word][year] = sum_yearstat[word]
        os.remove(tmp_out_pref + str(year) + "-binary.pkl")
        os.remove(tmp_out_pref + str(year) + "-weighted.pkl")
        os.remove(tmp_out_pref + str(year) + "-deg.pkl")
        os.remove(tmp_out_pref + str(year) + "-sum.pkl")
    ioutils.write_pickle(binary_yearstats, out_pref + "-binary.pkl")
    ioutils.write_pickle(weighted_yearstats, out_pref + "-weighted.pkl")
    ioutils.write_pickle(deg_yearstats, out_pref + "-deg.pkl")
    ioutils.write_pickle(sum_yearstats, out_pref + "-sum.pkl")

def main(proc_num, lock, out_pref, tmp_out_pref, in_dir, years, year_indexes, word_indices, word_list, thresh):
    random.shuffle(years)
    print proc_num, "Start loop"
    while True:
        lock.acquire()
        work_left = False
        for year in years:
            existing_files = set(os.listdir(in_dir + "/netstats"))
            fname = tmp_out_pref.split("/")[-1] + str(year) + "-binary.pkl"
            if fname in existing_files:
                continue
            work_left = True
            print proc_num, "year", year
            with open(in_dir + "/netstats/"+ fname, "w") as fp:
                fp.write("")
            fp.close()
            break
        lock.release()
        if not work_left:
            print proc_num, "Finished"
            break

        print proc_num, "Retrieving mat for year", year
        if thresh != None:
            mat = matstore.retrieve_mat_as_coo_thresh(in_dir + str(year) + ".bin", thresh)
        else:
            mat = matstore.retrieve_mat_as_coo(in_dir + str(year) + ".bin")
        print mat.shape
        mat.setdiag(0)
        indices = word_indices[year]
        mat = mat.tocsr()
        weighted_word_stats = {}
        binary_word_stats = {}
        deg_word_stats = {}
        sum_word_stats = {}
        print proc_num, "Getting stats for year", year
        for word in word_list:
            weighted, binary, deg, sum = compute_word_stats(mat, word, indices, year_indexes[year])
            weighted_word_stats[word] = weighted
            binary_word_stats[word] = binary
            deg_word_stats[word] = deg
            sum_word_stats[word] = sum

        print proc_num, "Writing stats for year", year
        ioutils.write_pickle(weighted_word_stats, tmp_out_pref + str(year) + "-weighted.pkl")
        ioutils.write_pickle(binary_word_stats, tmp_out_pref + str(year) + "-binary.pkl")
        ioutils.write_pickle(deg_word_stats, tmp_out_pref + str(year) + "-deg.pkl")
        ioutils.write_pickle(sum_word_stats, tmp_out_pref + str(year) + "-sum.pkl")

def run_parallel(num_procs, out_pref, in_dir, years, year_indexes, word_infos, thresh):
    word_set = set([])
    word_indices = {}
    for year, year_info in word_infos.iteritems():
        word_set = word_set.union(set(year_info[0]))
        word_indices[year] = year_info[1]
    word_list = list(word_set)
    lock = Lock()
    tmp_out_pref = out_pref + "tmp-"
    procs = [Process(target=main, args=[i, lock, out_pref, tmp_out_pref, in_dir, years, year_indexes, word_indices, word_list, thresh]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print "Merging"
    merge(out_pref, tmp_out_pref, years, word_list)
