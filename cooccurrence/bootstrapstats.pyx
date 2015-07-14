import random
import os
import argparse
import collections 
import sys

from multiprocessing import Process, Lock
from scipy.sparse import coo_matrix

import ioutils
from cooccurrence import matstore
from cooccurrence.laplaceppmigen import make_ppmi_mat
from cooccurrence.symconf import make_conf_mat

import numpy as np 
cimport numpy as np

NAN = float('nan')
STATS = ["deg", "sum", "bclust", "wclust"]

def compute_word_stats(mat, word, word_index):
    word_i = word_index[word] 
    vec = mat[word_i, :]
    indices = vec.nonzero()[1]
    vec = vec[:, indices]
    if len(indices) <= 1:
        return {"deg" : len(indices), "sum" : vec.sum(), "bclust" : 0, "wclust" : 0}
    weights = vec/vec.sum()
    reduced = mat[indices, :]
    reduced = reduced[:, indices]
    reduced.eliminate_zeros()
    weighted = (weights * reduced).sum() / (float(len(indices)) - 1)
    binary = float(reduced.nnz) / (len(indices) * (len(indices) - 1)) 
    deg = len(indices)
    sum = vec.sum()
    return  {"deg" : deg, "sum" : sum, "bclust" : binary, "wclust" : weighted}

def merge(out_pref, years, full_word_list, id):
    merged_word_stats = {}
    for stat in STATS:
        merged_word_stats[stat] = {}
        for word in full_word_list:
            merged_word_stats[stat][word] = {}
    for year in years:
        year_stats = ioutils.load_pickle(out_pref + str(year) + "-tmp" + str(id) + ".pkl")
        for stat, stat_vals in year_stats.iteritems():
            for word in full_word_list:
                if not word in stat_vals:
                    merged_word_stats[stat][word][year] = NAN
                else:
                    merged_word_stats[stat][word][year] = stat_vals[word]
        os.remove(out_pref + str(year) + "-tmp" + str(id) + ".pkl")
    ioutils.write_pickle(merged_word_stats, out_pref + "-" + str(id) + ".pkl")

def main(proc_num, lock, out_pref, in_dir, word_infos, num_boots, smooth, eff_sample_size, alpha, id):
    years = word_infos.keys()
    random.shuffle(years)
    print proc_num, "Start loop"
    while True:
        lock.acquire()
        work_left = False
        for year in years:
            existing_files = set(os.listdir(in_dir + "/bootstats"))
            fname = out_pref.split("/")[-1] + str(year) + "-tmp" + str(id) + ".pkl"
            if fname in existing_files:
                continue
            work_left = True
            print proc_num, "year", year
            with open(in_dir + "/bootstats/"+ fname, "w") as fp:
                fp.write("")
            fp.close()
            break
        lock.release()
        if not work_left:
            print proc_num, "Finished"
            break

        print proc_num, "Retrieving mat for year", year
        old_mat = matstore.retrieve_mat_as_coo(in_dir + str(year) + ".bin", min_size=250000)
        old_mat = old_mat.tocsr()
        word_indices = word_infos[year][1]
        old_mat = old_mat[word_indices, :]
        old_mat = old_mat[:, word_indices]
        year_index = collections.OrderedDict()
        word_list = word_infos[year][0]
        for i in xrange(len(word_list)):
            year_index[word_list[i]] = i

        word_stat_vecs = collections.defaultdict(dict)
        for stat in STATS:
            for word in word_list:
                word_stat_vecs[stat][word] = np.zeros((num_boots,))

        for boot_iter in range(num_boots):
            print proc_num, "Bootstrapping mat for year", year, "boot_iter", boot_iter
            boot_mat = old_mat.copy()
            boot_mat.data = np.random.multinomial(eff_sample_size, old_mat.data/old_mat.data.sum())
            boot_mat.data = boot_mat.data.astype(np.float64, copy=False)
            boot_mat = (boot_mat + boot_mat.T) / 2.0
            if alpha != None:
                row_d, col_d, data_d = make_conf_mat(boot_mat, alpha, eff_sample_size, 0) 
                conf_mat = coo_matrix((data_d, (row_d, col_d)))
                conf_mat = conf_mat.tocsr()
            else:
                conf_mat = None
            row_d, col_d, data_d = make_ppmi_mat(boot_mat, conf_mat, smooth, eff_sample_size)
            boot_mat = coo_matrix((data_d, (row_d, col_d)))
            boot_mat = boot_mat.tocsr()

            print proc_num, "Getting bootstrap stats for year", year, "boot_iter", boot_iter
            for word in word_list:
                single_word_stats = compute_word_stats(boot_mat, word, year_index)
                for stat in single_word_stats:
                    word_stat_vecs[stat][word][boot_iter] = single_word_stats[stat]
        
        print proc_num, "Writing stats for year", year
        ioutils.write_pickle(word_stat_vecs, out_pref + str(year) + "-tmp" + str(id) + ".pkl")

def run_parallel(num_procs, out_pref, in_dir, year_indexes, num_boots, smooth, eff_sample_size, alpha, id):
    word_set = set([])
    word_indices = {}
    for year, year_info in year_indexes.iteritems():
        word_set = word_set.union(set(year_info[0]))
        word_indices[year] = year_info[1]
    word_list = list(word_set)
    lock = Lock()
    procs = [Process(target=main, args=[i, lock, out_pref, in_dir, year_indexes, num_boots, smooth, eff_sample_size, alpha, id]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print "Merging"
    merge(out_pref, year_indexes.keys(), word_list, id)
