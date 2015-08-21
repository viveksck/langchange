import random
import os
import argparse
import sys
import collections

from Queue import Empty
from multiprocessing import Process, Queue
from scipy.sparse import coo_matrix

import ioutils
from cooccurrence import matstore
from cooccurrence.laplaceppmigen import make_ppmi_mat
from cooccurrence.symconf import make_conf_mat
from cooccurrence.netstats import compute_word_stats
from cooccurrence.indexing import get_full_word_list

import numpy as np 
cimport numpy as np

NAN = float('nan')
STATS = ["deg", "sum", "bclust", "wclust"]

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

def reduce_mat(old_mat, year_index_info):
    word_indices = year_index_info["indices"]
    reduced_mat = old_mat[word_indices, :]
    reduced_mat = old_mat[:, word_indices]
    year_index = collections.OrderedDict()
    # need to make new index since matrix changed
    word_list = year_index_info["list"]
    for i in xrange(len(word_list)):
        year_index[word_list[i]] = i
    reduced_mat, year_index

def bootstrap_mat(mat, eff_sample_size):
    boot_mat = mat.copy()
    boot_mat.data = np.random.multinomial(eff_sample_size, mat.data/mat.data.sum())
    boot_mat.data = boot_mat.data.astype(np.float64, copy=False)
    boot_mat = (boot_mat + boot_mat.T) / 2.0

def main(proc_num, queue, out_pref, in_dir, year_index_infos, num_boots, smooth, eff_sample_size, alpha, fwer_control, id):
    print proc_num, "Start loop"
    while True:
        try: 
            year = queue.get(block=False)
        except Empty:
            print proc_num, "Finished"
            break

        print proc_num, "Retrieving mat for year", year
        old_mat = matstore.retrieve_mat_as_coo(in_dir + str(year) + ".bin", min_size=250000)
        old_mat = old_mat.tocsr()
        reduced_mat, year_index = reduce_mat(old_mat, year_index_infos[year])
        word_stat_vecs = collections.defaultdict(dict)
        for stat in STATS:
            for word in year_index_infos["list"]:
                word_stat_vecs[stat][word] = np.zeros((num_boots,))

        for boot_iter in range(num_boots):
            print proc_num, "Bootstrapping mat for year", year, "boot_iter", boot_iter
            boot_mat = bootstrap_mat(reduced_mat, eff_sample_size)
            boot_mat = make_ppmi_mat(boot_mat, conf_mat, smooth, eff_sample_size)
            boot_mat = boot_mat.tocsr()

            if alpha != None:
                conf_mat = make_conf_mat(boot_mat, alpha, eff_sample_size, 0, fwer_control=fwer_control) 
                conf_mat = conf_mat.tocsr()
            else:
                conf_mat = None

            print proc_num, "Getting bootstrap stats for year", year, "boot_iter", boot_iter
            for word in year_index_infos[year]["list"]:
                single_word_stats = compute_word_stats(boot_mat, word, year_index)
                for stat in single_word_stats:
                    word_stat_vecs[stat][word][boot_iter] = single_word_stats[stat]
        
        print proc_num, "Writing stats for year", year
        ioutils.write_pickle(word_stat_vecs, out_pref + str(year) + "-tmp" + str(id) + ".pkl")

def run_parallel(num_procs, out_pref, in_dir, year_index_infos, num_boots, smooth, eff_sample_size, alpha, fwer_control, id):
    queue = Queue()
    for year in year_index_infos.keys():
        queue.put(year)
    procs = [Process(target=main, args=[i, queue, out_pref, in_dir, year_index_infos, num_boots, smooth, eff_sample_size, alpha, fwer_control, id]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print "Merging"
    merge(out_pref, year_index_infos.keys(), get_full_word_list(year_index_infos), id)
