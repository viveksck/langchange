import random
import os
import argparse
from Queue import Empty
from multiprocessing import Process, Queue
from scipy.sparse import coo_matrix

import ioutils
from cooccurrence import matstore

import numpy as np 
cimport numpy as np

NAN = float('nan')
STATS = ["deg", "sum", "bclust", "wclust"]

def compute_word_stats(mat, word, word_index, index_set = None):
    if not word in word_index:
        return {"deg" : NAN, "sum" : NAN, "bclust" : NAN, "wclust" : NAN}
    word_i = word_index[word] 
    if index_set != None and not word_i in index_set:
        return {"deg" : NAN, "sum" : NAN, "bclust" : NAN, "wclust" : NAN}
    if word_i >= mat.shape[0]: 
        return {"deg" : NAN, "sum" : NAN, "bclust" : NAN, "wclust" : NAN}
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

def get_year_stats(mat, year_index, word_list, index_set = None):
    mat.setdiag(0)
    mat = mat.tocsr()
    year_stats = {stat:{} for stat in STATS}
    for word in word_list:
        for word in word_list:
            single_word_stats = compute_word_stats(mat, word, year_index, index_set = index_set)
            for stat in single_word_stats:
                year_stats[stat][word] = single_word_stats[stat]
    return year_stats

def merge(out_pref, years, full_word_list):
    merged_word_stats = {}
    for stat in STATS:
        merged_word_stats[stat] = {}
        for word in full_word_list:
            merged_word_stats[stat][word] = {}
    for year in years:
        year_stats = ioutils.load_pickle(out_pref + str(year) + ".pkl")
        for stat, stat_vals in year_stats.iteritems():
            for word in full_word_list:
                if not word in stat_vals:
                    merged_word_stats[stat][word][year] = NAN
                else:
                    merged_word_stats[stat][word][year] = stat_vals[word]
        os.remove(out_pref + str(year) + "-tmp.pkl")
    ioutils.write_pickle(merged_word_stats, out_pref +  ".pkl")

def main(proc_num, queue, out_pref, in_dir, year_indexes, word_infos, thresh):
    print proc_num, "Start loop"
    while True:
        try: 
            year = queue.get(block=False)
        except Empty:
            print proc_num, "Finished"
            break

        print proc_num, "Retrieving mat for year", year
        if thresh != None:
            mat = matstore.retrieve_mat_as_coo_thresh(in_dir + str(year) + ".bin", thresh)
        else:
            mat = matstore.retrieve_mat_as_coo(in_dir + str(year) + ".bin")
        print proc_num, "Getting stats for year", year
        year_stats = get_year_stats(mat, year_indexes[year], index_set = set(word_infos[year][1]))

        print proc_num, "Writing stats for year", year
        ioutils.write_pickle(year_stats, out_pref + str(year) + "-tmp.pkl")
        queue.task_done()

def run_parallel(num_procs, out_pref, in_dir, years, year_indexes, word_infos, thresh):
    word_set = set([])
    word_indices = {}
    for year, year_info in word_infos.iteritems():
        word_set = word_set.union(set(year_info[0]))
        word_indices[year] = year_info[1]
    word_list = list(word_set)
    queue = Queue()
    random.shuffle(years)
    for year in years:
        queue.put(year)
    procs = [Process(target=main, args=[i, queue, out_pref, in_dir, year_indexes, word_indices, word_list, thresh]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print "Merging"
    merge(out_pref, years, word_list)
