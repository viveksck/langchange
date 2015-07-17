import random
import os
import argparse
import collections
from Queue import Empty
from multiprocessing import Process, Queue
from scipy.sparse import coo_matrix

import ioutils
from cooccurrence import matstore
from cooccurrence.indexing import get_word_indices
from cooccurrence.makesecondorder import make_secondorder_mat
from cooccurrence.netstats import get_year_stats
from cooccurrence.makeknnnet import make_knn_mat

STATS = ["deg", "sum", "bclust", "wclust"]
NAN = float('nan')

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

def main(proc_num, queue, out_pref, in_dir, year_indices, word_infos, knn, thresh):
    random.shuffle(years)
    print proc_num, "Start loop"
    while True:
        try: 
            year = queue.get(block=False)
        except Empty:
            print proc_num, "Finished"
            break
        print proc_num, "Making second orders for year", year
        old_mat = matstore.retrieve_mat_as_coo(in_dir + str(year) + ".bin")
        row_d, col_d, data_d, keep_rows = make_secondorder_mat(old_mat, thresh=thresh, min_cooccurs=0)
        second_mat = coo_matrix((data_d, (row_d, col_d)))
        if knn != None:
            row_d, col_d, data_d = make_knn_mat(second_mat)
            second_mat = coo_matrix((data_d, (row_d, col_d)))
        year_stats = get_year_stats(second_mat, year_indices[year], word_infos[year][0], index_set = set(word_infos[year][1]))
        print proc_num, "Writing stats for year", year
        ioutils.write_pickle(year_stats, out_pref + str(year) + "-tmp.pkl")

def run_parallel(num_procs, out_pref, in_dir, years, word_infos, knn=None, thresh=0):
    queue = Queue()
    random.shuffle(years)
    for year in years:
        queue.add(year)
    word_set = set([])
    word_indices = {}
    for year, year_info in word_infos.iteritems():
        word_set = word_set.union(set(year_info[0]))
        word_indices[year] = year_info[1]
    word_list = list(word_set)
    procs = [Process(target=main, args=[i, queue, out_pref, in_dir, word_infos, knn, thresh]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    merge(out_pref, years, word_list)
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Computes network statistics for second order data.")
    parser.add_argument("dir", help="path to directory with co-occurrence data and index")
    parser.add_argument("word_file", help="path to sorted word file(s).", default=None)
    parser.add_argument("num_procs", type=int, help="number of processes to spawn")
    parser.add_argument("--num-words", type=int, help="Number of words (of decreasing average frequency) to include. Must also specifiy word file and index.", default=-1)
    parser.add_argument("--start-year", type=int, help="start year (inclusive)", default=1900)
    parser.add_argument("--end-year", type=int, help="start year (inclusive)", default=2000)
    parser.add_argument("--thresh", type=float, help="optional threshold", default=None)
    parser.add_argument("--knn", type=float, help="optional number of nearest neighbours", default=None)
    args = parser.parse_args()
    years = range(args.start_year, args.end_year + 1)
    word_pickle = ioutils.load_pickle(args.word_file)
    if not args.start_year in word_pickle:
        word_lists = {}
        for year in years:
            word_lists[year] = word_pickle
    else:
        word_lists = word_pickle
    word_infos = {}
    year_indexes = {}
    for year, word_list in word_lists.iteritems():
        year_index = ioutils.load_pickle(args.dir + "/" + str(year) + "-index.pkl") 
        year_indexes[year] = year_index
        if args.num_words != -1:
            word_list = word_list[:args.num_words]
        word_list, word_indices = get_word_indices(word_list, year_index)
        word_infos[year] = (word_list, word_indices)
    outpref ="/secondnetstats-" + str(args.thresh) + "-" + str(args.knn) + "/" 
    ioutils.mkdir(args.dir + outpref)
    outpref += args.word_file.split("/")[-1].split(".")[0]
    if args.num_words != -1:
        outpref += "-top" + str(args.num_words)
    if args.thresh != None:
        outpref += "-" + str(args.thresh)
    run_parallel(args.num_procs, outpref, args.dir + "/", years, word_infos, knn=args.knn, thresh=args.thresh)
