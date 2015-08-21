import random
import os
import argparse
from Queue import Empty
from multiprocessing import Process, Queue
from scipy.sparse import coo_matrix

import ioutils
from cooccurrence.makesecondorder import make_secondorder_mat
from cooccurrence.netstats import get_year_stats
from cooccurrence.makeknnnet import make_knn_mat
from cooccurrence.indexing import get_full_word_list
from vecanalysis.representations.representation_factory import simple_create_representation

REP_TYPE = "PPMI"

#only care about degree and sum
STATS = ["deg", "sum"]
NAN = float('nan')

def merge(out_pref, years, full_word_list):
    merged_word_stats = {}
    for stat in STATS:
        merged_word_stats[stat] = {}
        for word in full_word_list:
            merged_word_stats[stat][word] = {}
    for year in years:
        year_stats = ioutils.load_pickle(out_pref + str(year) + "-tmp.pkl")
        for stat, stat_vals in year_stats.iteritems():
            for word in full_word_list:
                if not word in stat_vals:
                    merged_word_stats[stat][word][year] = NAN
                else:
                    merged_word_stats[stat][word][year] = stat_vals[word]
        os.remove(out_pref + str(year) + "-tmp.pkl")
    for stat, stat_val in merged_word_stats.iteritems():
        ioutils.write_pickle(stat_val, out_pref + "-" + stat + ".pkl")

def worker(proc_num, queue, out_pref, in_dir, year_index_infos, knn, thresh):
    random.shuffle(years)
    print proc_num, "Start loop"
    while True:
        try: 
            year = queue.get(block=False)
        except Empty:
            print proc_num, "Finished"
            break
        print proc_num, "Making second orders for year", year
        old_embed = simple_create_representation(REP_TYPE, in_dir + str(year) + ".bin", thresh=thresh)
        old_embed = old_embed.get_subembed(year_index_infos[year]["list"])
        old_mat = old_embed.m.tocoo()
        row_d, col_d, data_d, keep_rows = make_secondorder_mat(old_mat, thresh=thresh, min_cooccurs=0, shrink_mat=False)
        second_mat = coo_matrix((data_d, (row_d, col_d)))
        if knn != None:
            row_d, col_d, data_d = make_knn_mat(second_mat, knn)
            second_mat = coo_matrix((data_d, (row_d, col_d)))
        year_stats = get_year_stats(second_mat, old_embed.wi, old_embed.iw, stats=STATS)
        print proc_num, "Writing stats for year", year
        ioutils.write_pickle(year_stats,  out_pref + str(year) + "-tmp.pkl")

def run_parallel(num_procs, out_pref, in_dir, year_index_infos, knn=None, thresh=0):
    queue = Queue()
    years = year_index_infos.keys()
    random.shuffle(years)
    for year in years:
        queue.put(year)
    procs = [Process(target=worker, args=[i, queue, out_pref, in_dir, year_index_infos, knn, thresh]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    merge(out_pref, years, get_full_word_list(year_index_infos))
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Computes network statistics for second order data.")
    parser.add_argument("dir", help="path to directory with co-occurrence data and index")
    parser.add_argument("word_file", help="path to sorted word file(s).", default=None)
    parser.add_argument("num_procs", type=int, help="number of processes to spawn")
    parser.add_argument("--num-words", type=int, help="Number of words (of decreasing average frequency) to include. Must also specifiy word file and index.", default=-1)
    parser.add_argument("--start-year", type=int, help="start year (inclusive)", default=1900)
    parser.add_argument("--end-year", type=int, help="start year (inclusive)", default=2000)
    parser.add_argument("--thresh", type=float, help="optional threshold", default=0)
    parser.add_argument("--knn", type=int, help="optional number of nearest neighbours", default=None)
    args = parser.parse_args()
    years = range(args.start_year, args.end_year + 1)
    year_index_infos = ioutils.load_year_index_infos(args.dir, years, args.word_file, num_words=args.num_words)
    outpref = args.dir + "/secondnetstats-" + str(args.thresh) + "-" + str(args.knn) + "/" 
    ioutils.mkdir(outpref)
    outpref += args.word_file.split("/")[-1].split(".")[0]
    if args.num_words != -1:
        outpref += "-top" + str(args.num_words)
    run_parallel(args.num_procs, outpref, args.dir + "/", year_index_infos, knn=args.knn, thresh=args.thresh)
