# get snap
import sys
sys.path.append("../snap-python/swig/")
import snap
import random
import os
import collections
import argparse
import numpy as np

from multiprocessing import Process, Lock

import ioutils
from cooccurrence import matstore
from cooccurrence.indexing import get_word_indices
from cooccurrence.snaputils import make_snap_graph

START_YEAR = 1900
END_YEAR = 2000

# SNAP CONSTANTS
REWIRE_EDGE_SWITCHES = 1
BFS_SAMPLES = 1

def compute_graph_stats(graph):
    stats = {}
    # get degree hist
    DegToCntV = snap.TIntPrV()
    snap.GetDegCnt(graph, DegToCntV)
    stats['deg_hist'] = deg_hist = {item.GetVal1():item.GetVal2() for item in DegToCntV}  
    # get avg degree
    avg_deg = 0.0
    num_nodes = 0.0
    for deg, count in deg_hist.items():
        avg_deg += deg * count
        num_nodes += count
    stats['avg_deg'] = avg_deg / num_nodes
    stats['num_nodes'] = num_nodes
    # get clustering
    CfVec = snap.TFltPrV()
    stats['avg_clust'] = snap.GetClustCf(graph, CfVec, -1)
    stats['clust_hist'] = {item.GetVal1():item.GetVal2() for item in CfVec}
    # get effective diameter
    stats['eff_diam'] = snap.GetBfsEffDiam(graph, BFS_SAMPLES)
    # get percentage of nodes in largest component
    stats['max_cc_size'] = snap.GetMxWccSz(graph)
    return stats

def merge(out_pref, tmp_dir, years):
    net_stats = collections.defaultdict(dict)
    rewire_net_stats = collections.defaultdict(dict)
    for year in years:
        year_stats = ioutils.load_pickle(tmp_dir + str(year) + "-tmp.pkl")
        rewire_year_stats = ioutils.load_pickle(tmp_dir + "rewire" + str(year) + "-tmp.pkl")
        for stat, val in year_stats.iteritems():
            net_stats[stat][year] = val
        for stat, val in rewire_year_stats.iteritems():
            rewire_net_stats[stat][year] = val
        os.remove(tmp_dir + str(year) + "-tmp.pkl")
        os.remove(tmp_dir + "rewire" + str(year) + "-tmp.pkl")
    for stat, year_vals in net_stats.iteritems():
        ioutils.write_pickle(year_vals, out_pref + "-" + stat + ".pkl")
    for stat, year_vals in rewire_net_stats.iteritems():
        ioutils.write_pickle(year_vals, out_pref + "-rw-" + stat + ".pkl")

def main(proc_num, lock, out_pref, tmp_dir, in_dir, years, word_infos, thresh):
    random.shuffle(years)
    print proc_num, "Start loop"
    while True:
        lock.acquire()
        work_left = False
        for year in years:
            existing_files = set(os.listdir(tmp_dir))
            fname = str(year) + "-tmp.pkl"
            if fname in existing_files:
                continue
            work_left = True
            print proc_num, "year", year
            with open(tmp_dir + fname, "w") as fp:
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

        mat.setdiag(0)
        if word_infos != None:
            word_indices = word_infos[year][1]
            indices = word_indices[word_indices < min(mat.shape[1], mat.shape[0])]
        else:
            indices = np.arange(mat.shape[0])
        year_graph = make_snap_graph(indices, mat) 
        print proc_num, "Getting statistics for year", year
        year_stats = compute_graph_stats(year_graph)
        rewire_year_stats = compute_graph_stats(snap.GenRewire(year_graph, REWIRE_EDGE_SWITCHES)) 
        ioutils.write_pickle(year_stats, tmp_dir + fname) 
        ioutils.write_pickle(rewire_year_stats, tmp_dir + "rewire" + fname) 

def run_parallel(num_procs, out_pref, tmp_dir, in_dir, years, word_info, thresh):
    lock = Lock()
    procs = [Process(target=main, args=[i, lock, out_pref, tmp_dir, in_dir, years, word_info, thresh]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print "Merging"
    merge(out_pref, tmp_dir, years)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merges years of raw 5gram data.")
    parser.add_argument("dir", help="path to network data (also where output goes)")
    parser.add_argument("num_procs", type=int, help="number of processes to spawn")
    parser.add_argument("--index_dir", help="path to directory with index files", default=None)
    parser.add_argument("--word-file", help="path to sorted word file(s). Must also specify index.", default=None)
    parser.add_argument("--num-words", type=int, help="Number of words (of decreasing average frequency) to include. Must also specifiy word file and index.", default=-1)
    parser.add_argument("--start-year", type=int, help="start year (inclusive)", default=START_YEAR)
    parser.add_argument("--end-year", type=int, help="start year (inclusive)", default=END_YEAR)
    parser.add_argument("--thresh", type=float, help="optional threshold", default=None)
    args = parser.parse_args()
    years = range(args.start_year, args.end_year + 1)
    if args.word_file != None:
        if args.index_dir == None:
            print >> sys.stderr, "Must specify index dir with word file!"
            sys.exit()
        word_pickle = ioutils.load_pickle(args.word_file)
        if not args.start_year in word_pickle:
            word_lists = {}
            for year in years:
                word_lists[year] = word_pickle
        else:
            word_lists = word_pickle
        word_infos = {}
        for year, word_list in word_lists.iteritems():
            year_index = ioutils.load_pickle(args.index_dir + "/" + str(year) + "-index.pkl") 
            if args.num_words != -1:
                word_list = word_list[:args.num_words]
            word_list, word_indices = get_word_indices(word_list, year_index)
            word_infos[year] = (word_list, word_indices)
        outpref ="/netstats/" + args.word_file.split("/")[-1].split(".")[0]
        if args.num_words != -1:
            outpref += "-top" + str(args.num_words)
    else:
        word_info = None
        outpref = "/netstats/net"
    if args.thresh != None:
        outpref += "-" + str(args.thresh)
    ioutils.mkdir(args.dir + "/netstats")
    run_parallel(args.num_procs, args.dir + outpref, args.dir + "/netstats/", args.dir + "/", years, word_info, args.thresh)       
